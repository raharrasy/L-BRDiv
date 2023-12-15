from Network import fc_network, LSTMNetwork
from torch import optim, nn
from torch.autograd import Variable
import numpy as np
import torch.distributions as dist
import torch
import torch.nn.functional as F

class AdhocAgent(object):
    """
        A class that encapsulates the  policy model controlling the ad hoc agent.
    """
    def __init__(self, obs_size, agent_o_size, configs, act_sizes, device, logger):
        """
            :param obs_size: The length of the entire observation inputted to the model
            :param agent_o_size: The length of u (graph-level features) for the model
            :param num_agents: Number of agents in env.
            :param configs: The dictionary config containing hyperparameters of the model
            :param act_sizes: Per-agent action space dimension
            :param device: Device to store model
            :param loss_writer: Tensorboard writer to log results
            :param model_grad_writer: Tensorboard writer to log model gradient magnitude
        """
        self.obs_size = obs_size
        self.agent_o_size = agent_o_size
        self.config = configs
        self.act_size = act_sizes
        self.device = device
        self.total_updates = 0

        policy_dimensions = [
            obs_size+self.config.model["agent_rep_size"], *self.config.model.actor_dims, self.act_size
        ]

        init_ortho = self.config.model.get("init_ortho", True)
        self.policy = fc_network(policy_dimensions, init_ortho).double().to(self.device)

        encoder_dimensions = [
            obs_size+self.act_size+1, *self.config.model.enc_dims, self.config.model["agent_rep_size"]
        ]
        self.encoder = LSTMNetwork(
            encoder_dimensions, init_ortho=init_ortho, device=self.device
        ).double()

        crit_dimensions = [
            obs_size + self.config.model["agent_rep_size"], *self.config.model.critic_dims, 1
        ]
        self.effective_crit_obs_size = obs_size
        self.action_value_function = fc_network(crit_dimensions, init_ortho).double().to(self.device)
        self.target_action_value_function = fc_network(crit_dimensions, init_ortho).double().to(self.device)

        self.hard_copy()

        # Initialize optimizer
        params_list = list(self.policy.parameters()) + list(self.action_value_function.parameters()) + list(self.encoder.parameters())
        self.optimizer = optim.Adam(
            params_list,
            lr=self.config.train["lr"]
        )

        self.logger = logger

    def to_one_hot(self, actions):
        """
            A method that changes agents' actions into a one-hot encoding format.
            :param actions: Agents' actions in form of an integer.
            :return: Agents' actions in a one-hot encoding form.
        """
        act_indices = np.asarray(actions).astype(int)
        one_hot_acts = np.eye(self.act_size)[act_indices]
        return one_hot_acts

    def get_teammate_representation(self, obs, prev_cell):
        """
            A method that computes teammate representation based on input.
        """
        input = torch.tensor(obs).double().to(self.device)[:, 0, :]
        batch_size = obs.shape[0]

        if prev_cell is None:
            prev_c0 = torch.zeros([batch_size, self.config.model["agent_rep_size"]]).double().to(self.device)
            prev_c1 = torch.zeros([batch_size, self.config.model["agent_rep_size"]]).double().to(self.device)
        else:
            prev_c0 = prev_cell[0]
            prev_c1 = prev_cell[1]

        agent_representation, c0, c1 = self.encoder(input, prev_c0, prev_c1)
        return agent_representation, (c0, c1)

    def decide_acts(self, obs_w_commands, with_log_probs=False, eval=False):
        """
            A method to decide the actions of agents given obs & target returns.
            :param obs_w_commands: A numpy array that has the obs concatenated with the target returns.
            :return: Sampled actions under the specific obs.
        """
        batch_size = obs_w_commands.size()[0]

        act_logits = self.policy(obs_w_commands)
        if not eval:
            particle_dist = dist.OneHotCategorical(logits=act_logits)
            original_acts = particle_dist.sample()
            acts = original_acts.argmax(dim=-1)
        else:
            particle_dist = dist.OneHotCategorical(logits=act_logits)
            original_acts = particle_dist.sample()
            acts = original_acts.argmax(dim=-1)

        acts_list = acts.tolist()

        if with_log_probs:
            return acts_list, torch.exp(particle_dist.log_prob(original_acts))
        return acts_list

    def onehot_from_logits(self, logits):
        """
            A method that converts action logits into actions under a one hot format.
        """
        argmax_acs = (logits == logits.max(-1, keepdim=True)[0]).double()

        return argmax_acs

    def compute_critic_loss(self, obs_batch, rewards_batch, done_batch, n_obs_batch, representations):
        """
            A method that computes the AHT agent's critic loss.
        """
        observation = obs_batch[:,:,0,:-self.act_size]
        n_observation = n_obs_batch[:,:,0,:-self.act_size]
        update_length = observation.size()[1]

        detached_representations = representations.detach()
        critic_input = torch.cat([observation, detached_representations[:,:-1,:]], dim=-1)
        target_critic_input = torch.cat([n_observation, detached_representations[:,1:,:]], dim=-1)

        predicted_value = []
        target_value = []

        bootstrapped_target = self.target_action_value_function(target_critic_input[:,-1,:])
        for t_id in reversed(range(update_length)):
            bootstrapped_target = rewards_batch[:,t_id].unsqueeze(-1) + self.config.train["gamma"] * \
                                  (1-done_batch[:,t_id].unsqueeze(-1))*bootstrapped_target

            target_value.append(
                bootstrapped_target.detach()
            )
            predicted_value.append(
                self.action_value_function(critic_input[:,t_id,:])
            )

        all_predicted_values = torch.cat(predicted_value, dim=-1)
        all_target_values = torch.cat(target_value, dim=-1)

        return (
            (all_predicted_values - all_target_values)**2
        ).mean()

    def compute_actor_and_entropy_loss(self, obs_batch, acts_batch, rewards_batch, done_batch, n_obs_batch, representations):
        """
            A method that computes the AHT agent's policy and entropy losses.
        """
        observation = obs_batch[:, :, 0, :-self.act_size]
        n_observation = n_obs_batch[:, :, 0, :-self.act_size]
        acts_executed = acts_batch[:, :, 0, :]
        update_length = observation.size()[1]

        detached_representations = representations
        bootstrap_input = torch.cat([n_observation[:, -1, :], detached_representations[:, -1, :]], dim=-1)
        baseline_input = torch.cat([observation, detached_representations[:, :-1, :]], dim=-1)

        predicted_value = []

        bootstrapped_value = self.action_value_function(bootstrap_input)
        for t_id in reversed(range(update_length)):
            bootstrapped_value = rewards_batch[:, t_id].unsqueeze(-1) + self.config.train["gamma"] * \
                                  (1 - done_batch[:, t_id].unsqueeze(-1)) * bootstrapped_value

            predicted_value.append(
                bootstrapped_value
            )

        all_predicted_values = torch.cat(list(reversed(predicted_value)), dim=-1)
        baseline_values = self.action_value_function(baseline_input).squeeze(-1)

        action_logits = self.policy(baseline_input)
        act_dist = dist.OneHotCategorical(logits=action_logits)
        act_log_likelihood = act_dist.log_prob(acts_executed)

        actor_loss = (-act_log_likelihood * (
            (all_predicted_values-baseline_values).detach()
        )).mean()

        entropy_loss = -act_dist.entropy().mean()

        return actor_loss, entropy_loss

    def update(self, batches, agent_representations):
        """
            A method that updates the policy model following sampled experiences.
            :param batches: A batch of obses and acts sampled from experience replay.
        """

        # Get obs and acts batch and prepare inputs to model.
        obs_batch, acts_batch = torch.tensor(batches[0]).to(self.device), torch.tensor(batches[1]).to(self.device)
        n_obs_batch = torch.tensor(batches[2]).to(self.device)
        done_batch = torch.tensor(batches[3]).double().to(self.device)
        rewards_batch = torch.tensor(batches[4]).double().to(self.device)

        batch_size, num_steps = obs_batch.size()[0], obs_batch.size()[1]

        # Prepare graph structure to GNNs
        # input_graph = self.create_graph(batch_size, num_agents)
        input_graph = None

        self.optimizer.zero_grad()


        # Compute SP Critic Loss
        total_critic_loss = self.compute_critic_loss(
            obs_batch, rewards_batch, done_batch, n_obs_batch, agent_representations
        )

        total_actor_loss, total_entropy_loss = self.compute_actor_and_entropy_loss(
            obs_batch, acts_batch, rewards_batch, done_batch, n_obs_batch, agent_representations
        )

        weighted_critic_loss = self.config.loss_weights["critic_loss_weight"] * total_critic_loss
        weighted_actor_loss = self.config.loss_weights["actor_loss_weight"] * total_actor_loss
        weighted_entropy_loss = self.config.loss_weights["entropy_regularizer_loss_weight"] * total_entropy_loss

        # Write losses to logs
        if self.total_updates % self.logger.train_log_period == 0:
            train_step = self.total_updates * self.logger.steps_per_update
            self.logger.log_item("Train/adhoc/critic_loss", total_critic_loss,
                                 train_step=train_step, updates=self.total_updates)
            self.logger.log_item("Train/adhoc/actor_loss", total_actor_loss,
                                 train_step=train_step, updates=self.total_updates)
            self.logger.log_item("Train/adhoc/entropy_loss", total_entropy_loss,
                                 train_step=train_step, updates=self.total_updates)
            self.logger.commit()

        # Backpropagate critic loss
        (weighted_critic_loss+weighted_actor_loss+weighted_entropy_loss).backward()

        # Clip grads if necessary
        if self.config.train['max_grad_norm'] > 0:
            nn.utils.clip_grad_norm_(self.action_value_function.parameters(), self.config.train['max_grad_norm'])
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.train['max_grad_norm'])
            nn.utils.clip_grad_norm_(self.encoder.parameters(), self.config.train['max_grad_norm'])

        # Log grad magnitudes if specified by config.
        if self.config.logger["log_grad"]:

            for name, param in self.policy.named_parameters():
                if not param.grad is None:
                    self.logger.log_item(
                        f"Train/grad/actor_{name}",
                        torch.abs(param.grad).mean(),
                        train_step=self.total_updates
                        )

            for name, param in self.action_value_function.named_parameters():
                if not param.grad is None:
                    self.logger.log_item(
                        f"Train/grad/critic_{name}",
                        torch.abs(param.grad).mean(),
                        train_step=self.total_updates
                        )

            for name, param in self.encoder.named_parameters():
                if not param.grad is None:
                    self.logger.log_item(
                        f"Train/grad/encoder_{name}",
                        torch.abs(param.grad).mean(),
                        train_step=self.total_updates
                        )

        self.optimizer.step()
        self.soft_copy(self.config.train["target_update_rate"])

        self.total_updates += 1

    def hard_copy(self):
        for target_param, param in zip(self.target_action_value_function.parameters(), self.action_value_function.parameters()):
            target_param.data.copy_(param.data)

    def soft_copy(self, tau=0.001):
        """
            A method that updates target networks based on the recent value of the critic network.
        """
        for target_param, param in zip(self.target_action_value_function.parameters(), self.action_value_function.parameters()):
            target_param.data.copy_(tau * param + (1 - tau) * target_param)

    def save_model(self, int_id, save_model=False):
        """
            A method to save model parameters.
            :param int_id: Integer indicating ID of the checkpoint.
        """
        if not save_model:
            return

        torch.save(self.policy.state_dict(),
                    f"adhoc_model/model_{int_id}.pt")

        torch.save(self.action_value_function.state_dict(),
                   f"adhoc_model/model_{int_id}-action-value.pt")

        torch.save(self.target_action_value_function.state_dict(),
                   f"adhoc_model/model_{int_id}-target-action-value.pt")

        torch.save(self.encoder.state_dict(),
                   f"adhoc_model/model_{int_id}-encoder.pt")

        torch.save(self.optimizer.state_dict(),
                   f"adhoc_model/model_{int_id}-optim.pt")

    def load_model(self, int_id):
        """
            A method that loads stored neural network models to be used by an AHT agent
        """

        self.policy.load_state_dict(
            torch.load(f"{self.config.run['load_dir']}/adhoc_model/model_{int_id}.pt")
        )

        self.action_value_function.load_state_dict(
            torch.load(f"{self.config.run['load_dir']}/adhoc_model/model_{int_id}-action-value.pt")
        )

        self.target_action_value_function.load_state_dict(
                torch.load(f"{self.config.run['load_dir']}/adhoc_model/model_{int_id}-target-action-value.pt")
        )

        self.encoder.load_state_dict(
            torch.load(f"{self.config.run['load_dir']}/adhoc_model/model_{int_id}-encoder.pt")
        )

        self.optimizer.load_state_dict(
            torch.load(f"{self.config.run['load_dir']}/adhoc_model/model_{int_id}-optim.pt")
        )
