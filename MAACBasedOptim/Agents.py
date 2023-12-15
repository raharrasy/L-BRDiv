from Network import fc_network
from torch import optim, nn
from torch.autograd import Variable
import numpy as np
import torch.distributions as dist
import torch
import torch.nn.functional as F
import math

class Agents(object):
    """
        A class that encapsulates the joint policy model controlling each agent population.
    """
    def __init__(self, obs_size, agent_o_size, num_agents, num_populations, configs, act_sizes, device, logger, mode="train"):
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
        self.next_log_update = 0
        self.next_log_update_lagrange = 0
        self.total_updates_lagrange = 0
        self.num_agents = num_agents
        self.with_any_play = self.config.any_play.get("with_any_play", False)
        self.mode = mode
        self.id_length = self.config.populations["num_populations"]
        self.num_populations = self.config.populations["num_populations"]
        self.effective_crit_obs_size = obs_size + self.num_agents

        total_checkpoints = self.config.run["total_checkpoints"]
        timesteps_per_checkpoint = self.config.run["num_timesteps"]//(total_checkpoints*(self.config.env.parallel["sp_collection"]+self.config.env.parallel["xp_collection"]))
        self.projected_total_updates = total_checkpoints * math.ceil((timesteps_per_checkpoint+0.0)/self.config.train["timesteps_per_update"])

        actor_dims = [self.obs_size-self.num_populations, *self.config.model.actor_dims, self.act_size]
        critic_dims = [self.num_agents * (self.obs_size + self.num_agents), *self.config.model.critic_dims, 1]

        if self.with_any_play:
            classifier_dims = [
                self.num_agents * (self.obs_size - self.num_populations), *self.config.any_play.classifier_dims,
                self.num_populations
            ]
        init_ortho = self.config.model.get("init_ortho", True)

        self.joint_policy = [
            fc_network(actor_dims, init_ortho).double().to(self.device)
            for _ in range(2*self.num_populations)
        ]

        self.old_joint_policy = [
            fc_network(actor_dims, init_ortho).double().to(self.device)
            for _ in range(2 * self.num_populations)
        ]

        for pol, old_pol in zip(self.joint_policy, self.old_joint_policy):
            for target_param, param in zip(old_pol.parameters(), pol.parameters()):
                target_param.data.copy_(param.data)

        self.joint_action_value_functions = [fc_network(critic_dims, init_ortho).double().to(self.device) for _ in range(self.num_agents)]

        lagrange_mat1 = self.config.train["init_lagrange"]*torch.ones([self.config.populations["num_populations"], self.config.populations["num_populations"]]).double().to(self.device)
        lagrange_mat1.fill_diagonal_(1)

        self.lagrange_multiplier_matrix1 = Variable(lagrange_mat1.data, requires_grad=False)

        self.normalizer1 = self.compute_const_lagrange1().mean(dim=-1, keepdim=False)
        self.target_joint_action_value_functions = [fc_network(critic_dims, init_ortho).double().to(self.device) for _ in range(self.num_agents)]

        if self.with_any_play:
            self.pop_classifier = fc_network(classifier_dims, init_ortho).double().to(self.device)

        self.hard_copy()

        # Initialize optimizer
        params_list = None
        critic_params_list = [
            *(param for critic in self.joint_action_value_functions for param in critic.parameters())
        ]

        actors_params_list = [
            *(param for actor in self.joint_policy for param in actor.parameters())
        ]

        self.critic_optimizer = optim.Adam(
            critic_params_list,
            lr=self.config.train["lr"]
        )

        self.actor_optimizer = optim.Adam(
            actors_params_list,
            lr=self.config.train["lr"]
        )

        self.logger = logger

    def compute_const_lagrange1(self):
        """
        A method that returns Lagrange multipliers related to first constraints
        :return: \alpha^{1}
        """
        return F.relu(self.lagrange_multiplier_matrix1)

    def to_one_hot_population_id(self, indices):
        """
        A method that changes population ID to a one hot encoding format
        :param indices: Population ID
        :return: one_hot_ids: Population ID in one hot format.
        """
        act_indices = np.asarray(indices).astype(int)
        one_hot_ids = np.eye(self.num_populations)[act_indices]

        return one_hot_ids

    def to_one_hot(self, actions):
        """
            A method that changes agents' actions into a one-hot encoding format.
            :param actions: Agents' actions in form of an integer.
            :return: Agents' actions in a one-hot encoding form.
        """
        act_indices = np.asarray(actions).astype(int)
        one_hot_acts = np.eye(self.act_size)[act_indices]
        return one_hot_acts

    def separate_act_select(self, input):
        """
        A method that quickly computes all policies' actions even if they have separate parameters
        :param input: Input vector
        :return: logits: Action logits
        """
        additional_input_length = self.num_agents
        per_id_input = [None] * (2 * self.num_populations)
        for pop_id in range(self.num_populations):
            for a_id in range(self.num_agents):
                per_id_input[2*pop_id+a_id] = input[
                    torch.logical_and(
                        input[:, :, -(self.num_populations + additional_input_length) + pop_id] == 1,
                        input[:, :, -self.num_agents + a_id] == 1
                    )
                ][:, :-(self.id_length+additional_input_length)]

        per_id_input_filtered = [(idx,inp) for idx, inp in enumerate(per_id_input) if not inp.nelement() == 0]
        executed_models = [policy for idx, policy in enumerate(self.joint_policy) if
                           not per_id_input[idx].nelement() == 0]

        futures = [
            torch.jit.fork(model, per_id_input_filtered[i][1]) for i, model
            in enumerate(executed_models)
        ]

        results = [torch.jit.wait(fut) for fut in futures]
        logits = torch.zeros([input.size()[0], input.size()[1], self.act_size]).double().to(self.device)

        id = 0
        for idx, _ in per_id_input_filtered:
            population_id = idx // 2
            agent_id = idx % 2
            logits[
                torch.logical_and(
                    input[:, :, -(self.num_populations + additional_input_length) + population_id] == 1,
                    input[:, :, -self.num_agents + agent_id] == 1
                )
            ] = results[id]
            id += 1

        return logits

    def separate_act_select_old(self, input):
        additional_input_length = self.num_agents
        per_id_input = [None] * (2 * self.num_populations)
        for pop_id in range(self.num_populations):
            for a_id in range(self.num_agents):
                per_id_input[2*pop_id+a_id] = input[
                    torch.logical_and(
                        input[:, :, -(self.num_populations + additional_input_length) + pop_id] == 1,
                        input[:, :, -self.num_agents + a_id] == 1
                    )
                ][:, :-(self.id_length+additional_input_length)]

        per_id_input_filtered = [(idx,inp) for idx, inp in enumerate(per_id_input) if not inp.nelement() == 0]
        executed_models = [policy for idx, policy in enumerate(self.old_joint_policy) if
                           not per_id_input[idx].nelement() == 0]

        futures = [
            torch.jit.fork(model, per_id_input_filtered[i][1]) for i, model
            in enumerate(executed_models)
        ]

        results = [torch.jit.wait(fut) for fut in futures]
        logits = torch.zeros([input.size()[0], input.size()[1], self.act_size]).double().to(self.device)

        id = 0
        for idx, _ in per_id_input_filtered:
            population_id = idx // 2
            agent_id = idx % 2
            logits[
                torch.logical_and(
                    input[:, :, -(self.num_populations + additional_input_length) + population_id] == 1,
                    input[:, :, -self.num_agents + agent_id] == 1
                )
            ] = results[id]
            id += 1

        return logits

    def decide_acts(self, obs_w_commands, with_log_probs=False, eval=False):
        """
            A method to decide the actions of agents given obs & target returns.
            :param obs_w_commands: A numpy array that has the obs concatenated with the target returns.
            :return: Sampled actions under the specific obs.
        """
        obs_w_commands = torch.tensor(obs_w_commands).to(self.device)
        batch_size, num_agents = obs_w_commands.size()[0], obs_w_commands.size()[1]

        in_population_agent_id = torch.eye(num_agents).repeat(batch_size, 1, 1).to(self.device)
        obs_w_commands = torch.cat([obs_w_commands, in_population_agent_id], dim=-1)

        act_logits = self.separate_act_select(obs_w_commands)
        if not eval:
            particle_dist = dist.OneHotCategorical(logits=act_logits)
            original_acts = particle_dist.sample()
            acts = original_acts.argmax(dim=-1)
        else:
            particle_dist = dist.Categorical(logits=act_logits)
            original_acts = torch.argmax(act_logits, dim=-1)
            acts = original_acts

        acts_list = acts.tolist()

        if with_log_probs:
            return acts_list, torch.exp(particle_dist.log_prob(original_acts))
        return acts_list

    def compute_pop_class_loss(self, obs_batch):
        """
        Unused method. Legacy from BRDiv to compute losses for their Any-Play baseline.
        :param obs_batch:
        :return:
        """

        real_input = obs_batch[:,:,:,:-self.num_populations].reshape(obs_batch.size()[0] * obs_batch.size()[1], -1)
        output_classes = obs_batch[:,:,:1,-self.num_populations:].reshape(obs_batch.size()[0] * obs_batch.size()[1], -1)

        logits = self.pop_classifier(real_input)
        pop_categorical_dist = dist.OneHotCategorical(logits=logits)

        return -pop_categorical_dist.log_prob(output_classes).mean()

    def compute_jsd_loss(self, obs_batch, acts_batch):
        """
            Unused method. Legacy from BRDiv to compute the optimized metric for TrajeDi.
            :param obs_batch:
            :return:
        """
        comparator_prob = None
        batch_size, num_steps, num_agents = obs_batch.size()[0], obs_batch.size()[1], obs_batch.size()[2]

        action_probs_per_population = []
        agent_real_ids = obs_batch[:, :, :, self.obs_size - self.num_populations: self.obs_size].argmax(dim=-1)

        for idx in range(self.num_populations):
            original_states = obs_batch[:, :, :, :self.obs_size - self.num_populations]
            original_states = original_states.view(batch_size*num_steps, num_agents, -1)

            pop_annot = torch.zeros_like(
                obs_batch.view(
                    batch_size*num_steps, num_agents, -1
                )[:, :, self.obs_size - self.num_populations:self.obs_size]).double().to(self.device)
            pop_annot[:, :, idx] = 1.0

            comparator_input = torch.cat([original_states, pop_annot, torch.eye(self.num_agents).repeat(original_states.size()[0],1,1).to(self.device)], dim=-1)

            comparator_act_logits = self.separate_act_select(comparator_input)
            comparator_act_logits = comparator_act_logits.view(batch_size, num_steps, num_agents, -1)

            action_logits = dist.OneHotCategorical(logits=comparator_act_logits).log_prob(acts_batch)
            action_probs_per_population.append(action_logits.unsqueeze(-1))

            if comparator_prob is None:
                comparator_prob = torch.exp(action_logits)
            else:
                comparator_prob = comparator_prob + torch.exp(action_logits)

        action_logits_per_population = torch.cat(action_probs_per_population, dim=-1)

        temp_pi= torch.gather(action_logits_per_population, -1, agent_real_ids.unsqueeze(-1)).squeeze(-1).sum(dim=-1)
        log_pi_i = temp_pi.sum(dim=-1)

        temp_pi_hat = action_logits_per_population.sum(dim=-2).sum(dim=-2)
        log_pi_hat = torch.log(torch.exp(temp_pi_hat).mean(dim=-1))

        summed_term_list = []
        separate_delta_list = []
        per_pop_delta = []
        for t in range(num_steps):
            multiplier = self.config.train["gamma_act_jsd"]**(torch.abs(t-torch.tensor(list(range(num_steps))).to(self.device)))
            multiplier = multiplier.unsqueeze(0).repeat(batch_size,1)
            delta_hat_var = action_logits_per_population.sum(dim=-2)
            separate_deltas = (temp_pi * multiplier).sum(dim=-1)
            log_average_only_delta = (delta_hat_var * multiplier.unsqueeze(-1).repeat(1, 1, self.num_populations)).sum(dim=-2)
            per_pop_delta.append(log_average_only_delta.unsqueeze(-2))
            average_only_delta = torch.log(torch.exp(log_average_only_delta).mean(dim=-1))

            separate_delta_list.append(separate_deltas.unsqueeze(-1))
            summed_term_list.append(average_only_delta.unsqueeze(-1))

        stacked_summed_term_list = torch.cat(summed_term_list, dim=-1)
        stacked_separate_delta_list = torch.cat(separate_delta_list, dim=-1)
        stacked_pop_delta = torch.cat(per_pop_delta, dim=-2)
        repeated_stacked_summed_term_list = stacked_summed_term_list.unsqueeze(-1).repeat(1, 1, self.num_populations)

        calculated_logs = repeated_stacked_summed_term_list - stacked_pop_delta
        calc_logs_mean_ovt = calculated_logs.mean(dim=-2)

        is_ratio = torch.exp(temp_pi_hat - log_pi_i.unsqueeze(-1).repeat(1, self.num_populations))
        jsd_loss = (is_ratio.detach() * calc_logs_mean_ovt).mean(dim=-1).mean()

        return jsd_loss

    def onehot_from_logits(self, logits):
        """
        Method to change action logits into actions under a one hot format.
        :param logits: Action logits
        :return: argmax_acs: Action under one-hot format
        """
        argmax_acs = (logits == logits.max(-1, keepdim=True)[0]).double()
        return argmax_acs

    # TODO
    def compute_sp_advantages(self, obs, n_obs, acts, dones, rews):
        """
        A function that computes the weighted advantage function as described in Expression 14
        :param obs: Agent observation
        :param n_obs: Agent next observation
        :param acts: Agent action
        :param dones: Agent done flag
        :param rews: Agent reward flag
        :return opt_diversity_values.detach() - baseline_diversity_values.detach(): weighted advantage function
        :return all_entropy_weight1.detach(): Variable entropy weights given \alpha^1
        :return all_entropy_weight1.detach(): Variable entropy weights given \alpha^2
        """
        batch_size = obs.size()[0]
        obs_length = obs.size()[1]
        obs_only_length = self.effective_crit_obs_size - self.num_populations - self.num_agents

        # Initialize empty lists that saves every important data for actor loss computation
        baseline_xp_matrics1 = []
        opt_xp_matrics1 = []
        baseline_xp_matrics2 = []
        opt_xp_matrics2 = []
        lagrangian_matrices11 = []
        lagrangian_matrices12 = []
        entropy_weight1 = []
        
        lagrange_matrix_mean_norm1 = self.compute_const_lagrange1().mean(dim=-1, keepdim=False)
        
        pos_entropy_weights1 = (lagrange_matrix_mean_norm1/self.normalizer1) * self.config.loss_weights["entropy_regularizer_loss"]

        # Initialize target diversity at end of episode to None
        target_diversity_value = None
        for idx in reversed(range(obs_length)):
            obs_idx = torch.cat(
                [obs[:, idx, :, :], torch.eye(self.num_agents).repeat(batch_size, 1, 1).to(self.device)], dim=-1)
            n_obs_idx = torch.cat(
                [n_obs[:, idx, :, :], torch.eye(self.num_agents).repeat(batch_size, 1, 1).to(self.device)],
                dim=-1)
            sp_rl_rew1 = rews[:, idx, 0]
            sp_rl_rew2 = rews[:, idx, 1]
            sp_rl_done = dones[:, idx]

            if idx == obs_length - 1:
                target_diversity_value1 = self.joint_action_value_functions[0](
                    n_obs_idx.view(n_obs_idx.size(0), 1, -1).squeeze(1)
                )

                target_diversity_value2 = self.joint_action_value_functions[1](
                    n_obs_idx.view(n_obs_idx.size(0), 1, -1).squeeze(1)
                )

            lagrangian_matrix11 = self.compute_const_lagrange1().detach().clone()
            lagrangian_matrix11.fill_diagonal_(0.0)
            lagrangian_matrix11 = lagrangian_matrix11.unsqueeze(0).repeat(batch_size, 1, 1)
            
            lagrangian_matrix12 = self.compute_const_lagrange1().detach().clone()
            lagrangian_matrix12.fill_diagonal_(1.0)
            lagrangian_matrix12 = lagrangian_matrix12.unsqueeze(0).repeat(batch_size, 1, 1)

            offset = self.num_populations + self.num_agents
            if -offset + self.num_populations == 0:
                accessed_index = obs_idx[:, :, -offset:].argmax(dim=-1)
            else:
                accessed_index = obs_idx[:, :, -offset:-offset + self.num_populations].argmax(dim=-1)

            idx1, idx2 = accessed_index[:, 0], accessed_index[:, 1]

            target_diversity_value1 = (
                    sp_rl_rew1.view(-1, 1) + (
                    self.config.train["gamma"] * (1 - sp_rl_done.view(-1, 1)) * target_diversity_value1)
            ).detach()

            target_diversity_value2 = (
                    sp_rl_rew2.view(-1, 1) + (
                    self.config.train["gamma"] * (1 - sp_rl_done.view(-1, 1)) * target_diversity_value2)
            ).detach()

            baseline_diversity_values1 = self.joint_action_value_functions[0](
                obs_idx.view(obs_idx.size(0), 1, -1).squeeze(1)
            )

            baseline_diversity_values2 = self.joint_action_value_functions[1](
                obs_idx.view(obs_idx.size(0), 1, -1).squeeze(1)
            )

            opt_xp_matrics1.append(target_diversity_value1)
            baseline_xp_matrics1.append(baseline_diversity_values1)
            opt_xp_matrics2.append(target_diversity_value2)
            baseline_xp_matrics2.append(baseline_diversity_values2)

            entropy_weight1.append(pos_entropy_weights1[idx1])

            lagrangian_matrices11.append(lagrangian_matrix11[torch.arange(obs_idx.size(0)), idx1, :])
            lagrangian_matrices12.append(lagrangian_matrix12[torch.arange(obs_idx.size(0)), idx1, :])

        # Combine lists into a single tensor for actor loss from SP data
        all_baseline_matrices1 = torch.cat(baseline_xp_matrics1, dim=0)
        all_opt_matrices1 = torch.cat(opt_xp_matrics1, dim=0)
        all_baseline_matrices2 = torch.cat(baseline_xp_matrics2, dim=0)
        all_opt_matrices2 = torch.cat(opt_xp_matrics2, dim=0)
        all_lagrangian_matrices11 = torch.cat(lagrangian_matrices11, dim=0)
        all_lagrangian_matrices12 = torch.cat(lagrangian_matrices12, dim=0)
        all_entropy_weight1 = torch.cat(entropy_weight1, dim=0)
        
        weights1 = torch.sum(all_lagrangian_matrices11, dim=-1)
        weights2 = torch.sum(all_lagrangian_matrices12, dim=-1)
        baseline_diversity_values1 = torch.ones_like(weights1)*all_baseline_matrices1.squeeze(1) + weights1 * all_baseline_matrices2.squeeze(1)
        opt_diversity_values1 = torch.ones_like(weights1)*all_opt_matrices1.squeeze(1) + weights1 * all_opt_matrices2.squeeze(1)
        baseline_diversity_values2 = weights2 * all_baseline_matrices2.squeeze(1)
        opt_diversity_values2 = weights2 * all_opt_matrices2.squeeze(1)

        return opt_diversity_values1.detach() - baseline_diversity_values1.detach(), opt_diversity_values2.detach() - baseline_diversity_values2.detach(), all_entropy_weight1.detach(), all_entropy_weight2.detach()

    def compute_sp_old_probs(self, obs, n_obs, acts, dones, rews):
        """
            A function that computes the previous policy's (before update) probability of selecting an action.  Required for MAPPO update.
            :param obs: Agent observation
            :param n_obs: Agent next observation
            :param acts: Agent action
            :param dones: Agent done flag
            :param rews: Agent reward flag
            :return old_log_likelihoods: Previous policies' action log probability
        """
        batch_size = obs.size()[0]
        obs_length = obs.size()[1]
        obs_only_length = self.effective_crit_obs_size - self.num_populations - self.num_agents

        # Initialize empty lists that saves every important data for actor loss computation
        action_likelihood = []
        action_entropy = []

        for idx in reversed(range(obs_length)):
            acts_idx = acts[:, idx, :, :]
            obs_idx = torch.cat(
                [obs[:, idx, :, :], torch.eye(self.num_agents).repeat(batch_size, 1, 1).to(self.device)], dim=-1)

            action_logits = self.separate_act_select_old(obs_idx)
            action_distribution = dist.OneHotCategorical(logits=action_logits)
            action_likelihood.append(action_distribution.log_prob(acts_idx))

        # Combine lists into a single tensor for actor loss from SP data
        old_log_likelihoods = torch.cat(action_likelihood, dim=0).sum(dim=-1)
        return old_log_likelihoods

    def compute_sp_actor_loss(self, obs, n_obs, acts, dones, rews, advantages1, advantages2, old_log_likelihoods, entropy_weight1, entropy_weight2):
        """
            A function that computes the policy's loss function based on self-play interaction
            :param obs: Agent observation
            :param n_obs: Agent next observation
            :param acts: Agent action
            :param dones: Agent done flag
            :param rews: Agent reward flag
            :param advantages: Weighted advantage value
            :param old_log_likelihoods: Previous policies log likelihood
            :param entropy_weight1: Variable entropy weights based on \alpha^1
            :param entropy_weight2: Variable entropy weights based on \alpha^2
            :return pol_loss: Policy loss
        """
        batch_size = obs.size()[0]
        obs_length = obs.size()[1]
        obs_only_length = self.effective_crit_obs_size - self.num_populations - self.num_agents

        # Initialize empty lists that saves every important data for actor loss computation
        action_likelihood = []
        action_entropy = []

        anneal_end = self.config.train["anneal_end"] * self.projected_total_updates
        for idx in reversed(range(obs_length)):
            acts_idx = acts[:, idx, :, :]
            obs_idx = torch.cat([obs[:, idx, :, :], torch.eye(self.num_agents).repeat(batch_size, 1, 1).to(self.device)], dim=-1)

            action_logits = self.separate_act_select(obs_idx)
            action_distribution = dist.OneHotCategorical(logits=action_logits)

            # Append computed measures to their lists
            action_likelihood.append(action_distribution.log_prob(acts_idx))
            action_entropy.append(action_distribution.entropy())

        # Combine lists into a single tensor for actor loss from SP data
        action_entropies = torch.cat(action_entropy, dim=0)
        entropy_weights = torch.cat([
            entropy_weight1.unsqueeze(dim=-1), entropy_weight2.unsqueeze(dim=-1)
        ], dim=-1)
        action_log_likelihoods = torch.cat(action_likelihood, dim=0)
        action_log_likelihoods1 = action_log_likelihoods[:,0] + (action_log_likelihoods[:,1].detach())
        action_log_likelihoods2 = action_log_likelihoods[:,1] + (action_log_likelihoods[:,0].detach())
        entropy_loss = (entropy_weights * -action_entropies).sum(dim=-1).mean()

        ratio1 = torch.exp(action_log_likelihoods1 - old_log_likelihoods.detach())
        ratio2 = torch.exp(action_log_likelihoods2 - old_log_likelihoods.detach())

        surr11 = ratio1 * advantages1
        surr21 = torch.clamp(
            ratio1,
            1 - self.config.train["eps_clip"],
            1 + self.config.train["eps_clip"]
        ) * advantages1

        surr12 = ratio2 * advantages2
        surr22 = torch.clamp(
            ratio2,
            1 - self.config.train["eps_clip"],
            1 + self.config.train["eps_clip"]
        ) * advantages2

        pol_list1 = torch.min(surr11, surr21)
        pol_list2 = torch.min(surr12, surr22)
        if self.config.train["with_dual_clip"]:
            pol_list1[advantages1 < 0] = torch.max(pol_list1[advantages1 < 0], self.config.train["dual_clip"]*advantages1[advantages1 < 0])
            pol_list2[advantages2 < 0] = torch.max(pol_list2[advantages2 < 0], self.config.train["dual_clip"]*advantages2[advantages2 < 0])
        pol_loss = -(pol_list1.mean() + pol_list2.mean())

        return pol_loss, entropy_loss

    def compute_sp_lagrange_loss(self, obs, n_obs, acts, dones, rews):
        """
            A function that computes the lagrange multiplier losses based on self-play interaction
            :param obs: Agent observation
            :param n_obs: Agent next observation
            :param acts: Agent action
            :param dones: Agent done flag
            :param rews: Agent reward flag
            :return Lagrange multiplier loss
        """
        batch_size = obs.size()[0]
        obs_length = obs.size()[1]
        obs_only_length = self.effective_crit_obs_size - self.num_populations - self.num_agents

        # Initialize empty lists that saves every important data for actor loss computation
        diff_matrix1, diff_matrix2 = [], []
        saved_id11 = []
        saved_id21 = []

        # Initialize target diversity at end of episode to None
        target_diversity_value = None
        for idx in reversed(range(obs_length)):
            obs_idx = torch.cat([obs[:, idx, :, :], torch.eye(self.num_agents).repeat(batch_size, 1, 1).to(self.device)], dim=-1)
            n_obs_idx = torch.cat([n_obs[:, idx, :, :], torch.eye(self.num_agents).repeat(batch_size, 1, 1).to(self.device)],
                                  dim=-1)
            sp_rl_rew = rews[:, idx, 1]
            sp_rl_done = dones[:, idx]

            if idx == obs_length - 1:
                target_diversity_value = self.joint_action_value_functions[1](
                    n_obs_idx.view(n_obs_idx.size(0), 1, -1).squeeze(1)
                )

            offset = self.num_populations + self.num_agents
            if -offset + self.num_populations == 0:
                accessed_index = obs_idx[:, :, -offset:].argmax(dim=-1)
            else:
                accessed_index = obs_idx[:, :, -offset:-offset + self.num_populations].argmax(dim=-1)

            idx1 = accessed_index[:, 0]
            repeated_idx1 = torch.repeat_interleave(idx1, self.num_populations, 0)
            repeated_idx1_final = torch.eye(self.num_populations).to(self.device)[repeated_idx1].unsqueeze(1)
            added_tensors1 = torch.eye(self.num_populations).to(self.device).repeat([batch_size,1]).unsqueeze(1)

            combined_tensors1 = torch.cat([repeated_idx1_final, added_tensors1], dim=1)

            obs_only = obs_idx[:, :, :obs_only_length]
            r_obs_only = obs_only.repeat([1, self.num_populations, 1]).view(
                -1, obs_only.size()[-2], obs_only.size()[-1]
            )

            eval_input1 = torch.cat([r_obs_only, combined_tensors1, torch.eye(self.num_agents).repeat(r_obs_only.size()[0], 1, 1).to(self.device)], dim=-1)
            
            baseline_matrix1 = self.joint_action_value_functions[1](
                eval_input1.view(eval_input1.size(0), 1, -1).squeeze(1)
            ).view(batch_size, self.num_populations)

            target_diversity_value = (
                    sp_rl_rew.view(-1, 1) + (
                        self.config.train["gamma"] * (1 - sp_rl_done.view(-1, 1)) * target_diversity_value)
            ).detach()

            resized_target_values1 = target_diversity_value.repeat(1, self.num_populations)
            resized_target_values1[torch.arange(baseline_matrix1.size()[0]), idx1] = baseline_matrix1[torch.arange(baseline_matrix1.size()[0]), idx1] + self.config.train["tolerance_factor"]

            # Append computed measures to their lists
            diff_matrix1.append(resized_target_values1 - baseline_matrix1 - self.config.train["tolerance_factor"])
            
            saved_id11.append(torch.repeat_interleave(idx1, self.num_populations))
            saved_id21.append(torch.arange(self.num_populations).repeat(idx1.size()[0]).to(self.device))
            
        # Combine lists into a single tensor for actor loss from SP data
        all_diff1 = torch.cat(diff_matrix1, dim=0).detach()
        all_id11 = torch.cat(saved_id11, dim=0)
        all_id21 = torch.cat(saved_id21, dim=0)
        
        return (all_diff1, all_id11, all_id21,), ((self.compute_const_lagrange1()**2).mean()**0.5)

    def compute_xp_lagrange_loss(self, xp_obs, xp_acts, xp_rews, xp_dones, xp_n_obses):
        """
            A function that computes the lagrange multiplier losses based on cross-play interaction
            :param obs: Agent observation
            :param n_obs: Agent next observation
            :param acts: Agent action
            :param dones: Agent done flag
            :param rews: Agent reward flag
            :return Lagrange multiplier loss
        """
        batch_size = xp_obs.shape[0]
        obs_length = xp_obs.shape[1]
        xp_num_agents = xp_obs.shape[2]
        obs_only_length = self.effective_crit_obs_size - self.num_populations - self.num_agents

        diff_matrix1, diff_matrix2 = [], []
        saved_id11, saved_id12 = [], []
        saved_id21, saved_id22 = [], []

        xp_targ_diversity_value = None
        for idx in reversed(range(obs_length)):
            xp_obs_id = torch.cat(
                [xp_obs[:, idx, :, :], torch.eye(self.num_agents).repeat([batch_size, 1, 1]).to(self.device)], dim=-1
            )

            if idx == obs_length - 1:
                xp_n_obs_inp = torch.cat([xp_n_obses[:, idx, :, :], torch.eye(self.num_agents).repeat(xp_obs_id.size()[0], 1, 1).to(self.device)], dim=-1)
                xp_targ_diversity_value = self.joint_action_value_functions[1](
                    xp_n_obs_inp.view(xp_n_obs_inp.size(0), 1, -1).squeeze(1)
                )

            xp_rl_rew = xp_rews[:, idx, 1]
            xp_rl_done = xp_dones[:, idx]

            xp_obs_only = xp_obs_id[:, :, :obs_only_length]

            offset = self.num_populations + self.num_agents
            if -offset + self.num_populations == 0:
                accessed_index = xp_obs_id[:, :, -offset:].argmax(dim=-1)
            else:
                accessed_index = xp_obs_id[:, :, -offset:-offset + self.num_populations].argmax(dim=-1)

            idx1, idx2 = accessed_index[:, 0], accessed_index[:, 1]
            repeated_idx1 = torch.repeat_interleave(idx1, 2, 0)
            repeated_idx2 = torch.repeat_interleave(idx2, 2, 0)
            added_matrix1 = torch.eye(self.num_populations).to(self.device)[repeated_idx1].view(batch_size, -1, self.num_populations)
            added_matrix2 = torch.eye(self.num_populations).to(self.device)[repeated_idx2].view(batch_size, -1, self.num_populations)

            xp_input1 = torch.cat([xp_obs_only, added_matrix1, torch.eye(self.num_agents).repeat(xp_obs_only.size()[0], 1, 1).to(self.device)], dim=-1)
            xp_input2 = torch.cat([xp_obs_only, added_matrix2, torch.eye(self.num_agents).repeat(xp_obs_only.size()[0], 1, 1).to(self.device)], dim=-1)

            baseline_matrix1 = self.joint_action_value_functions[1](
                xp_input1.view(xp_input1.size(0), 1, -1).squeeze(1),
            )

            baseline_matrix2 = self.joint_action_value_functions[1](
                xp_input2.view(xp_input2.size(0), 1, -1).squeeze(1),
            )


            xp_targ_diversity_value = (
                    xp_rl_rew.view(-1, 1) + (
                    self.config.train["gamma"] * (1 - xp_rl_done.view(-1, 1)) * xp_targ_diversity_value)
            ).detach()

            diff_matrix1.append(baseline_matrix1 - xp_targ_diversity_value - self.config.train["tolerance_factor"])
            diff_matrix2.append(baseline_matrix2 - xp_targ_diversity_value - self.config.train["tolerance_factor"])

            saved_id11.append(idx1)
            saved_id21.append(idx2)
            saved_id12.append(idx2)
            saved_id22.append(idx1)

        all_diff1 = torch.cat(diff_matrix1, dim=0).detach()
        all_diff2 = torch.cat(diff_matrix2, dim=0).detach()
        all_id11 = torch.cat(saved_id11, dim=0)
        all_id21 = torch.cat(saved_id21, dim=0)
        all_id12 = torch.cat(saved_id12, dim=0)
        all_id22 = torch.cat(saved_id22, dim=0)

        return (all_diff1, all_diff2, all_id11, all_id21, all_id12, all_id22), ((self.compute_const_lagrange1()**2).mean()**0.5), ((self.compute_const_lagrange2()**2).mean()**0.5)

    def compute_sp_critic_loss(
            self, obs_batch, n_obs_batch,
            acts_batch, sp_rew_batch, sp_done_batch
    ):
        """
            A function that computes critic loss based on self-play interaction
            :param obs: Agent observation
            :param n_obs: Agent next observation
            :param acts: Agent action
            :param dones: Agent done flag
            :param rews: Agent reward flag
            :return Lagrange multiplier loss
        """
        batch_size = obs_batch.size()[0]
        obs_length = obs_batch.size()[1]

        predicted_values1 = []
        target_values1 = []
        target_value1 = None

        predicted_values2 = []
        target_values2 = []
        target_value2 = None
        for idx in reversed(range(obs_length)):
            obs_idx = torch.cat([obs_batch[:,idx,:,:], torch.eye(self.num_agents).repeat(batch_size,1,1).to(self.device)], dim=-1)
            n_obs_idx = torch.cat([n_obs_batch[:,idx,:,:], torch.eye(self.num_agents).repeat(batch_size, 1, 1).to(self.device)], dim=-1)

            sp_v_values1 = self.joint_action_value_functions[0](obs_idx.view(obs_idx.size(0), 1, -1).squeeze(1))
            sp_rl_rew1 = sp_rew_batch[:, idx, 0]
            sp_v_values2 = self.joint_action_value_functions[1](obs_idx.view(obs_idx.size(0), 1, -1).squeeze(1))
            sp_rl_rew2 = sp_rew_batch[:, idx, 1]
            sp_rl_done = sp_done_batch[:, idx]

            if idx == obs_length-1:
                target_value1 = self.target_joint_action_value_functions[0](n_obs_idx.view(n_obs_idx.size(0), 1, -1).squeeze(1))
                target_value2 = self.target_joint_action_value_functions[1](n_obs_idx.view(n_obs_idx.size(0), 1, -1).squeeze(1))

            target_value1 = (
                    sp_rl_rew1.view(-1, 1) + (self.config.train["gamma"] * (1 - sp_rl_done.view(-1, 1)) * target_value1)
            ).detach()

            target_value2 = (
                    sp_rl_rew2.view(-1, 1) + (self.config.train["gamma"] * (1 - sp_rl_done.view(-1, 1)) * target_value2)
            ).detach()

            predicted_values1.append(sp_v_values1)
            target_values1.append(target_value1)
            predicted_values2.append(sp_v_values2)
            target_values2.append(target_value2)

        predicted_values1 = torch.cat(predicted_values1, dim=0)
        all_target_values1 = torch.cat(target_values1, dim=0)
        predicted_values2 = torch.cat(predicted_values2, dim=0)
        all_target_values2 = torch.cat(target_values2, dim=0)

        sp_critic_loss1 = (0.5 * ((predicted_values1 - all_target_values1) ** 2)).mean()
        sp_critic_loss2 = (0.5 * ((predicted_values2 - all_target_values2) ** 2)).mean()
        return sp_critic_loss1 + sp_critic_loss2

    def compute_xp_advantages(self, xp_obs, xp_acts, xp_rews, xp_dones, xp_n_obses):
        """
            A function that computes the weighted advantage function as described in Expression 14 based on XP interaction
            :param obs: Agent observation
            :param n_obs: Agent next observation
            :param acts: Agent action
            :param dones: Agent done flag
            :param rews: Agent reward flag
            :return opt_diversity_values.detach() - baseline_diversity_values.detach(): weighted advantage function
            :return all_entropy_weight1.detach(): Variable entropy weights given \alpha^1
            :return all_entropy_weight1.detach(): Variable entropy weights given \alpha^2
        """
        batch_size = xp_obs.size()[0]
        obs_length = xp_obs.size()[1]
        xp_num_agents = xp_obs.size()[2]
        obs_only_length = self.effective_crit_obs_size - self.num_populations - self.num_agents

        opt_xp_matrics = []
        baseline_xp_matrics = []
        lagrangian_weights1 = []
        lagrangian_weights2 = []
        entropy_weight1 = []
        entropy_weight2 = []

        xp_targ_diversity_value = None
        # Compute added stuff related to index

        lagrange_matrix_mean_norm1 = self.compute_const_lagrange1().mean(dim=-1, keepdim=False)
        lagrange_matrix_mean_norm2 = self.compute_const_lagrange2().mean(dim=-1, keepdim=False)

        pos_entropy_weights1 = (lagrange_matrix_mean_norm1 / self.normalizer1) * self.config.loss_weights[
            "entropy_regularizer_loss"
        ]
        pos_entropy_weights2 = (lagrange_matrix_mean_norm2 / self.normalizer2) * self.config.loss_weights[
            "entropy_regularizer_loss"
        ]

        for idx in reversed(range(obs_length)):
            xp_obs_id = torch.cat([xp_obs[:, idx, :, :], torch.eye(self.num_agents).repeat(batch_size, 1, 1).to(self.device)], dim=-1)
            if idx == obs_length - 1:
                xp_n_obses_id = torch.cat([xp_n_obses[:, idx, :, :], torch.eye(self.num_agents).repeat(batch_size, 1, 1).to(self.device)], dim=-1)
                xp_targ_diversity_value = self.joint_action_value_functions[1](
                    xp_n_obses_id.view(xp_n_obses_id.size(0), 1, -1).squeeze(1)
                )

            xp_rl_rew = xp_rews[:, idx, 1]
            xp_rl_done = xp_dones[:, idx]

            xp_obs_only = xp_obs_id[:, :, :obs_only_length]
            offset = self.num_populations + self.num_agents
            if -offset + self.num_populations == 0:
                accessed_index = xp_obs_id[:, :, -offset:].argmax(dim=-1)
            else:
                accessed_index = xp_obs_id[:, :, -offset:-offset + self.num_populations].argmax(dim=-1)

            idx1, idx2 = accessed_index[:, 0], accessed_index[:, 1]
            baseline_matrix = self.joint_action_value_functions[1](
                xp_obs_id.view(xp_obs_id.size(0), 1, -1).squeeze(1)
            )

            lagrangian_matrix1 = self.compute_const_lagrange1().unsqueeze(0).repeat(batch_size, 1, 1)
            lagrangian_matrix2 = self.compute_const_lagrange2().unsqueeze(0).repeat(batch_size, 1, 1)

            xp_targ_diversity_value = (
                    xp_rl_rew.view(-1, 1) + (
                    self.config.train["gamma"] * (1 - xp_rl_done.view(-1, 1)) * xp_targ_diversity_value)
            ).detach()

            lagrangian_weights1.append(lagrangian_matrix1[torch.arange(batch_size), idx1, idx2].unsqueeze(-1))
            lagrangian_weights2.append(lagrangian_matrix2[torch.arange(batch_size), idx2, idx1].unsqueeze(-1))
            opt_xp_matrics.append(-xp_targ_diversity_value)
            baseline_xp_matrics.append(-baseline_matrix)
            entropy_weight1.append(pos_entropy_weights1[idx1])
            entropy_weight2.append(pos_entropy_weights2[idx2])

        all_baseline_matrices = torch.cat(baseline_xp_matrics, dim=0)
        all_opt_matrices = torch.cat(opt_xp_matrics, dim=0)
        all_lagrangian_matrices1 = torch.cat(lagrangian_weights1, dim=0)
        all_lagrangian_matrices2 = torch.cat(lagrangian_weights2, dim=0)
        all_entropy_weights1 = torch.cat(entropy_weight1, dim=0)
        all_entropy_weights2 = torch.cat(entropy_weight2, dim=0)

        if not self.config.train["with_lagrange"]:
            all_lagrangian_matrices1 = self.config.loss_weights["xp_loss_weights"] * torch.ones_like(
                all_lagrangian_matrices1)
            all_lagrangian_matrices2 = self.config.loss_weights["xp_loss_weights"] * torch.ones_like(
                all_lagrangian_matrices2)

        xp_lagrange_advantages = ((all_opt_matrices-all_baseline_matrices) * (
                all_lagrangian_matrices1+all_lagrangian_matrices2
            )
        ).squeeze(1)
        return xp_lagrange_advantages.detach(), all_entropy_weights1.detach(), all_entropy_weights2.detach()

    def compute_xp_old_probs(self, xp_obs, xp_acts, xp_rews, xp_dones, xp_n_obses):
        """
            A function that computes the previous policy's (before update) probability of selecting an action based on cross-play interaction data.
            Required for MAPPO update.
            :param obs: Agent observation
            :param n_obs: Agent next observation
            :param acts: Agent action
            :param dones: Agent done flag
            :param rews: Agent reward flag
            :return old_log_likelihoods: Previous policies' action log probability
        """
        obs_length = xp_obs.shape[1]
        xp_num_agents = xp_obs.shape[2]

        action_likelihood = []

        for idx in reversed(range(obs_length)):
            xp_obs_id = xp_obs[:, idx, :, :]
            xp_acts_idx = xp_acts[:, idx, :, :]
            xp_act_log_input = torch.cat(
                [xp_obs_id, torch.eye(self.num_agents).repeat(xp_obs_id.size()[0], 1, 1).to(self.device)],
                dim=-1
            )

            action_logits = self.separate_act_select_old(xp_act_log_input)
            final_selected_logits = action_logits
            action_distribution = dist.OneHotCategorical(logits=final_selected_logits)
            action_likelihood.append(action_distribution.log_prob(xp_acts_idx))

        action_log_likelihoods = torch.cat(action_likelihood, dim=0).sum(dim=-1)
        return action_log_likelihoods

    def compute_xp_actor_loss(self, xp_obs, xp_acts, xp_rews, xp_dones, xp_n_obses, advantages, old_log_likelihoods, entropy_weight1, entropy_weight2):
        """
            A function that computes the policy's loss function based on cross-play interaction
            :param obs: Agent observation
            :param n_obs: Agent next observation
            :param acts: Agent action
            :param dones: Agent done flag
            :param rews: Agent reward flag
            :param advantages: Weighted advantage value
            :param old_log_likelihoods: Previous policies log likelihood
            :param entropy_weight1: Variable entropy weights based on \alpha^1
            :param entropy_weight2: Variable entropy weights based on \alpha^2
            :return pol_loss: Policy loss
        """
        obs_length = xp_obs.shape[1]
        xp_num_agents = xp_obs.shape[2]

        action_likelihood = []
        action_entropies = []

        anneal_end = self.config.train["anneal_end"] * self.projected_total_updates
        for idx in reversed(range(obs_length)):
            xp_obs_id = xp_obs[:, idx, :, :]
            xp_acts_idx = xp_acts[:, idx, :, :]
            xp_act_log_input = torch.cat(
                [xp_obs_id, torch.eye(self.num_agents).repeat(xp_obs_id.size()[0], 1, 1).to(self.device)],
                dim=-1
            )

            action_logits = self.separate_act_select(xp_act_log_input)
            action_distribution = dist.OneHotCategorical(logits=action_logits)
            action_likelihood.append(action_distribution.log_prob(xp_acts_idx))
            action_entropies.append(action_distribution.entropy())

        action_log_likelihoods = torch.cat(action_likelihood, dim=0).sum(dim=-1)
        action_entropies = torch.cat(action_entropies, dim=0)

        entropy_weights = torch.cat([entropy_weight1.unsqueeze(dim=-1), entropy_weight2.unsqueeze(dim=-1)], dim=-1)
        entropy_loss = (entropy_weights * -action_entropies).sum(dim=-1).mean()
        xp_ratio = torch.exp(action_log_likelihoods - old_log_likelihoods.detach())
        surr1 = xp_ratio * advantages
        surr2 = torch.clamp(
            xp_ratio,
            1 - self.config.train["eps_clip"],
            1 + self.config.train["eps_clip"]
        ) * advantages

        xp_pol_list = torch.min(surr1, surr2)
        if self.config.train["with_dual_clip"]:
            xp_pol_list[advantages < 0] = torch.max(xp_pol_list[advantages < 0], self.config.train["dual_clip"]*advantages[advantages < 0])
        xp_pol_loss = -xp_pol_list.mean()

        return xp_pol_loss, entropy_loss

    def compute_xp_critic_loss(self, xp_obs, xp_acts, xp_rew, xp_dones, xp_n_obses):
        """
            A function that computes critic loss based on cross-play interaction
            :param obs: Agent observation
            :param n_obs: Agent next observation
            :param acts: Agent action
            :param dones: Agent done flag
            :param rews: Agent reward flag
            :return Lagrange multiplier loss
        """
        batch_size = xp_obs.shape[0]
        obs_length = xp_obs.shape[1]
        xp_num_agents = xp_obs.shape[2]

        predicted_values1 = []
        target_values1 = []
        predicted_values2 = []
        target_values2 = []

        xp_targ_values1 = None
        xp_targ_values2 = None

        for idx in reversed(range(obs_length)):
            xp_obs_id = xp_obs[:, idx, :, :]
            xp_critic_state_input = torch.cat(
                [xp_obs_id, torch.eye(self.num_agents).repeat([batch_size, 1, 1]).to(self.device)],dim=-1
            )

            xp_v_values1 = self.joint_action_value_functions[0](
                xp_critic_state_input.view(xp_critic_state_input.size(0), 1, -1).squeeze(1)
            )
            xp_v_values2 = self.joint_action_value_functions[1](
                xp_critic_state_input.view(xp_critic_state_input.size(0), 1, -1).squeeze(1)
            )

            if idx == obs_length - 1:
                xp_n_obses_id = xp_n_obses[:, idx, :, :]
                xp_critic_n_state_input = torch.cat(
                    [xp_n_obses_id, torch.eye(self.num_agents).repeat([batch_size, 1, 1]).to(self.device)],
                    dim=-1
                )

                xp_targ_values1 = self.target_joint_action_value_functions[0](
                    xp_critic_n_state_input.view(xp_critic_n_state_input.size(0), 1, -1).squeeze(1)
                )
                xp_targ_values2 = self.target_joint_action_value_functions[1](
                    xp_critic_n_state_input.view(xp_critic_n_state_input.size(0), 1, -1).squeeze(1)
                )

            xp_rl_rew1 = xp_rew[:, idx, 0]
            xp_rl_rew2 = xp_rew[:, idx, 1]
            xp_rl_done = xp_dones[:, idx]

            xp_targ_values1 = (
                xp_rl_rew1.view(-1, 1) + (self.config.train["gamma"] * (1 - xp_rl_done.view(-1, 1)) * xp_targ_values1)
            ).detach()

            xp_targ_values2 = (
                xp_rl_rew2.view(-1, 1) + (self.config.train["gamma"] * (1 - xp_rl_done.view(-1, 1)) * xp_targ_values2)
            ).detach()

            predicted_values1.append(xp_v_values1)
            target_values1.append(xp_targ_values1)
            predicted_values2.append(xp_v_values2)
            target_values2.append(xp_targ_values2)

        predicted_values1 = torch.cat(predicted_values1, dim=0)
        predicted_values2 = torch.cat(predicted_values2, dim=0)
        all_target_values1 = torch.cat(target_values1, dim=0)
        all_target_values2 = torch.cat(target_values2, dim=0)
        xp_critic_loss1 = (0.5 * ((predicted_values1 - all_target_values1) ** 2)).mean()
        xp_critic_loss2 = (0.5 * ((predicted_values2 - all_target_values2) ** 2)).mean()
        return xp_critic_loss1 + xp_critic_loss2


    def update_policy_and_critic_and_lagrange(self, batches, xp_batches):
        """
            A method that updates the joint policy model following sampled self-play and cross-play experiences.
            :param batches: A batch of obses and acts sampled from self-play experience replay.
            :param xp_batches: A batch of experience from cross-play experience replay.
        """

        self.total_updates += 1

        # Get obs and acts batch and prepare inputs to model.
        obs_batch, acts_batch = torch.tensor(batches[0]).to(self.device), torch.tensor(batches[1]).to(self.device)
        sp_n_obs_batch = torch.tensor(batches[2]).to(self.device)
        sp_done_batch = torch.tensor(batches[3]).double().to(self.device)
        rewards_batch = torch.tensor(batches[4]).double().to(self.device)

        if self.config.loss_weights["xp_loss_weights"] != 0:
            xp_obs, xp_acts = torch.tensor(xp_batches[0]).to(self.device), torch.tensor(xp_batches[1]).to(self.device)
            xp_n_obses = torch.tensor(xp_batches[2]).to(self.device)
            xp_dones = torch.tensor(xp_batches[3]).double().to(self.device)
            xp_rews = torch.tensor(xp_batches[4]).double().to(self.device)

        sp_advantages1, sp_advantages2, entropy_weight1, entropy_weight2 = self.compute_sp_advantages(
            obs_batch, sp_n_obs_batch, acts_batch, sp_done_batch, rewards_batch
        )

        sp_old_log_probs = self.compute_sp_old_probs(
            obs_batch, sp_n_obs_batch, acts_batch, sp_done_batch, rewards_batch
        )

        if self.config.loss_weights["xp_loss_weights"] != 0:
            xp_advantages, xp_ent_weight1,  xp_ent_weight2 = self.compute_xp_advantages(
                xp_obs, xp_acts, xp_rews, xp_dones, xp_n_obses
            )

            xp_old_log_probs = self.compute_xp_old_probs(
                xp_obs, xp_acts, xp_rews, xp_dones, xp_n_obses
            )

        for pol, old_pol in zip(self.joint_policy, self.old_joint_policy):
            for target_param, param in zip(old_pol.parameters(), pol.parameters()):
                target_param.data.copy_(param.data)

        for _ in range(self.config.train["epochs_per_update"]):
            self.actor_optimizer.zero_grad()

            # Compute SP Actor Loss
            sp_pol_loss, sp_action_entropies = self.compute_sp_actor_loss(
                obs_batch, sp_n_obs_batch, acts_batch, sp_done_batch, rewards_batch, sp_advantages1, sp_advantages2, sp_old_log_probs, entropy_weight1, entropy_weight2
            )
            if self.config.loss_weights["jsd_weight"] == 0:
                act_diff_log_mean = 0
            else:
                act_diff_log_mean = self.compute_jsd_loss(obs_batch, acts_batch)

            if self.config.loss_weights["xp_loss_weights"] != 0:
                # Compute XP Actor Loss
                total_xp_actor_loss, total_xp_entropy_loss = self.compute_xp_actor_loss(
                    xp_obs, xp_acts, xp_rews, xp_dones, xp_n_obses, xp_advantages, xp_old_log_probs, xp_ent_weight1,  xp_ent_weight2
                )

            xp_multiplier = (self.config.env.parallel["xp_collection"]+0.0)/self.config.env.parallel["sp_collection"]
            total_actor_loss = sp_pol_loss + xp_multiplier*total_xp_actor_loss + sp_action_entropies + xp_multiplier*total_xp_entropy_loss + (
                        act_diff_log_mean * self.config.loss_weights["jsd_weight"])

            total_actor_loss.backward()

            if self.config.train['max_grad_norm'] > 0:
                for model in self.joint_policy:
                    nn.utils.clip_grad_norm_(model.parameters(), self.config.train['max_grad_norm'])

            self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        # Compute SP Critic Loss
        sp_critic_loss = self.compute_sp_critic_loss(
            obs_batch, sp_n_obs_batch, acts_batch, rewards_batch, sp_done_batch
        )

        # Compute SP Lagrange Loss
        if self.total_updates % self.config.train["lagrange_update_period"] == 0:
            if self.config.train["with_lagrange"]:
                sp_lagrange_data, lagrange_mult_norm_sp1, lagrange_mult_norm_sp2 = self.compute_sp_lagrange_loss(
                    obs_batch, sp_n_obs_batch, acts_batch, sp_done_batch, rewards_batch
                )

        # Get XP data and preprocess it for matrix computation
        # total_xp_critic_loss, total_xp_actor_loss, total_xp_entropy_loss, xp_pred_div, xp_lagrange_loss = 0, 0, 0, 0, 0
        if self.config.loss_weights["xp_loss_weights"] != 0:
            # Compute XP Critic Loss
            total_xp_critic_loss = self.compute_xp_critic_loss(
                xp_obs, xp_acts, xp_rews, xp_dones, xp_n_obses
            )

            # Compute XP Lagrange Loss
            if self.config.train["with_lagrange"]:
                if self.total_updates % self.config.train["lagrange_update_period"] == 0:
                    xp_lagrange_data, lagrange_mult_norm_xp1, lagrange_mult_norm_xp2 = self.compute_xp_lagrange_loss(
                        xp_obs, xp_acts, xp_rews, xp_dones, xp_n_obses
                    )

        if self.with_any_play:
            total_classifier_loss = self.config.any_play["any_play_classifier_loss_weight"] * self.compute_pop_class_loss(obs_batch)

        total_critic_loss = sp_critic_loss * self.config.loss_weights["sp_val_loss_weight"] + total_xp_critic_loss * self.config.loss_weights["xp_val_loss_weight"]
        if self.config.train["with_lagrange"]:
            if self.total_updates % self.config.train["lagrange_update_period"] == 0:
                #total_lagrange_loss = self.config.loss_weights["lagrange_weights"] * (sp_lagrange_loss + xp_lagrange_loss)
                sp_all_diff1, sp_all_diff2, sp_all_id11, sp_all_id21, sp_all_id12, sp_all_id22 = sp_lagrange_data
                xp_all_diff1, xp_all_diff2, xp_all_id11, xp_all_id21, xp_all_id12, xp_all_id22 = xp_lagrange_data

                all_diff1 = torch.cat([sp_all_diff1.view(-1), xp_all_diff1.view(-1)], dim=0)
                all_id11 = torch.cat([sp_all_id11.view(-1), xp_all_id11.view(-1)], dim=0)
                all_id21 = torch.cat([sp_all_id21.view(-1), xp_all_id21.view(-1)], dim=0)
                all_id12 = torch.cat([sp_all_id12.view(-1), xp_all_id12.view(-1)], dim=0)
                all_id22 = torch.cat([sp_all_id22.view(-1), xp_all_id22.view(-1)], dim=0)
                all_diff2 = torch.cat([sp_all_diff2.view(-1), xp_all_diff2.view(-1)], dim=0)
                for ii in range(self.num_populations):
                    for jj in range(self.num_populations):
                        if ii != jj:
                            eligible_diffs = all_diff1[torch.logical_and(all_id11 == ii, all_id21 == jj)]
                            if eligible_diffs.size()[0] != 0:
                                self.lagrange_multiplier_matrix1[ii][jj] = F.relu(F.relu(self.lagrange_multiplier_matrix1[ii][jj]) - (self.config.train["lagrange_lr"] * self.config.loss_weights["lagrange_weights"] * eligible_diffs.mean()))
                            eligible_diffs2 = all_diff2[torch.logical_and(all_id12 == ii, all_id22 == jj)]
                            if eligible_diffs2.size()[0] != 0:
                                self.lagrange_multiplier_matrix2[ii][jj] = F.relu(F.relu(self.lagrange_multiplier_matrix2[ii][jj]) - (self.config.train["lagrange_lr"] * self.config.loss_weights["lagrange_weights"] * eligible_diffs2.mean()))

        # Write losses to logs
        self.next_log_update += self.logger.train_log_period
        train_step = (self.total_updates-1) * self.logger.steps_per_update
        self.logger.log_item("Train/sp/actor_loss", sp_pol_loss,
                                 train_step=train_step, updates=self.total_updates-1)
        self.logger.log_item("Train/sp/critic_loss", sp_critic_loss,
                                 train_step=train_step, updates=self.total_updates-1)
        self.logger.log_item("Train/sp/entropy", sp_action_entropies,
                                 train_step=train_step, updates=self.total_updates-1)
        self.logger.log_item("Train/jsd_loss", act_diff_log_mean*self.config.loss_weights["jsd_weight"],
                                 train_step=train_step, updates=self.total_updates-1)
        self.logger.log_item("Train/xp/actor_loss", total_xp_actor_loss,
                                 train_step=train_step, updates=self.total_updates-1)
        self.logger.log_item("Train/xp/critic_loss", total_xp_critic_loss,
                                 train_step=train_step, updates=self.total_updates-1)
        self.logger.log_item("Train/xp/entropy", total_xp_entropy_loss,
                                 train_step=train_step, updates=self.total_updates-1)
        if self.total_updates % self.config.train["lagrange_update_period"] == 0:
            if self.config.train["with_lagrange"]:
                self.logger.log_item("Train/sp/lagrange_mult_norm", (lagrange_mult_norm_sp1 + lagrange_mult_norm_sp2)/2,
                                         train_step=train_step, updates=self.total_updates-1)
                self.logger.log_item("Train/xp/lagrange_mult_norm", (lagrange_mult_norm_xp1 + lagrange_mult_norm_xp2)/2,
                                         train_step=train_step, updates=self.total_updates-1)
        
        self.logger.commit()

        # Backpropagate critic loss
        if not self.with_any_play:
            total_critic_loss.backward()
        else:
            (total_critic_loss+total_classifier_loss).backward()

        # Clip grads if necessary
        if self.config.train['max_grad_norm'] > 0:
            for idx, model in enumerate(self.joint_policy):
                nn.utils.clip_grad_norm_(model.parameters(), self.config.train['max_grad_norm'])

        # Log grad magnitudes if specified by config.
        if self.config.logger["log_grad"]:

            for idx, model in enumerate(self.joint_policy):
                for name, param in model.named_parameters():
                    if not param.grad is None:
                        self.logger.log_item(
                            f"Train/grad/actor_{idx}_{name}",
                            torch.abs(param.grad).mean(),
                            train_step=self.total_updates-1
                        )

            for idx, model in enumerate(self.joint_policy):
                for name, param in model.named_parameters():
                    if not param.grad is None:
                        self.logger.log_item(
                            f"Train/grad/critic_{name}",
                            torch.abs(param.grad).mean(),
                            train_step=self.total_updates-1
                            )

        self.critic_optimizer.step()
        self.soft_copy(self.config.train["target_update_rate"])

    def _rank3_trace(self, x):
        return torch.einsum('ijj->i', x)

    def _rank3_diag(self, x):
        eye = torch.eye(x.size(1)).to(self.device).type_as(x)
        out = eye * x.unsqueeze(2).expand(*x.size(), x.size(1))
        return out

    def diversity_loss_computation(self, lagrangian_matrix1, lagrangian_matrix2, xp_matrix):
        num_populations = xp_matrix.size()[-1]
        cross_play_matrix_diagonal = torch.diagonal(xp_matrix, dim1=-2, dim2=-1)
        lagrange_matrix_diagonal = torch.diagonal(lagrangian_matrix1, dim1=-2, dim2=-1)

        div_loss = None
        diag_cross_play = torch.diag_embed(cross_play_matrix_diagonal)
        diag_lagrange = torch.diag_embed(lagrange_matrix_diagonal)

        ones_like_diag = torch.ones_like(diag_cross_play).to(self.device)
        diag_ones_bmm = torch.bmm(diag_cross_play, ones_like_diag)
        final_lagrange_matrix = lagrangian_matrix1 - diag_lagrange

        tolerance_matrix = self.config.train["tolerance_factor"] * torch.ones_like(diag_cross_play).to(self.device)
        tolerance_matrix_diagonal = torch.diagonal(tolerance_matrix, dim1=-2, dim2=-1)
        final_tolerance_matrix = tolerance_matrix - torch.diag_embed(tolerance_matrix_diagonal)

        maximised_matrix = (final_lagrange_matrix * (diag_ones_bmm - xp_matrix - final_tolerance_matrix)).sum(dim=-1).sum(dim=-1)
        div_loss1 = -(maximised_matrix/(
            self.num_populations * (self.num_populations-1)
        ))

        lagrange_matrix_diagonal2 = torch.diagonal(lagrangian_matrix2, dim1=-2, dim2=-1)
        diag_lagrange2 = torch.diag_embed(lagrange_matrix_diagonal2)
        final_lagrange_matrix2 = lagrangian_matrix2 - diag_lagrange2

        maximised_matrix2 = (
            final_lagrange_matrix2 * (diag_ones_bmm - torch.permute(xp_matrix + final_tolerance_matrix, (0,2, 1)))
        ).sum(dim=-1).sum(dim=-1)
        div_loss2 = -(maximised_matrix2/(
            self.num_populations * (self.num_populations-1)
        ))

        div_loss = 0.5*div_loss1 + 0.5*div_loss2
        diagonal_udrl_loss = -self._rank3_trace(xp_matrix)/num_populations
        final_loss = div_loss + diagonal_udrl_loss
        return final_loss, diagonal_udrl_loss, div_loss

    def hard_copy(self):
        for idx in range(len(self.joint_action_value_functions)):
            for target_param, param in zip(self.target_joint_action_value_functions[idx].parameters(), self.joint_action_value_functions[idx].parameters()):
                target_param.data.copy_(param.data)

    def soft_copy(self, tau=0.001):
        for idx in range(len(self.joint_action_value_functions)):
            for target_param, param in zip(self.target_joint_action_value_functions[idx].parameters(), self.joint_action_value_functions[idx].parameters()):
                target_param.data.copy_(tau * param + (1 - tau) * target_param)

    def save_model(self, int_id, save_model=False):
        """
            A method to save model parameters.
            :param int_id: Integer indicating ID of the checkpoint.
        """
        if not save_model:
            return

        for id, model in enumerate(self.joint_policy):
            torch.save(model.state_dict(), f"models/model_{id}_{int_id}.pt")

        for idx in range(len(self.joint_action_value_functions)):
            torch.save(self.joint_action_value_functions[idx].state_dict(),
                       f"models/model_{int_id}-action-value-{idx}.pt")

        for idx in range(len(self.target_joint_action_value_functions)):
            torch.save(self.target_joint_action_value_functions[idx].state_dict(),
                       f"models/model_{int_id}-target-action-value-{idx}.pt")

        torch.save(self.actor_optimizer.state_dict(),
                   f"models/model_{int_id}-act-optim.pt")

        torch.save(self.critic_optimizer.state_dict(),
                   f"models/model_{int_id}-crit-optim.pt")

        torch.save(self.lagrange_multiplier_matrix1,
                   f"models/model_{int_id}-lagrange1.pt")

        torch.save(self.lagrange_multiplier_matrix2,
                   f"models/model_{int_id}-lagrange2.pt")

    def load_model(self, int_id, overridden_model_dir=None):
        """
        A method to load stored models to be used by agents
        """

        if self.mode == "train":
            model_dir = self.config['load_dir']

            for id, model in enumerate(self.joint_policy):
                model.load_state_dict(
                    torch.load(f"{model_dir}/models/model_{id}_{int_id}.pt")
                )

            for id, model in enumerate(self.joint_action_value_functions):
                self.joint_action_value_functions[id].load_state_dict(
                    torch.load(f"{model_dir}/models/model_{int_id}-action-value-{id}.pt")
                )

            for id, model in enumerate(self.joint_action_value_functions):
                self.target_joint_action_value_functions.load_state_dict(
                    torch.load(f"{model_dir}/models/model_{int_id}-target-action-value-{id}.pt")
                )

            self.actor_optimizer.load_state_dict(
                torch.load(f"{model_dir}/models/model_{int_id}-act-optim.pt")
            )

            self.critic_optimizer.load_state_dict(
                torch.load(f"{model_dir}/models/model_{int_id}-crit-optim.pt")
            )

            self.lagrange_multiplier_matrix1.load_state_dict(
                torch.load(f"{model_dir}/models/model_{int_id}-lagrange1.pt")
            )

            self.lagrange_multiplier_matrix2.load_state_dict(
                torch.load(f"{model_dir}/models/model_{int_id}-lagrange2.pt")
            )

        else:
            model_dir = self.config.env['model_load_dir']
            if not overridden_model_dir is None:
                model_dir = overridden_model_dir

            for id, model in enumerate(self.joint_policy):
                model.load_state_dict(
                    torch.load(f"{model_dir}/model_{id}_{int_id}.pt", map_location=self.device)
                )
