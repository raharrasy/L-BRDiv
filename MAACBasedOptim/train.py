import gym
import random
import lbforaging
import coopreaching
import matrixgames
import torch
import string
import json
import numpy as np
from ExpReplay import EpisodicExperienceReplay
from Agents import Agents
import os
import wandb
from omegaconf import OmegaConf

class DiversityTraining(object):
    """
        A class that runs an experiment on learning with Upside Down Reinforcement Learning (UDRL).
    """
    def __init__(self, config):
        """
            Constructor for UDRLTraining class
                Args:
                    config : A dictionary containing required hyperparameters for UDRL training
        """
        self.config = config
        cuda_device = "cuda"
        if config.run['device_id'] != -1:
            cuda_device = cuda_device + ":" + str(config.run['device_id'])
        self.device = torch.device(cuda_device if config.run['use_cuda'] and torch.cuda.is_available() else "cpu")
        self.env_name = config.env["name"]

        self.logger = Logger(config)

        # Other experiment related variables
        self.exp_replay = None
        self.cross_play_exp_replay = None

        self.sp_selected_agent_idx = None
        self.xp_selected_agent_idx_p1 = None
        self.xp_selected_agent_idx_p2 = None

        self.stored_obs = None
        self.stored_nobs = None

        self.stored_obs_xp = None
        self.stored_nobs_xp = None

    def get_obs_sizes(self, obs_space):
        """
            Method to get the size of the envs' obs space and length of obs features. Must be defined for every envs.
        """
        out_shape = list(obs_space.shape)
        out_shape[-1] += self.config.populations["num_populations"]
        if "Foraging" in self.env_name:
            return tuple(out_shape), obs_space.shape[-1] + self.config.populations["num_populations"]
        if "Reaching" in self.env_name:
            return tuple(out_shape), obs_space.shape[-1] + self.config.populations["num_populations"]
        if "matrix" in self.env_name:
            return tuple(out_shape), obs_space.shape[-1] + self.config.populations["num_populations"]
        return None

    def create_directories(self):
        """
            A method that creates the necessary directories for storing resulting logs & parameters.
        """
        if not os.path.exists("models"):
            os.makedirs("models")

    def to_one_hot_population_id(self, indices):
        act_indices = np.asarray(indices).astype(int)
        one_hot_ids = np.eye(self.config.populations["num_populations"])[act_indices]

        return one_hot_ids

    def eval_sp_policy_performance(self, agent_population, logger, logging_id, eval=False):
        """
            A method to evaluate the resulting performance of a trained agent population model
            :param agent_population: An agent population model in form of a GNN-based joint policy.
            :param logger: A tensorboard writer used for writing results.
            :param logging_id: Checkpoint ID for logging.
        """

        # Create env for policy eval
        def make_env(env_name):
            def _make():
                env = gym.make(
                    env_name
                )
                return env

            return _make

        num_pops = self.config.populations["num_populations"]
        returns = np.zeros((num_pops, num_pops, 2))

        for pop_id in range(num_pops):
            env_train = gym.vector.AsyncVectorEnv([
                make_env(
                    self.config.env["name"]
                ) for idx in range(self.config.env.parallel["eval"])
            ])

            # Initialize objects to track returns.
            device = torch.device("cuda" if self.config.run['use_cuda'] and torch.cuda.is_available() else "cpu")
            num_dones = [0] * self.config.env.parallel["eval"]
            per_worker_rew1 = [0.0] * self.config.env.parallel["eval"]
            per_worker_rew2 = [0.0] * self.config.env.parallel["eval"]
            per_worker_non_disc_rew1 = [0.0] * self.config.env.parallel["eval"]
            per_worker_non_disc_rew2 = [0.0] * self.config.env.parallel["eval"]

            # Initialize initial obs and states for model
            obs, _ = env_train.reset(seed=[self.config.run["eval_seed"] + idx for idx in range(self.config.env.parallel["eval"])])
            time_elapsed = np.zeros([obs.shape[0], obs.shape[1], 1])
            avgs1 = []
            avgs2 = []
            avgs_non_disc1 = []
            avgs_non_disc2 = []

            while (any([k < self.config.run["num_eval_episodes"] for k in num_dones])):
                #acts = agent_population.decide_acts(np.concatenate([obs, remaining_target, time_elapsed], axis=-1))
                one_hot_id_shape = list(obs.shape)[:-1]
                one_hot_ids = self.to_one_hot_population_id(pop_id*np.ones(one_hot_id_shape))

                # Decide agent's action based on model & target returns.
                acts = agent_population.decide_acts(np.concatenate([obs, one_hot_ids], axis=-1), eval=eval)
                # Execute prescribed action
                n_obs, rews, dones, _, infos = env_train.step(acts)

                obs = n_obs
                time_elapsed = time_elapsed+1

                # Log per thread returns
                per_worker_rew1 = [k + (self.config.train["gamma"]**(te[0][0]-1))*l[0] for k, te, l in zip(per_worker_rew1, time_elapsed, rews)]
                per_worker_rew2 = [k + (self.config.train["gamma"]**(te[0][0]-1))*l[1] for k, te, l in zip(per_worker_rew2, time_elapsed, rews)]
                per_worker_non_disc_rew1 = [k + l[0] for k, l in zip(per_worker_non_disc_rew1, rews)]
                per_worker_non_disc_rew2 = [k + l[1] for k, l in zip(per_worker_non_disc_rew2, rews)]
                for idx, flag in enumerate(dones):
                    # If an episode in one of the threads ends...
                    if flag:
                        # Reset all relevant variables used in tracking and send logged returns in the finished thread to a storage
                        time_elapsed[idx] = np.zeros([obs.shape[1], 1])
                        if num_dones[idx] < self.config.run['num_eval_episodes']:
                            num_dones[idx] += 1
                            avgs1.append(per_worker_rew1[idx])
                            avgs2.append(per_worker_rew2[idx])
                            avgs_non_disc1.append(per_worker_non_disc_rew1[idx])
                            avgs_non_disc2.append(per_worker_non_disc_rew2[idx])
                        per_worker_rew1[idx] = 0
                        per_worker_rew2[idx] = 0
                        per_worker_non_disc_rew1[idx] = 0
                        per_worker_non_disc_rew2[idx] = 0

            # Log achieved returns.
            returns[pop_id, pop_id, 0] = np.mean(avgs1)
            returns[pop_id, pop_id, 1] = np.mean(avgs2)
            env_train.close()

            if not eval:
                logger.log_item(
                    f"Returns/sp/discounted_{pop_id}_1",
                    np.mean(avgs1),
                    checkpoint=logging_id)
                logger.log_item(
                    f"Returns/sp/discounted_{pop_id}_2",
                    np.mean(avgs2),
                    checkpoint=logging_id)
                logger.log_item(
                    f"Returns/sp/nondiscounted_{pop_id}_1",
                    np.mean(avgs_non_disc1),
                    checkpoint=logging_id)
                logger.log_item(
                    f"Returns/sp/nondiscounted_{pop_id}_2",
                    np.mean(avgs_non_disc2),
                    checkpoint=logging_id)
            else:
                logger.log_item(
                    f"Returns/sp/discounted_greedy_{pop_id}_1",
                    np.mean(avgs1),
                    checkpoint=logging_id)
                logger.log_item(
                    f"Returns/sp/discounted_greedy_{pop_id}_2",
                    np.mean(avgs2),
                    checkpoint=logging_id)
                logger.log_item(
                    f"Returns/sp/nondiscounted_greedy_{pop_id}_1",
                    np.mean(avgs_non_disc1),
                    checkpoint=logging_id)
                logger.log_item(
                    f"Returns/sp/nondiscounted_greedy_{pop_id}_2",
                    np.mean(avgs_non_disc2),
                    checkpoint=logging_id)
        return returns

    def eval_xp_policy_performance(self, agent_population, logger, logging_id, eval=False):
        """
            A method to evaluate the resulting performance of a trained agent population model
            :param agent_population: An agent population model in form of a GNN-based joint policy.
            :param logger: A tensorboard writer used for writing results.
            :param logging_id: Checkpoint ID for logging.
        """

        # Create env for policy eval
        def make_env(env_name):
            def _make():
                env = gym.make(
                    env_name
                )
                return env

            return _make

        num_pops = self.config.populations["num_populations"]
        returns = np.zeros((num_pops, num_pops,2))
        for pop_id in range(num_pops):
            for oppo_id in range(pop_id+1, num_pops):
                env_train = gym.vector.AsyncVectorEnv([
                    make_env(
                        self.config.env["name"]
                    ) for idx in range(self.config.env.parallel["eval"])
                ])

                # Initialize objects to track returns.
                device = torch.device("cuda" if self.config.run['use_cuda'] and torch.cuda.is_available() else "cpu")
                num_dones = [0] * self.config.env.parallel["eval"]
                per_worker_rew1 = [0.0] * self.config.env.parallel["eval"]
                per_worker_non_disc_rew1 = [0.0] * self.config.env.parallel["eval"]
                per_worker_rew2 = [0.0] * self.config.env.parallel["eval"]
                per_worker_non_disc_rew2 = [0.0] * self.config.env.parallel["eval"]


                # Initialize initial obs and states for model
                obs, _ = env_train.reset(seed=[self.config.run["eval_seed"] + idx for idx in range(self.config.env.parallel["eval"])])
                time_elapsed = np.zeros([obs.shape[0], obs.shape[1], 1])
                avgs1 = []
                avgs2 = []
                avgs_non_disc1 = []
                avgs_non_disc2 = []

                while (any([k < self.config.run["num_eval_episodes"] for k in num_dones])):

                    one_hot_id_shape_indiv = list(obs.shape)[:-2]
                    one_hot_id_shape_indiv.append(1)
                    one_hot_ids = self.to_one_hot_population_id(pop_id*np.ones(one_hot_id_shape_indiv))
                    one_hot_ids2 = self.to_one_hot_population_id(oppo_id*np.ones(one_hot_id_shape_indiv))
                    selected_acts, act_logits = agent_population.decide_acts(
                        np.concatenate([obs, np.concatenate([one_hot_ids, one_hot_ids2], axis=-2)], axis=-1), True, eval=eval
                    )
                    n_obs, rews, dones, _, infos = env_train.step(selected_acts)

                    obs = n_obs
                    time_elapsed = time_elapsed + 1
                    # Log per thread returns
                    per_worker_rew1 = [k + (self.config.train["gamma"] ** (te[0][0] - 1)) * l[0] for k, te, l in
                                      zip(per_worker_rew1, time_elapsed, rews)]
                    per_worker_rew2 = [k + (self.config.train["gamma"] ** (te[0][0] - 1)) * l[1] for k, te, l in
                                      zip(per_worker_rew2, time_elapsed, rews)]
                    per_worker_non_disc_rew1 = [k + l[0] for k, l in zip(per_worker_non_disc_rew1, rews)]
                    per_worker_non_disc_rew2 = [k + l[1] for k, l in zip(per_worker_non_disc_rew2, rews)]
                    for idx, flag in enumerate(dones):
                        # If an episode in one of the threads ends...
                        if flag:
                            time_elapsed[idx] = np.zeros([obs.shape[1], 1])
                            if num_dones[idx] < self.config.run['num_eval_episodes']:
                                num_dones[idx] += 1
                                avgs1.append(per_worker_rew1[idx])
                                avgs_non_disc1.append(per_worker_non_disc_rew1[idx])
                                avgs2.append(per_worker_rew2[idx])
                                avgs_non_disc2.append(per_worker_non_disc_rew2[idx])
                            per_worker_rew1[idx] = 0
                            per_worker_non_disc_rew1[idx] = 0
                            per_worker_rew2[idx] = 0
                            per_worker_non_disc_rew2[idx] = 0

                returns[pop_id, oppo_id,0] = np.mean(avgs1)
                returns[oppo_id, pop_id,0] = np.mean(avgs1)
                returns[pop_id, oppo_id,1] = np.mean(avgs2)
                returns[oppo_id, pop_id,1] = np.mean(avgs2)
                env_train.close()
                if not eval:
                    logger.log_item(
                        f"Returns/xp/discounted_{pop_id}_{oppo_id}_1",
                        np.mean(avgs1),
                        checkpoint=logging_id)
                    logger.log_item(
                        f"Returns/xp/nondiscounted_{pop_id}_{oppo_id}_1",
                        np.mean(avgs_non_disc1),
                        checkpoint=logging_id)
                    logger.log_item(
                        f"Returns/xp/discounted_{pop_id}_{oppo_id}_2",
                        np.mean(avgs2),
                        checkpoint=logging_id)
                    logger.log_item(
                        f"Returns/xp/nondiscounted_{pop_id}_{oppo_id}_2",
                        np.mean(avgs_non_disc2),
                        checkpoint=logging_id)
                else:
                    logger.log_item(
                        f"Returns/xp/discounted_greedy_{pop_id}_{oppo_id}_1",
                        np.mean(avgs1),
                        checkpoint=logging_id)
                    logger.log_item(
                        f"Returns/xp/discounted_greedy_{pop_id}_{oppo_id}_2",
                        np.mean(avgs2),
                        checkpoint=logging_id)
                    logger.log_item(
                        f"Returns/xp/nondiscounted_greedy_{pop_id}_{oppo_id}_1",
                        np.mean(avgs_non_disc1),
                        checkpoint=logging_id)
                    logger.log_item(
                        f"Returns/xp/nondiscounted_greedy_{pop_id}_{oppo_id}_2",
                        np.mean(avgs_non_disc2),
                        checkpoint=logging_id)
        return returns

    def self_play_data_gathering(self, env, agent_population, obs_sizes, real_obs_size, act_sizes_all):
        # Get required data from selected agents
        target_timesteps_elapsed = self.config.train["timesteps_per_update"]

        timesteps_elapsed = 0
        if not self.sp_selected_agent_idx:
            self.sp_selected_agent_idx = [np.random.choice(list(range(self.config.populations["num_populations"])), 1)[0] for _ in range(self.config.env.parallel["sp_collection"])]

        real_obs_header_size = [self.config.env.parallel["sp_collection"], target_timesteps_elapsed]
        act_header_size = [self.config.env.parallel["sp_collection"], target_timesteps_elapsed]

        real_obs_header_size.extend(list(real_obs_size))
        act_header_size.extend(list(act_sizes_all))

        stored_real_obs = np.zeros(real_obs_header_size)
        stored_next_real_obs = np.zeros(real_obs_header_size)
        stored_acts = np.zeros(act_header_size)
        stored_rewards = np.zeros([self.config.env.parallel["sp_collection"], target_timesteps_elapsed, 2])
        stored_dones = np.zeros([self.config.env.parallel["sp_collection"], target_timesteps_elapsed])

        while timesteps_elapsed < target_timesteps_elapsed:
            one_hot_id_shape = list(self.stored_obs.shape)[:-1]
            one_hot_ids = self.to_one_hot_population_id(np.expand_dims(np.asarray(self.sp_selected_agent_idx), axis=-1) * np.ones(one_hot_id_shape))
            
            # Decide agent's action based on model and execute.
            real_input = np.concatenate([self.stored_obs, one_hot_ids], axis=-1)
            acts, act_log_prob = agent_population.decide_acts(real_input, True)
            self.stored_nobs, rews, dones, _, infos = env.step(acts)
            real_n_input = np.concatenate([self.stored_nobs, one_hot_ids], axis=-1)

            # Store data from most recent timestep into tracking variables
            one_hot_acts = agent_population.to_one_hot(acts)

            stored_real_obs[:, timesteps_elapsed] = real_input
            stored_next_real_obs[:, timesteps_elapsed] = real_n_input
            stored_acts[:, timesteps_elapsed] = one_hot_acts
            stored_rewards[:, timesteps_elapsed, :] = rews
            stored_dones[:, timesteps_elapsed] = dones

            self.stored_obs = self.stored_nobs
            timesteps_elapsed += 1

            # TODO Change agent id in finished envs.
            for idx, flag in enumerate(dones):
                # If an episode collected by one of the threads ends...
                if flag:
                    self.sp_selected_agent_idx[idx] = np.random.choice(list(range(self.config.populations["num_populations"])), 1)[0]

        for r_obs, nr_obs, acts, rewards, dones in zip(stored_real_obs, stored_next_real_obs, stored_acts, stored_rewards, stored_dones):
            self.exp_replay.add_episode(r_obs, acts, rewards, dones, nr_obs)

    def cross_play_data_gathering(self, env, agent_population, obs_sizes, real_obs_size, act_sizes_all):
        # Get required data from selected agents
        target_timesteps_elapsed = self.config.train["timesteps_per_update"]
        if (not self.xp_selected_agent_idx_p1) or (not self.xp_selected_agent_idx_p2):
            self.xp_selected_agent_idx_p1 = [np.random.choice(list(range(self.config.populations["num_populations"])), 1)[0]
                                             for _ in range(self.config.env.parallel["xp_collection"])]

            self.xp_selected_agent_idx_p2 = []
            for idx in range(len(self.xp_selected_agent_idx_p1)):
                sampled_pair = np.random.choice(list(range(self.config.populations["num_populations"])), 1)[0]
                while sampled_pair == self.xp_selected_agent_idx_p1[idx]:
                    sampled_pair = np.random.choice(list(range(self.config.populations["num_populations"])), 1)[0]

                self.xp_selected_agent_idx_p2.append(sampled_pair)

        timesteps_elapsed = 0

        real_obs_header_size = [self.config.env.parallel["xp_collection"], target_timesteps_elapsed]
        act_header_size = [self.config.env.parallel["xp_collection"], target_timesteps_elapsed]

        real_obs_header_size.extend(real_obs_size)
        act_header_size.extend(list(act_sizes_all))

        stored_real_obs = np.zeros(real_obs_header_size)
        stored_next_real_obs = np.zeros(real_obs_header_size)
        stored_acts = np.zeros(act_header_size)
        stored_rewards = np.zeros([self.config.env.parallel["xp_collection"], target_timesteps_elapsed, 2])
        stored_dones = np.zeros([self.config.env.parallel["xp_collection"], target_timesteps_elapsed])

        while timesteps_elapsed < target_timesteps_elapsed:
            one_hot_id_shape_indiv = list(self.stored_obs_xp.shape)[:-2]
            one_hot_id_shape_indiv.append(1)
            one_hot_ids = self.to_one_hot_population_id(np.expand_dims(np.asarray(self.xp_selected_agent_idx_p1), axis=-1)*np.ones(one_hot_id_shape_indiv))
            one_hot_ids2 = self.to_one_hot_population_id(np.expand_dims(np.asarray(self.xp_selected_agent_idx_p2), axis=-1)*np.ones(one_hot_id_shape_indiv))

            # decide actions
            real_input = np.concatenate([self.stored_obs_xp, np.concatenate([one_hot_ids, one_hot_ids2], axis=-2)], axis=-1)
            acts, act_logits = agent_population.decide_acts(
                real_input, True
            )
            self.stored_nobs_xp, rews, dones, _, infos = env.step(acts)

            next_one_hot_id_shape_indiv = list(self.stored_nobs_xp.shape)[:-2]
            next_one_hot_id_shape_indiv.append(1)
            next_one_hot_ids = self.to_one_hot_population_id(np.expand_dims(np.asarray(self.xp_selected_agent_idx_p1), axis=-1)*np.ones(next_one_hot_id_shape_indiv))
            next_one_hot_ids2 = self.to_one_hot_population_id(np.expand_dims(np.asarray(self.xp_selected_agent_idx_p2), axis=-1)*np.ones(next_one_hot_id_shape_indiv))
            next_real_input = np.concatenate([self.stored_nobs_xp, np.concatenate([next_one_hot_ids, next_one_hot_ids2], axis=-2)], axis=-1)

            # Store data from most recent timestep into tracking variables
            one_hot_acts = agent_population.to_one_hot(acts)
            self.stored_obs_xp = self.stored_nobs_xp

            stored_real_obs[:, timesteps_elapsed] = real_input
            stored_next_real_obs[:, timesteps_elapsed] = next_real_input
            stored_acts[:, timesteps_elapsed] = one_hot_acts
            stored_rewards[:, timesteps_elapsed, :] = rews
            stored_dones[:, timesteps_elapsed] = dones

            timesteps_elapsed += 1

            for idx, flag in enumerate(dones):
                # If an episode collected by one of the threads ends...
                if flag:
                    self.xp_selected_agent_idx_p1[idx] = np.random.choice(list(range(self.config.populations["num_populations"])), 1)[0]
                    sampled_pairing = np.random.choice(list(range(self.config.populations["num_populations"])), 1)[0]
                    while sampled_pairing == self.xp_selected_agent_idx_p1[idx]:
                        sampled_pairing = np.random.choice(list(range(self.config.populations["num_populations"])), 1)[0]
                    self.xp_selected_agent_idx_p2[idx] = sampled_pairing

        for cur_obs, acts, rew, done, next_obs in zip(stored_real_obs, stored_acts, stored_rewards, stored_dones,
                                                      stored_next_real_obs):
            self.cross_play_exp_replay.add_episode(cur_obs, acts, rew, done, next_obs)

    def run(self):
        """
            A method that encompasses the main training loop for UDRL.
        """

        # Create logging directories & utilities
        def randomString(stringLength=10):
            letters = string.ascii_lowercase
            return ''.join(random.choice(letters) for i in range(stringLength))

        # Initialize environment, agent population model & experience replay based on obs vector sizes
        env1 = gym.make(
            self.config.env["name"]
        )

        obs_sizes, agent_o_size = self.get_obs_sizes(env1.observation_space)
        act_sizes = env1.action_space.nvec[0]
        act_sizes_all = (len(env1.action_space.nvec), act_sizes)

        obs_size_list = list(obs_sizes)
        real_obs_size = tuple(obs_size_list)

        def make_env(env_name):
            def _make():
                env = gym.make(
                    env_name
                )
                return env

            return _make

        env = gym.vector.AsyncVectorEnv([
            make_env(
                self.config.env["name"]
            ) for idx in range(self.config.env.parallel["sp_collection"])
        ])

        env_xp = gym.vector.AsyncVectorEnv([
            make_env(
                self.config.env["name"]
            ) for idx in range(self.config.env.parallel["xp_collection"])
        ])

        test_obses, _ = env.reset(seed=[self.config.run["seed"] + idx for idx in range(self.config.env.parallel["sp_collection"])])
        test_xp_obses, _ = env_xp.reset(seed=[self.config.run["seed"] + idx for idx in range(self.config.env.parallel["xp_collection"])])

        device = self.device
        obs_size = test_obses.shape[-1] + self.config.populations["num_populations"]

        self.create_directories()

        agent_population = Agents(obs_size, agent_o_size, obs_sizes[0], self.config.populations["num_populations"], self.config, act_sizes, device, self.logger)
        self.exp_replay = EpisodicExperienceReplay(
            real_obs_size, act_sizes_all, max_episodes=self.config.env.parallel["sp_collection"], max_eps_length=self.config.train["timesteps_per_update"]
        )

        self.cross_play_exp_replay = EpisodicExperienceReplay(
            real_obs_size, act_sizes_all, max_episodes=self.config.env.parallel["xp_collection"], max_eps_length=self.config.train["timesteps_per_update"]
        )

        # Save randomly initialized NN or load from pre-existing parameters if specified in argparse.
        if self.config.run["load_from_checkpoint"] == -1:
            agent_population.save_model(0, save_model=self.logger.save_model)
            sp_mat = self.eval_sp_policy_performance(agent_population, self.logger, 0)
            xp_mat = self.eval_xp_policy_performance(agent_population, self.logger, 0)
            self.logger.log_xp_matrix("Returns/xp_matrix", sp_mat + xp_mat, checkpoint=0)
        else:
            agent_population.load_model(self.config.run["load_from_checkpoint"])

        # Compute number of episodes required for training in each checkpoint.
        checkpoints_elapsed = self.config.run["load_from_checkpoint"] if self.config.run["load_from_checkpoint"] != -1 else 0
        total_checkpoints = self.config.run["total_checkpoints"]
        timesteps_per_checkpoint = self.config.run["num_timesteps"]//(total_checkpoints*(self.config.env.parallel["sp_collection"]+self.config.env.parallel["xp_collection"]))

        actual_div_loss=0
        for ckpt_id in range(checkpoints_elapsed, total_checkpoints):
            # Record number of episodes that has elapsed in a checkpoint

            self.stored_obs, _= env.reset()
            self.stored_obs_xp, _ = env_xp.reset()

            timesteps_elapsed = 0
            cross_play_timesteps_elapsed = 0
            lagrange_timesteps_elapsed = 0
            lagrange_cross_play_timesteps_elapsed = 0

            while (
                    timesteps_elapsed < timesteps_per_checkpoint or
                    lagrange_timesteps_elapsed < timesteps_per_checkpoint
            ) or ( self.config.loss_weights["xp_loss_weights"] != 0 and (
                cross_play_timesteps_elapsed < timesteps_per_checkpoint or
                lagrange_cross_play_timesteps_elapsed < timesteps_per_checkpoint)
            ):
                # Do Policy update
                self.self_play_data_gathering(
                    env, agent_population, obs_sizes, real_obs_size, act_sizes_all
                )

                if self.config.loss_weights["xp_loss_weights"] != 0:
                    self.cross_play_data_gathering(
                        env_xp, agent_population, obs_sizes, real_obs_size, act_sizes_all
                    )

                timesteps_elapsed += self.config.train["timesteps_per_update"]
                cross_play_timesteps_elapsed += self.config.train["timesteps_per_update"]
                lagrange_timesteps_elapsed += self.config.train["timesteps_per_update"]
                lagrange_cross_play_timesteps_elapsed += self.config.train["timesteps_per_update"]

                batches = self.exp_replay.sample_all()
                if self.config.loss_weights["xp_loss_weights"] != 0:
                    batches_xp = self.cross_play_exp_replay.sample_all()
                else:
                    batches_xp = None

                agent_population.update_policy_and_critic_and_lagrange(batches, batches_xp)
                self.exp_replay = EpisodicExperienceReplay(
                    real_obs_size, act_sizes_all, max_episodes=self.config.env.parallel["sp_collection"], max_eps_length=self.config.train["timesteps_per_update"]
                )

                if self.config.loss_weights["xp_loss_weights"] != 0:
                    self.cross_play_exp_replay = EpisodicExperienceReplay(
                        real_obs_size, act_sizes_all, max_episodes=self.config.env.parallel["xp_collection"], max_eps_length=self.config.train["timesteps_per_update"]
                    )

                lagrange_timesteps_elapsed += self.config.train["timesteps_per_update"]
                lagrange_cross_play_timesteps_elapsed += self.config.train["timesteps_per_update"]

            # Eval policy after sufficient number of episodes were collected.
            agent_population.save_model(ckpt_id+1, save_model=(self.logger.save_model
                                                               and ((ckpt_id+1) % self.logger.save_model_period == 0))
                                       )
            sp_mat = self.eval_sp_policy_performance(agent_population, self.logger, ckpt_id+1)
            xp_mat = self.eval_xp_policy_performance(agent_population, self.logger, ckpt_id+1)
            self.logger.log_xp_matrix("Returns/xp_matrix", sp_mat + xp_mat, checkpoint=ckpt_id+1)
            self.logger.commit()
        return actual_div_loss

class Logger:
    def __init__(self, config):
        
        logger_period = config.logger.logger_period 
        self.save_model = config.logger.get("save_model", False)
        self.save_model_period = config.logger.get("save_model_period", 20)
        self.steps_per_update = (config.env.parallel.sp_collection + config.env.parallel.xp_collection) * config.train.timesteps_per_update
        if logger_period < 1:
            # Frequency
            self.train_log_period = int(logger_period * config.run.num_timesteps // self.steps_per_update)
        else:
            # Period
            self.train_log_period = logger_period

        self.verbose = config.logger.get("verbose", False)
        self.run = wandb.init(
            project=config.logger.project,
            entity=config.logger.entity,
            config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
            tags=config.logger.get("tags", None),
            notes=config.logger.get("notes", None),
            group=config.logger.get("group", None),
            mode=config.logger.get("mode", None),
            reinit=True,
            )
        self.define_metrics()

    def log(self, data, step=None, commit=False):
        wandb.log(data, step=step, commit=commit)

    def log_item(self, tag, val, step=None, commit=False, **kwargs):
        self.log({tag: val, **kwargs}, step=step, commit=commit)
        if self.verbose:
            print(f"{tag}: {val}")

    def commit(self):
        self.log({}, commit=True)

    def log_xp_matrix(self, tag, mat, step=None, columns=None, rows=None, commit=False, **kwargs):
        if rows is None:
            rows = [str(i) for i in range(mat.shape[0])]
        if columns is None:
            columns = [str(i) for i in range(mat.shape[1])]
        tab = wandb.Table(
                columns=columns,
                data=mat,
                rows=rows
                )
        wandb.log({tag: tab, **kwargs}, step=step, commit=commit)

    def define_metrics(self):
        wandb.define_metric("train_step")
        wandb.define_metric("checkpoint")
        wandb.define_metric("Train/*", step_metric="train_step")
        wandb.define_metric("Returns/*", step_metric="checkpoint")
