import numpy as np
import copy

class EpisodicExperienceReplay(object):
    """
        Class that encapsulates the storage used for teammate generation and AHT training.
    """
    def __init__(self, ob_shape, act_shape, max_episodes=100000, max_eps_length=20):
        """
            Constructor of the experience replay class.
            :param ob_shape: Observation shape.
            :param act_shape: Action space dimensionality.
            :param max_episodes: Maximum number of stored episodes.
            :param max_eps_length: Maximum length of each episode.
        """

        real_ob_shape = [ob_shape[0]]
        real_ob_shape.extend(list(ob_shape))

        self.num_episodes = 0
        self.pointer = 0
        self.size = 0
        self.max_episodes = max_episodes
        self.max_eps_length = max_eps_length
        self.ob_shape = ob_shape

        obs_shape, acts_shape = [max_episodes, max_eps_length], [max_episodes, max_eps_length]
        obs_shape.extend(list(ob_shape))
        acts_shape.extend(list(act_shape))

        stored_orders_shape = copy.deepcopy(obs_shape)
        stored_orders_shape[-1] = 1

        self.obs = np.zeros(obs_shape)
        self.actions = np.zeros(acts_shape)
        self.next_obs = np.zeros(obs_shape)
        self.dones = np.zeros([max_episodes, max_eps_length])
        self.rewards = np.zeros([max_episodes, max_eps_length, 2])

    def get_num_episodes(self):
        """
            A method that returns the number of episodes stored in the experience replay.
            :return: Number of episodes stored in the replay.
        """
        return self.num_episodes

    def get_size(self):
        """
            A method that returns the number of transitions stored in the experience replay.
            :return: Number of transitions stored in the replay.
        """
        return self.size

    def add_episode(self, obses, acts, rewards, dones, next_obses):
    #def add_episode(self, obses, acts, rewards, total_rew, eps_length, gamma):
        """
            A method to add an episode of experiences into the replay buffer. Target returns that are appended
            with obs are changed in hindsight according to the achieved returns.
            :param obses: List of stored obses.
            :param acts: List of stored acts.
            :param rewards: List of rewards per timestep.
            :param total_rew: The total returns for an episode.
            :param eps_length: The length of an episode.
            :param gamma: The discount rate used.
        """

        eps_length = self.max_eps_length

        rewards = rewards[:eps_length]
        self.obs[self.pointer,:eps_length] = obses[:eps_length]
        self.actions[self.pointer,:eps_length] = acts[:eps_length]
        self.rewards[self.pointer,:eps_length, :] = rewards[:eps_length, :]
        self.next_obs[self.pointer,:eps_length] = next_obses[:eps_length]
        self.dones[self.pointer, :eps_length] = dones[:eps_length]

        self.size = min(self.size + eps_length + self.max_eps_length, self.max_episodes * self.max_eps_length)
        self.num_episodes = min(self.num_episodes + 1, self.max_episodes)
        self.pointer = (self.pointer + 1) % self.max_episodes

    def sample_all(self):
        """
            A method to return everything stored in the buffer.
            :return: Everything contained in buffer.
        """
        return self.obs, self.actions, self.next_obs, self.dones, self.rewards

    def save(self, dir_location):
        """
            A method that stores every variable into disk.
        """
        with open(dir_location+'/obs.npy', 'wb') as f:
            np.save(f, self.obs)
        with open(dir_location + '/n_obs.npy', 'wb') as f:
            np.save(f, self.next_obs)
        with open(dir_location+'/actions.npy', 'wb') as f:
            np.save(f, self.actions)
        with open(dir_location+'/dones.npy', 'wb') as f:
            np.save(f, self.dones)
        with open(dir_location+'/num_episodes.npy', 'wb') as f:
            np.save(f, np.asarray([self.num_episodes]))
        with open(dir_location + '/size.npy', 'wb') as f:
            np.save(f, np.asarray([self.size]))
        with open(dir_location+'/pointer.npy', 'wb') as f:
            np.save(f, np.asarray([self.pointer]))
        with open(dir_location + '/rewards.npy', 'wb') as f:
            np.save(f, self.rewards)

    def load(self, dir_location):
        """
            A method that loads experiences stored within a disk.
        """
        self.obs = np.load(dir_location+"/obs.npy")
        self.next_obs = np.load(dir_location+"n_obs.npy")
        self.actions = np.load(dir_location+"/actions.npy")
        self.num_episodes = np.load(dir_location + "/num_episodes.npy")[0]
        self.pointer = np.load(dir_location + "/pointer.npy")[0]
        self.dones = np.load(dir_location + "/dones.npy")
        self.rewards = np.load(dir_location + "/rewards.npy")
        self.size = np.load(dir_location+"/size.npy")

