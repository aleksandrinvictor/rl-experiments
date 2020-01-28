from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random


class RewardTracker(object):

    def __init__(self, size: int, target=None):
        """ 
            Store last size rewards for printing stats
            Parameters
            ----------
            size: int
                How many rewards to store
            target: float
                Score to achieve. When mean score for the last <size> episodes
                is more than or equal the target the game terminates
        """
        self.target = target
        self.size = size
        self.rewards = []

    def push(self, reward):
        self.rewards.append(reward)

    def get_mean_reward(self):
        """ 
            Returns last self.size mean reward
        """
        return np.mean(self.rewards[-self.size:])

    def is_task_solved(self):
        if self.target is None:
            return False

        if self.get_mean_reward() >= self.target:
            return True
        else:
            return False


class ReplayBuffer(object):
    def __init__(self, size):
        """
        Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)

        # add data to storage
        if len(self) == self._maxsize:
            self._storage.pop(0)

        self._storage.append(data)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = np.random.choice(len(self), batch_size)

        # collect <s,a,r,s',done> for each index
        batch = [self._storage[i] for i in idxes]

        obs_t = np.array([i[0] for i in batch])
        action = np.array([i[1] for i in batch])
        reward = np.array([i[2] for i in batch])
        obs_tp1 = np.array([i[3] for i in batch])
        done = np.array([i[4] for i in batch])

        return obs_t, action, reward, obs_tp1, done
