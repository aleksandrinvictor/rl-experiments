import gym
import numpy as np


class Sampler:

    def __init__(
        self,
        env: gym.Env,
        n_steps: int = 3
    ):

        self.env = env
        self.current_state = env.reset()
        self.n_steps = n_steps

    def sample(self, agent):
        trajectories = []

        states_t, actions, rewards, states_tp1 = [], [], [], []

        for j in range(self.n_steps):
            action = agent.get_actions(np.array([self.current_state]))
            next_state, reward, done, _ = self.env.step(action)

            states_t.append(self.current_state)
            actions.append(action)
            rewards.append(reward)
            states_tp1.append(next_state)

            self.current_state = next_state
            if done:
                self.current_state = self.env.reset()
                break

        return {
            'states_t': states_t,
            'actions': actions,
            'rewards': rewards,
            'states_tp1': states_tp1,
        }
