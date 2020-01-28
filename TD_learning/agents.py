from collections import defaultdict
import random
import math
import numpy as np


class QLearningAgent:
    def __init__(self, alpha, epsilon, discount, get_legal_actions):
        """
        Q-Learning Agent
        based on https://inst.eecs.berkeley.edu/~cs188/sp19/projects.html
        and Practical_RL https://github.com/yandexdataschool/Practical_RL
        """

        self.get_legal_actions = get_legal_actions
        self._qvalues = defaultdict(lambda: defaultdict(lambda: 0))
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount

    def get_qvalue(self, state, action):
        """ Returns Q(state,action) """
        return self._qvalues[state][action]

    def set_qvalue(self, state, action, value):
        """ Sets the Qvalue for [state,action] to the given value """
        self._qvalues[state][action] = value

    def get_value(self, state):
        """
        Compute your agent's estimate of V(s) using current q-values
        V(s) = max_over_action Q(state,action) over possible actions.
        """
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return 0.0
        if len(possible_actions) == 0:
            return 0.0

        value = max(self.get_qvalue(state, action)
                    for action in possible_actions)

        return value

    def update(self, state, action, reward, next_state):
        """
           Q-Value update here:
           Q(s,a) := (1 - alpha) * Q(s,a) + alpha * (r + gamma * max_a Q(s',a'))
        """

        # agent parameters
        gamma = self.discount
        learning_rate = self.alpha

        q_s_a_hat = reward + gamma * self.get_value(next_state)
        q_s_a = learning_rate * q_s_a_hat + \
            (1 - learning_rate) * self.get_qvalue(state, action)

        self.set_qvalue(state, action, q_s_a)

    def get_best_action(self, state):
        """
        Compute the best action to take in a state (using current q-values). 
        """
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        best_action = None
        curr_best_q = None

        for action in possible_actions:
            q_s_a = self.get_qvalue(state, action)
            if curr_best_q is None or q_s_a > curr_best_q:
                best_action = action
                curr_best_q = q_s_a

        return best_action

    def get_action(self, state):
        """
        Compute the action to take in the current state, including exploration.  
        With probability self.epsilon, we should take a random action.
            otherwise - the best policy action (self.get_best_action).
        """

        # Pick Action
        possible_actions = self.get_legal_actions(state)
        action = None

        # If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        # agent parameters:
        epsilon = self.epsilon

        if epsilon >= random.uniform(0, 1):
            chosen_action = random.choice([a for a in possible_actions])
        else:
            chosen_action = self.get_best_action(state)

        return chosen_action


class EVSarsaAgent(QLearningAgent):
    """ 
    An agent that changes some of q-learning functions to implement Expected Value SARSA. 
    Note: this demo assumes that your implementation of QLearningAgent.update uses get_value(next_state).
    If it doesn't, please add
        def update(self, state, action, reward, next_state):
            and implement it for Expected Value SARSA's V(s')
    """

    def get_value(self, state):
        """ 
        Returns Vpi for current state under epsilon-greedy policy:
          V_{pi}(s) = sum _{over a_i} {pi(a_i | s) * Q(s, a_i)}

        Hint: all other methods from QLearningAgent are still accessible.
        """
        epsilon = self.epsilon
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return 0.0
        if len(possible_actions) == 0:
            return 0.0

        if epsilon >= random.uniform(0, 1):
            pi = [1 / len(possible_actions)] * len(possible_actions)
        else:
            pi = [0] * len(possible_actions)
            best_action = self.get_best_action(state)
            pi[best_action] = 1

        state_value = sum(pi[action] * self.get_qvalue(state, action)
                          for action in possible_actions)

        return state_value
