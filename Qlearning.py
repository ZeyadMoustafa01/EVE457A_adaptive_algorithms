import numpy as np

class QLearningTable:
    def __init__(self, num_states, num_actions, learning_rate=0.1, discount_factor=0.9, exploration_prob=0.1):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.num_states = num_states
        self.num_actions = num_actions

        # Initialize Q-table with zeros
        self.q_table = np.zeros((num_states, num_actions))

    def select_action(self, state):
        # Epsilon-greedy policy to select action
        if np.random.rand() < self.exploration_prob:
            # Exploration: Choose a random action
            return np.random.randint(self.num_actions)
        else:
            # Exploitation: Choose action with the highest Q-value
            return np.argmax(self.q_table[state])

    def update_q_value(self, state, action, reward, next_state):
        # Q-value update based on the Bellman equation
        current_q = self.q_table[state, action]
        max_future_q = np.max(self.q_table[next_state, :])
        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount_factor * max_future_q)
        self.q_table[state, action] = new_q