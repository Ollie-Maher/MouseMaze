import numpy as np

class q_learning:

    def __init__(self, state_space, action_space, alpha=0.01, gamma=0.95, epsilon=0.1):

        self.action_space = action_space
        self.q_table = np.zeros((state_space[0], state_space[1], action_space))

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def run_policy(self, state, exploit = False):
        if np.random.rand() >= self.epsilon or exploit: #Exploit
            return np.argmax(self.q_table[state["agent"], state["target"]])  
            
        else: #Explore
            return np.random.randint(self.action_space) 

    def update_policy(self, reward, state, next_state):        
        self.q_table[state["agent"], state["target"]] = (1-self.alpha) * self.q_table[state[0], state[1]] + self.alpha * (reward + self.gamma * np.amax(self.q_table[next_state["agent"], next_state["target"]]))
