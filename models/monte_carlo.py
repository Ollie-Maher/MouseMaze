import numpy as np

class constant_alpha_MC:

    def __init__(self, state_space, action_space,  alpha=0.001, gamma=0.95, epsilon=0.1):

        self.action_space = action_space
        self.q_table = np.zeros((state_space[0], state_space[1], action_space))

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.rlog = np.empty(0)
        self.slog = np.empty((0,3),dtype=int)

    def episode_log(self, state, action, reward):
        #log of rewards
        self.rlog = np.append(self.rlog, reward)
        #log of states
        self.slog = np.append(self.slog, [[int(state["agent"]), int(state["target"]), int(action)]], axis = 0) 

    def run_policy(self, state, exploit = False):
        if np.random.rand() >= self.epsilon or exploit: #Exploit
            return np.argmax(self.q_table[state["agent"], state["target"]])  
            
        else: #Explore
            return np.random.randint(self.action_space) 

    def update_policy(self):
        for t in range(self.rlog.size):
            g = 0
            for i in range((self.rlog.size - t)):
                g += self.rlog[t+i] * (self.gamma ** i)

            self.q_table[self.slog[t,0],self.slog[t,1],self.slog[t,2]] = self.q_table[self.slog[t,0],self.slog[t,1],self.slog[t,2]] + self.alpha * (g - self.q_table[self.slog[t,0],self.slog[t,1],self.slog[t,2]])
        self.rlog = np.empty(0)
        self.slog = np.empty((0,3), dtype=int)
