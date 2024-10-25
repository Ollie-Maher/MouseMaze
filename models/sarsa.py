import numpy as np

class n_step_sarsa:

    def __init__(self, state_space, action_space, update_step = 10, alpha=0.01, gamma=0.95, epsilon=0.1):

        self.action_space = action_space
        self.q_table = np.zeros((state_space[0], state_space[1], action_space))

        self.update_step = update_step
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        #log of rewards
        self.rlog = np.empty(0)
        #log of states
        self.slog = np.empty((0,3),dtype=int)

    def episode_log(self, state, action, reward):
        if self.rlog.size == self.update_step: #cycles log when full
            self.rlog = np.delete(self.rlog, [0])
            self.rlog = np.append(self.rlog, reward)
            self.slog = np.delete(self.slog, [0], axis=0)
            self.slog = np.append(self.slog, [[int(state["agent"]), int(state["target"]), int(action)]], axis = 0)
        else: #filling log
            self.rlog = np.append(self.rlog, reward)
            self.slog = np.append(self.slog, [[int(state["agent"]), int(state["target"]), int(action)]], axis = 0)

    def run_policy(self, state, exploit = False):
        if np.random.rand() >= self.epsilon or exploit: #Exploit
            return np.argmax(self.q_table[state["agent"], state["target"]])
            
        else: #Explore
            return np.random.randint(self.action_space) 

    def update_policy(self):
        if self.rlog.size == self.update_step:
            g = 0
            for i in range(self.update_step):
                g += self.rlog[i] * (self.gamma ** i)
            g += (self.gamma ** self.update_step) * self.q_table[self.slog[-1,0],self.slog[-1,1],self.slog[-1,2]]

            self.q_table[self.slog[0,0],self.slog[0,1],self.slog[0,2]] = self.q_table[self.slog[0,0],self.slog[0,1],self.slog[0,2]] + self.alpha * (g - self.q_table[self.slog[0,0],self.slog[0,1],self.slog[0,2]])
