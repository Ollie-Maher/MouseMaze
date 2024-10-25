import gym
from gym import spaces
import numpy as np
import pygame

#Bridge strings for testing: "A1-A2 A2-A3 A3-A4 A4-A5 A5-A6 A6-A7 A1-B1 B1-C1 C1-D1 D1-E1 E1-F1 F1-G1", "all"

class MouseMaze(gym.Env):
    metadata = {'render_modes': ['human']}
    
    def __init__(self, size = 13): #Initialise environment
        super(MouseMaze, self).__init__()
        
        self.size = size  #Size of the grid

        #Define action space and directions
        self.action_space = spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([0, 1]), #UP
            1: np.array([0, -1]), #DOWN
            2: np.array([-1, 0]), #LEFT
            3: np.array([1, 0]), #RIGHT
        }


        #Setting up observation space
        #Not entirely clear why this is here, is present in gridworld example, has no clear unique use
        self.observation_space = spaces.Dict(
            {
            "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
        })

        #Initialize the grid map
        self.grid = np.zeros((self.size, self.size), dtype=object)
        
        #Set up platforms at each odd, odd-numbered grid space
        for i in range(0, self.size, 2):
            for j in range(0, self.size, 2):
                self.grid[i, j] = 'P'  # 'P' for platform

        #pygame.Surface used for 'render()'  
        self.viewer = None
        
    def _get_obs(self): #Returns observations for **agent**
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self): #Returns information for **user**
        return {}

    def step(self, action): #Takes agent action, runs environment for 1 timestep, returns obs, rewards, done, info
        reward = -1  #Penalize each step
        done = False
        
        direction = self._action_to_direction[action] #

        next_pos = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        if self.grid[next_pos[0], next_pos[1]] == 'B' or self.grid[next_pos[0], next_pos[1]] == 'P':
            self._agent_location = next_pos

        
        observation = self._get_obs()
        info = self._get_info()

        #Check if reached the goal
        if all(self._agent_location == self._target_location):
            done = True
            reward = 0
        
        return observation, reward, done, info
    
    def reset(self, seed = None): #Resets environment, returns obs, info
        super().reset(seed = seed)

        #Set agent and target location
        self._agent_location = np.array([self.np_random.integers(0,self.size // 2 + 1, dtype=int) * 2, self.np_random.integers(0,self.size // 2 + 1, dtype=int) * 2])
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = np.array([self.np_random.integers(0,self.size // 2 + 1, dtype=int) * 2, self.np_random.integers(0,self.size // 2 + 1, dtype=int) * 2])

        observation = self._get_obs()
        info = self._get_info()


        return observation, info
    
    def bridges_set(self, bridges=[]): #Set up bridges

        if len(bridges) == 0: #Sets all bridge locations to 'B'
            for i in range(1,self.size,2):
                for j in range(1,self.size,2):
                    self.grid[i,j-1] = 'B'
                    self.grid[i-1,j] = 'B'
                    self.grid[i,12] = 'B'
                    self.grid[12,j] = 'B'

        else: #Takes each coord pair and sets midpoint to 'B'
            for bridge in bridges:
                plat1, plat2 = bridge.split('-') #Split into platform pairs
                row1 = 2 * (ord(plat1[0].upper()) - ord('A'))

                col1 = 2 * (int(plat1[1:]) - 1 ) # Subtract 1 because grid is 0-indexed

                row2 = 2 *( ord(plat2[0].upper()) - ord('A'))

                col2 = 2 * (int(plat2[1:]) - 1)  # Subtract 1 because grid is 0-indexed

                print((row1+row2)/2)
                print((col1+col2)/2)

                self.grid[int((row1+row2)/2),int((col1+col2)/2)] = 'B' # Set midpoint to bridge
        
    def render(self, mode='human', close=False):
        if close:
            pygame.quit()
            return

        screen_size = 600  # Size of the window
        grid_size = screen_size // self.size  # Size of each grid square

        if self.viewer is None:
            pygame.init()
            self.viewer = pygame.display.set_mode((screen_size, screen_size))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        self.viewer.fill((255, 255, 255))  # Fill the screen with white

        # Draw the grid, platforms, and bridges
        for i in range(self.size):
            for j in range(self.size):
                rect = pygame.Rect(j * grid_size, i * grid_size, grid_size, grid_size)
                if self.grid[i, j] == 'P':
                    pygame.draw.rect(self.viewer, (0, 0, 255), rect)  # Draw platforms in blue
                elif self.grid[i, j] == 'B':
                    pygame.draw.rect(self.viewer, (165, 42, 42), rect)  # Draw bridges in brown
                pygame.draw.rect(self.viewer, (0, 0, 0), rect, 1)  # Draw grid lines

        # Draw the agent
        agent_rect = pygame.Rect(self._agent_location[1] * grid_size, self._agent_location[0] * grid_size, grid_size, grid_size)
        pygame.draw.rect(self.viewer, (0, 255, 0), agent_rect)  # Draw agent in green

        # Draw the target
        target_rect = pygame.Rect(self._target_location[1] * grid_size, self._target_location[0] * grid_size, grid_size, grid_size)
        pygame.draw.rect(self.viewer, (255, 0, 0), target_rect)  # Draw target in red

        pygame.display.flip()

    def close(self):
        pygame.quit()