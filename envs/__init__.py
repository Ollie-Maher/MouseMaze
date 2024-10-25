import gym
#from maze_env import MouseMaze

# Define an __all__
__all__ = ['MouseMaze']


# Register MouseMaze for gym.make

def register_envs():
    gym.envs.registration.register(
        id='MouseMaze-v0',
        entry_point='envs.maze_env:MouseMaze'
    )


register_envs()
