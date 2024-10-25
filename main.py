import pygame
import gym
import envs
from models import q_learning
from models import two_layer, replay_buffer
from models import constant_alpha_MC
from models import n_step_sarsa
from models import expected_sarsa
import numpy as np
import pandas as pd
import json
import results.plot_stats as plots



#Funtions for converting observation coords
def lineariseObservation(observation,  gridwidth = 13):
    x, y = observation['agent']
    agent_loc = y * gridwidth + x
    a, b = observation['target']
    target_loc = b * gridwidth + a
    return { "agent": agent_loc, "target": target_loc}

def observation_to_input(observation):
        inputs = np.zeros(52)


        #Would need to figure out how to configure this for more adaptability
        inputs[observation["agent"][0] + 26] = 1
        inputs[observation["agent"][1] + 39] = 1
        inputs[observation["target"][0]] = 1
        inputs[observation["target"][1] + 13] = 1

        return inputs


#Humanised test
def human_test(env):
    done = False

    env.reset()
    env.render(mode='human')

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT: # Up
                    a,b,done,c = env.step(1)
                if event.key == pygame.K_RIGHT: # Down
                    a,b,done,c = env.step(0)
                if event.key == pygame.K_UP: # Left
                    a,b,done,c = env.step(2)
                if event.key == pygame.K_DOWN: #Right
                    a,b,done,c = env.step(3)

                env.render(mode='human')
    
    env.close()

#Monte Carlo
def mc_test(env, rendermode = "none", episodes = 10000):
    state_space = np.array([169, 169])
    action_space = env.action_space.n
    agent = constant_alpha_MC(state_space, action_space)


    # Training loop
    stats = pd.DataFrame(data = np.zeros((episodes,2)), columns=["Steps", "Reward"], index= np.arange(episodes))
    q_values = np.zeros((episodes + 1, state_space[0], state_space[1], action_space))

    for episode in range(episodes):
        print(episode)
        q_values[episode] = agent.q_table
        state, _ = env.reset()
        linear_state = lineariseObservation(state)
        done = False
        while not done:
            action = agent.run_policy(linear_state)
            next_state, reward, done, _ = env.step(action)
            next_linear_state = lineariseObservation(next_state)
            agent.episode_log(linear_state, action, reward)
            linear_state = next_linear_state
            stats.Steps.iloc[episode] += 1
            stats.Reward.iloc[episode] += reward
            if stats.Steps.iloc[episode] >= 1000:
                done = True
            if rendermode == "human":
                env.render(mode = rendermode)
        agent.update_policy()
    
    q_values[episodes] = agent.q_table
    q_values = q_values.reshape((episodes+1, state_space[0]*state_space[1]*action_space))


    return stats, q_values

#Sarsa model
def sarsa_test(env, rendermode = "none", episodes = 10000):

    state_space = np.array([169, 169])
    action_space = env.action_space.n
    agent = n_step_sarsa(state_space, action_space)

    stats = pd.DataFrame(data = np.zeros((episodes,2)), columns=["Steps", "Reward"], index= np.arange(episodes))
    q_values = np.zeros((episodes + 1, state_space[0], state_space[1], action_space))

    #training Loop
    for episode in range(episodes):
        print(episode)
        q_values[episode] = agent.q_table
        state, _ = env.reset()
        linear_state = lineariseObservation(state)
        done = False

        agent.slog = np.append(agent.slog, [[int(linear_state["agent"]), int(linear_state["target"]), int(agent.run_policy(linear_state))]], axis = 0)
        
        while not done:
            action = agent.run_policy(linear_state)
            next_state, reward, done, _ = env.step(action)
            next_linear_state = lineariseObservation(next_state)
            agent.episode_log(linear_state, action, reward)
            linear_state = next_linear_state
            stats.Steps.iloc[episode] += 1
            stats.Reward.iloc[episode] += reward
            if stats.Steps.iloc[episode] >= 1000:
                done = True
            if rendermode == "human":
                env.render(mode = rendermode)
            
            agent.update_policy()
    
    q_values[episodes] = agent.q_table
    q_values = q_values.reshape((episodes+1, state_space[0]*state_space[1]*action_space))


    return stats, q_values

#Q-learning model
def q_learning_test(env, rendermode = "none", episodes = 10000):
    state_space = np.array([169, 169])
    action_space = env.action_space.n
    agent = q_learning(state_space, action_space)


    # Training loop
    stats = pd.DataFrame(data = np.zeros((episodes,2)), columns=["Steps", "Reward"], index= np.arange(episodes))
    q_values = np.zeros((episodes + 1, state_space[0], state_space[1], action_space))

    for episode in range(episodes):
        print(episode)
        q_values[episode] = agent.q_table
        state, _ = env.reset()
        linear_state = lineariseObservation(state)
        done = False
        while not done:
            action = agent.run_policy(linear_state)
            next_state, reward, done, _ = env.step(action)
            next_linear_state = lineariseObservation(next_state)
            agent.update_policy(reward, linear_state, next_linear_state)
            linear_state = next_linear_state
            stats.Steps.iloc[episode] += 1
            stats.Reward.iloc[episode] += reward
            if stats.Steps.iloc[episode] >= 1000:
                done = True
            if rendermode == "human":
                env.render(mode = rendermode)

    
    q_values[episodes] = agent.q_table
    q_values = q_values.reshape((episodes+1, state_space[0]*state_space[1]*action_space))


    return stats, q_values

#Expected sarsa model
def expected_sarsa_test(env, rendermode = "none", episodes = 10000):
    state_space = np.array([169, 169])
    action_space = env.action_space.n
    agent = expected_sarsa(state_space, action_space)


    # Training loop
    stats = pd.DataFrame(data = np.zeros((episodes,2)), columns=["Steps", "Reward"], index= np.arange(episodes))
    q_values = np.zeros((episodes + 1, state_space[0], state_space[1], action_space))

    for episode in range(episodes):
        print(episode)
        q_values[episode] = agent.q_table
        state, _ = env.reset()
        linear_state = lineariseObservation(state)
        done = False
        while not done:
            action = agent.run_policy(linear_state)
            next_state, reward, done, _ = env.step(action)
            next_linear_state = lineariseObservation(next_state)
            agent.update_policy(reward, linear_state, next_linear_state)
            linear_state = next_linear_state
            stats.Steps.iloc[episode] += 1
            stats.Reward.iloc[episode] += reward
            if stats.Steps.iloc[episode] >= 1000:
                done = True
            if rendermode == "human":
                env.render(mode = rendermode)

    
    q_values[episodes] = agent.q_table
    q_values = q_values.reshape((episodes+1, state_space[0]*state_space[1]*action_space))


    return stats, q_values

#DQN model
def deep_q_test(env, rendermode = "none", episodes = 10000):
    pass

#Main
env = gym.make('MouseMaze-v0')

d = open('maze/envs/maze_structures.json')
mazes = json.load(d)


maze_choice = 0 #int(input("Which maze: 0, 1, 2, 3?"))
if maze_choice == 0:
    env.bridges_set()
else:
    env.bridges_set(mazes[f'maze_{maze_choice}']['structure'])
env.reset()

run_test = 'S' #input("Run which tests: N = none, Q = Q-learning, H = human, L = 2-layer")
    
if run_test == 'H':
    human_test(env)
elif run_test == 'M':
    stats, q_values = mc_test(env)
    pd.DataFrame(q_values).to_csv('.//results//mc_q_values.csv')
    stats.to_csv('.//results//mc_stats.csv')
elif run_test == 'S':
    stats, q_values = sarsa_test(env)
    pd.DataFrame(q_values).to_csv('./maze/results/sarsa_q_values.csv')
    stats.to_csv('./maze/results/sarsa_stats.csv')
elif run_test == 'Q': #1 step only, could modify for n-step
    stats, q_values = q_learning_test(env)
    pd.DataFrame(q_values).to_csv('./maze/results/q_learning_q_values.csv')
    stats.to_csv('./maze/results/q_learning_stats.csv')
elif run_test == 'E': #1 step only, could modify for n-step
    stats, q_values = expected_sarsa_test(env)
    pd.DataFrame(q_values).to_csv('./maze/results/e_sarsa_q_values.csv')
    stats.to_csv('./maze/results/e_sarsa_stats.csv')
