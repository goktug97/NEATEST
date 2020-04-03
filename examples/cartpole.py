#!/usr/bin/env python3

import gym
import matplotlib.pyplot as plt

import neat

env = gym.make('CartPole-v0')

cartpole_ai = neat.NEAT(n_networks = 150,
                        input_size = env.observation_space.shape[0],
                        output_size = env.action_space.n,
                        bias = True,
                        c1 = 1.0, c2 = 1.0, c3 = 0.4,
                        distance_threshold = 0.5,
                        weight_mutation_rate = 0.8,
                        node_mutation_rate = 0.03,
                        connection_mutation_rate = 0.05,
                        interspecies_mating_rate = 0.001,
                        disable_rate = 0.75,
                        noise_magnitude = 0.01,
                        stegnant_threshold = float('inf'))

N_PLAYS = 12
while True:
    rewards = [] 
    print(f'Generation: {cartpole_ai.generation}')
    for genome in cartpole_ai.population:
        total_reward = 0
        for _ in range(N_PLAYS):
            observation = env.reset()
            env._max_episode_steps = 500
            done = False
            while not done:
                output  = genome(observation)
                output = max(range(len(output)), key=lambda x: output[x])
                observation, reward, done, info = env.step(output)
                total_reward += reward
        total_reward = total_reward / N_PLAYS
        rewards.append(total_reward)
    cartpole_ai.next_generation(rewards)
    print(f'Max Reward Session: {cartpole_ai.best_fitness}')
    print(f'Max Reward Step: {max(rewards)}')

    plt.cla()
    cartpole_ai.best_genome.draw()
    plt.draw()
    plt.pause(0.001)

    if cartpole_ai.best_fitness == 500:
        break

env = gym.wrappers.Monitor(env, '.', force = True)
observation = env.reset()
env._max_episode_steps = 500
done = False
total_reward = 0
while not done:
    output  = cartpole_ai.best_genome(observation)
    output = max(range(len(output)), key=lambda x: output[x])
    observation, reward, done, info = env.step(output)
    env.render()
    total_reward += reward
print(total_reward)
env.close()

