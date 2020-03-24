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
max_fitness = -float('inf')
while True:
    for genome in cartpole_ai.population:
        total_reward = 0
        for _ in range(N_PLAYS):
            observation = env.reset()
            done = False
            while not done:
                output  = genome(observation)
                output = max(range(len(output)), key=lambda x: output[x])
                observation, reward, done, info = env.step(output)
                total_reward += reward
        total_reward = total_reward / N_PLAYS
        if total_reward > max_fitness:
            max_fitness = total_reward
            best_genome = genome
        genome.fitness = total_reward
    print(f'Generation: {cartpole_ai.generation}, Max Fitness: {max_fitness}')
    plt.cla()
    best_genome.draw()
    plt.draw()
    plt.pause(0.001)
    if best_genome.fitness == 200:
        break
    cartpole_ai.next_generation()

observation = env.reset()
env._max_episode_steps = 500
done = False
total_reward = 0
while not done:
    output  = best_genome(observation)
    output = max(range(len(output)), key=lambda x: output[x])
    observation, reward, done, info = env.step(output)
    env.render()
    total_reward += reward
print(total_reward)

env.close()

