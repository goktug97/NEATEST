#!/usr/bin/env python3

import gym
import matplotlib.pyplot as plt

import neat

cartpole_ai = neat.NEAT(n_networks = 300,
                        input_size = 4,
                        output_size = 1,
                        bias = True,
                        c1 = 1.0, c2 = 1.0, c3 = 4.0,
                        distance_threshold = 2.5,
                        weight_mutation_rate = 0.8,
                        node_mutation_rate = 0.3,
                        connection_mutation_rate = 0.5,
                        interspecies_mating_rate = 0.001,
                        disable_rate = 0.75,
                        noise_magnitude = 0.01,
                        stegnant_threshold = float('inf'),
                        input_activation = neat.steepened_sigmoid,
                        hidden_activation = neat.steepened_sigmoid,
                        output_activation = neat.steepened_sigmoid)

env = gym.make('CartPole-v0')
max_fitness = -float('inf')
while True:
    for genome in cartpole_ai.population:
        obsvervation = env.reset()
        done = False
        total_reward = 0
        while not done:
            output  = int(round(genome(obsvervation)[0]))
            observation, reward, done, info = env.step(output)
            total_reward += reward
        genome.fitness = total_reward
        if total_reward > max_fitness:
            max_fitness = total_reward
            best_genome = genome
    print(f'Generation: {cartpole_ai.generation}, Max Fitness: {max_fitness}')
    plt.cla()
    best_genome.draw()
    plt.draw()
    plt.pause(0.001)
    cartpole_ai.next_generation()
env.close()

