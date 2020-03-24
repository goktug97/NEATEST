import random

import matplotlib.pyplot as plt
import numpy as np
import cv2

import snake
import neat

SEED = 123
PLAYBACK = True
MAP_SIZE = 6

np.random.seed(SEED)
random.seed(SEED)

snake_ai = neat.NEAT(n_networks = 200,
                     input_size = (MAP_SIZE-2)**2 + 6,
                     output_size = 3,
                     bias = True,
                     c1 = 1.0, c2 = 1.0, c3 = 0.4,
                     distance_threshold = 3.0,
                     weight_mutation_rate = 0.8,
                     node_mutation_rate = 0.03,
                     connection_mutation_rate = 0.05,
                     interspecies_mating_rate = 0.001,
                     disable_rate = 0.75,
                     stegnant_threshold = 15,
                     input_activation = neat.steepened_sigmoid,
                     hidden_activation = neat.steepened_sigmoid,
                     output_activation = neat.steepened_sigmoid)

snake_game = snake.Game(MAP_SIZE, 3)
ACTIONS = [-1, 0, 1]

while True:
    best_fitness = -float('inf')
    for genome in snake_ai.population:
        snake_game.reset()
        length = len(snake_game.snake.body)
        playback = []
        step = 0
        while not snake_game.done:
            screen = snake_game.draw(50)
            playback.append(screen)
            game_map = snake_game.map[1:-1, 1:-1]
            game_map = game_map.flatten()
            observation = np.concatenate(
                [game_map, snake_game.apple, snake_game.snake.head,
                 [snake_game.snake.x_direction, snake_game.snake.x_direction]])
            output = genome(observation)
            snake_game.step(ACTIONS[np.argmax(output)])
            if length != len(snake_game.snake.body):
                length = len(snake_game.snake.body)
                step = 0
            else:
                step +=1
            if step > (MAP_SIZE-2)**2:
                break

        score = np.sum(snake_game.map[1:-1, 1:-1])
        if snake_game.won:
            score = score ** 2
        elif snake_game.done:
            score -= 1
        genome.fitness = score
        if genome.fitness > best_fitness:
            best_fitness = genome.fitness
            best_game = playback
            best_genome = genome

    plt.cla()
    best_genome.draw()
    plt.draw()
    plt.pause(0.001)

    if PLAYBACK:
        for screen in best_game:
            cv2.imshow('cvwindow', screen)
            key = cv2.waitKey(200)
            if key == 27:
                break
