import random

import matplotlib.pyplot as plt
import numpy as np
import cv2

import snake
import neat

PLAYBACK = True
MAP_SIZE = 6
VISION_RADIUS = 1

snake_ai = neat.NEAT(n_networks = 300,
                     input_size = (VISION_RADIUS * 2 + 1) ** 2 - 2,
                     output_size = 3,
                     bias = True,
                     c1 = 1.0, c2 = 1.0, c3 = 4.0,
                     distance_threshold = 2.5,
                     weight_mutation_rate = 0.8,
                     node_mutation_rate = 0.3,
                     connection_mutation_rate = 0.5,
                     interspecies_mating_rate = 0.001,
                     disable_rate = 0.75,
                     stegnant_threshold = 15)

snake_game = snake.Game(MAP_SIZE, 3)
ACTIONS = [-1, 0, 1]

while True:
    rewards = []
    for genome in snake_ai.population:
        snake_game.reset()
        length = len(snake_game.snake.body)
        playback = []
        step = 0
        while not snake_game.done:
            screen = snake_game.draw(50)
            playback.append(screen)

            map = snake_game.map.copy()
            map[snake_game.apple[1], snake_game.apple[0]] = -1
            body = np.array(snake_game.snake.body)
            map = np.pad(map, ((VISION_RADIUS-1, VISION_RADIUS-1),
                               (VISION_RADIUS-1, VISION_RADIUS-1)),
                         constant_values=1)
            vision = map[
                snake_game.snake.head[1]-1:
                snake_game.snake.head[1]+2*VISION_RADIUS,
                snake_game.snake.head[0]-1:
                snake_game.snake.head[0]+2*VISION_RADIUS]
            if snake_game.snake.x_direction == -1:
                vision = np.fliplr(vision)
                vision = np.flipud(vision)
            if snake_game.snake.y_direction == -1:
                vision = vision.T
                vision = np.fliplr(vision)
            if snake_game.snake.y_direction == 1:
                vision = vision.T
                vision = np.flipud(vision)
            vision = vision.flatten()
            vision = np.delete(vision, (VISION_RADIUS * 2 + 1) *
                               VISION_RADIUS + VISION_RADIUS - 1)
            vision = np.delete(vision, (VISION_RADIUS * 2 + 1) *
                               VISION_RADIUS + VISION_RADIUS - 1)
            output = genome(vision)
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
        rewards.append(score)
        if genome.fitness > snake_ai.best_fitness:
            best_game = playback

    snake_ai.next_generation(rewards)

    plt.cla()
    snake_ai.best_genome.draw(horizontal_distance=5.0)
    plt.draw()
    plt.pause(0.001)

    if PLAYBACK:
        for screen in best_game:
            cv2.imshow('cvwindow', screen)
            key = cv2.waitKey(200)
            if key == 27:
                break
