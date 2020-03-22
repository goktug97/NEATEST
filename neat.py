#!/usr/bin/env python3
import random
import copy
from typing import Union, List, Callable
from itertools import groupby
import functools
import math
import statistics

from genome import Genome
from node import Node, NodeType
from connection import Connection

class ContextGenome(Genome):
    def __init__(self, nodes: List[Node], connections: List[Connection]):
        self.species = None
        self.fitness = 0
        super().__init__(nodes, connections)

def Context(genome: Genome) -> ContextGenome:
    return ContextGenome(genome.nodes, genome.connections)

def random_genome(input_size: int, output_size: int, bias: bool = False,
                  activation: Callable[[float], float] = lambda x: x) -> ContextGenome:
    '''Create fc neural network without hidden layers with random weights.'''
    connections: List[Connection] = []
    input_nodes = [Node(i, NodeType.INPUT, activation) for i in range(input_size)]
    if bias:
        input_nodes.append(Node(input_size, NodeType.BIAS, value=1.0))
    output_nodes = [Node(i+len(input_nodes), NodeType.OUTPUT, activation)
                    for i in range(output_size)]
    for i in range(len(input_nodes)):
        input_node = input_nodes[i]
        for j in range(output_size):
            output_node = output_nodes[j]
            connections += [Connection(input_node, output_node, random.random())]
    return ContextGenome(input_nodes+output_nodes, connections)

def create_population(size: int, input_size: int,
                      output_size: int, bias: bool,
                      activation:Callable[[float], float]=
                      lambda x: x) -> List[ContextGenome]:
    population: List[ContextGenome] = []
    for _ in range(size):
        population.append(random_genome(
            input_size, output_size,
            bias=bias, activation=activation))
    return population

def steepened_sigmoid(x):
    return 1 / (1 + math.exp(-4.9 * x))

@functools.total_ordering
class Species(object):
    id: int = 0
    def __init__(self, representer: Genome):
        self.genomes: List[Genome] = [representer]
        self.representer: Genome = representer
        self.species_id: int = Species.assign_id()
        self.max_fitness = representer.fitness
        self.total_adjusted_fitness = 0
        self.stegnant = 0

    def add_genome(self, genome: Genome) -> None:
        self.genomes.append(genome)
        if genome.fitness > self.max_fitness:
            self.max_fitness = genome.fitness
            self.stegnant = 0
        else:
            self.stegnant += 1

    def reset(self):
        self.total_adjusted_fitness = 0
        self.representer = random.choice(self.genomes)
        self.genomes = []

    @classmethod
    def assign_id(cls):
        cls.id += 1
        return cls.id

    def __hash__(self):
        return hash(str(Species.id))

    def __gt__(self, other):
        return self.id > other.id


class NEAT(object):
    def __init__(self, size: int, input_size: int, output_size: int):
        self.population = create_population(
            size, input_size, output_size, True,
            activation = steepened_sigmoid)
        self.generation = 0
        self.species = [Species(random.choice(self.population))]
        self.update_species()

    def update_species(self):
        THRESHOLD = 4.0
        for species in self.species:
            species.reset()
        for genome in self.population:
            known = False
            for species in self.species:
                if genome.distance(species.representer, 1.0, 1.0, 3.0) < THRESHOLD:
                    species.add_genome(genome)
                    genome.species = species
                    known = True
                    break
            if not known: 
                new_species = Species(genome)
                self.species.append(new_species)
                genome.species = new_species

        # Remove if a species is empty
        for idx in reversed(range(len(self.species))):
            if not len(self.species[idx].genomes):
                self.species.pop(idx)
        print(len(self.species))

    def next_generation(self):
        self.update_species()
        self.adjust_fitness_scores()

        population = []

        sorted_population = sorted(
            self.population, key = lambda x: x.species.species_id)
        grouped_population = [list(it) for k, it in groupby(
            sorted_population, lambda x: x.species.species_id)]

        # Preserve champions in the species
        for group in grouped_population:
            if len(group) > 5:
                best_genome = max(group, key = lambda x: x.fitness)
                population.append(best_genome)

        while len(population) < len(self.population):
            species = self.get_random_species()
            genome_1 = self.get_random_genome(species)
            species = self.get_random_species() if random.random() < 0.001 else species
            genome_2 = self.get_random_genome(species)
            new_genome = Context(genome_1.crossover(genome_2))
            if random.random() < 0.8:
                new_genome.weight_mutation()
            if random.random() < 0.03:
                new_genome.add_node_mutation()
            if random.random() < 0.3:
                new_genome.add_connection_mutation()
            population.append(new_genome)

        self.population = population
        self.generation += 1

    def adjust_fitness_scores(self) -> None:
        for genome in self.population:
            genome.fitness = genome.fitness / len(genome.species.genomes)
            genome.species.total_adjusted_fitness += genome.fitness


    def get_random_species(self) -> Species:
        total = 0
        for species in self.species:
            if species.stegnant < 15:
                total += species.total_adjusted_fitness
        r = random.uniform(0, total)
        upto = 0
        for species in self.species:
            if upto + species.total_adjusted_fitness >= r:
                return species
            upto += species.total_adjusted_fitness

    @staticmethod
    def get_random_genome(species: Species) -> Genome:
        total = sum(genome.fitness for genome in species.genomes)
        r = random.uniform(0, total)
        upto = 0
        for genome in species.genomes:
            if upto + genome.fitness >= r:
                return genome
            upto += genome.fitness


if __name__ == '__main__':
    # TESTING
    import matplotlib.pyplot as plt
    import snake
    import numpy as np
    import cv2

    # Settings
    SEED = 123
    VISION_RADIUS = 1
    MAP_SIZE = 6 # Size of the map including walls
    SNAKE_SIZE = 3 # Starting size
    INPUT_SIZE = (VISION_RADIUS * 2 + 1) ** 2 - 2
    N_NETWORKS = 1000
    BEST_PERCENT = 0.1
    PLAYBACK = True

    np.random.seed(SEED)
    random.seed(SEED)

    assert SNAKE_SIZE >= MAP_SIZE-3

    OUTPUT_SIZE = 3 # If you change this you have to change L139
    ACTIONS = [-1, 0, 1] # Left, Straight, Right

    neat = NEAT(N_NETWORKS, INPUT_SIZE, OUTPUT_SIZE)

    while True:
        game = snake.Game(MAP_SIZE, SNAKE_SIZE)
        best_fitness = -float('Inf')
        for genome in neat.population:
            new_game = copy.deepcopy(game)
            length = len(new_game.snake.body)
            playback = []
            step = 0
            while not new_game.done:
                screen = new_game.draw(50)
                playback.append(screen)

                game_map = new_game.map.copy()
                game_map[new_game.apple[1], new_game.apple[0]] = -1
                body = np.array(new_game.snake.body)
                game_map = np.pad(game_map, ((VISION_RADIUS-1, VISION_RADIUS-1), 
                                   (VISION_RADIUS-1, VISION_RADIUS-1)),
                             constant_values=1)
                vision = game_map[
                    new_game.snake.head[1]-1:
                    new_game.snake.head[1]+2*VISION_RADIUS,
                    new_game.snake.head[0]-1:
                    new_game.snake.head[0]+2*VISION_RADIUS]

                if new_game.snake.x_direction == -1:
                    vision = np.fliplr(vision)
                    vision = np.flipud(vision)
                if new_game.snake.y_direction == -1:
                    vision = vision.T
                    vision = np.fliplr(vision)
                if new_game.snake.y_direction == 1:
                    vision = vision.T
                    vision = np.flipud(vision)

                vision = vision.flatten()

                vision = np.delete(vision, (VISION_RADIUS * 2 + 1) *
                                   VISION_RADIUS + VISION_RADIUS - 1)
                vision = np.delete(vision, (VISION_RADIUS * 2 + 1) *
                                   VISION_RADIUS + VISION_RADIUS - 1)

                input = np.concatenate([vision, [1]])

                output = genome(input)

                action = ACTIONS[np.argmax(output)]
                new_game.step(action)

                if len(new_game.snake.body) != length:
                    step = 0
                    length = len(new_game.snake.body)
                else:
                    step += 1
                if step > MAP_SIZE ** 2:
                    break
            score = len(new_game.snake.body)
            genome.fitness = score
            if score > best_fitness:
                best_genome = genome
                best_game = playback
                best_fitness = score
        best_genome.draw()
        plt.draw()
        plt.pause(0.001)
        neat.next_generation()

        if PLAYBACK:
            for screen in best_game:
                cv2.imshow('cvwindow', screen)
                key = cv2.waitKey(200)
                if key == 27:
                    break
