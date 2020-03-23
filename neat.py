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


def steepened_sigmoid(x):
    return 1 / (1 + math.exp(-4.9 * x))


class ContextGenome(Genome):
    '''Genome class that holds data which depends on the context.'''
    def __init__(self, nodes: List[Node], connections: List[Connection]):
        self.species: Species
        self.fitness: float = 0.0
        self.generation: int = 0
        super().__init__(nodes, connections)

    def crossover(self, other: Genome) -> Genome:
        new_genome = super(ContextGenome, self).crossover(other)
        return ContextGenome(new_genome.nodes, new_genome.connections)


@functools.total_ordering
class Species(object):
    id: int = 0
    def __init__(self, representer: ContextGenome):
        self.genomes: List[ContextGenome] = [representer]
        self.representer: ContextGenome = representer
        self.species_id: int = Species.assign_id()
        self.max_fitness: float = representer.fitness
        self.improved_generation: int = 0
        self.total_adjusted_fitness: float = 0.0
        self.stegnant: int = 0

    def add_genome(self, genome: ContextGenome) -> None:
        self.genomes.append(genome)
        if genome.fitness > self.max_fitness:
            self.stegnant = 0
            self.improved_generation = genome.generation
            self.max_fitness = genome.fitness
        else:
            self.stegnant = genome.generation - self.improved_generation

    def reset(self) -> None:
        self.total_adjusted_fitness = 0.0
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
    def __init__(self, n_networks: int, input_size: int, output_size: int,
                 bias: bool, c1: float, c2: float, c3: float,
                 distance_threshold: float, 
                 weight_mutation_rate: float,
                 node_mutation_rate: float,
                 connection_mutation_rate: float,
                 interspecies_mating_rate: float = 0.001,
                 disable_rate: float = 0.75,
                 stegnant_threshold: int = 15,
                 input_activation: Callable[[float], float]=steepened_sigmoid,
                 hidden_activation: Callable[[float], float]=steepened_sigmoid,
                 output_activation: Callable[[float], float]=steepened_sigmoid):

        self.n_networks = n_networks
        self.input_size = input_size
        self.output_size = output_size
        self.bias = bias
        self.c1, self.c2, self.c3 = c1, c2, c3
        self.distance_threshold = distance_threshold
        self.disable_rate = disable_rate
        self.weight_mutation_rate = weight_mutation_rate
        self.interspecies_mating_rate = interspecies_mating_rate
        self.node_mutation_rate = node_mutation_rate
        self.connection_mutation_rate = connection_mutation_rate
        self.stegnant_threshold = stegnant_threshold
        self.input_activation = input_activation
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        self.generation = 0

        self.create_population()
        self.species = [Species(random.choice(self.population))]
        self.update_species()

    def random_genome(self) -> ContextGenome:
        '''Create fc neural network without hidden layers with random weights.'''
        connections: List[Connection] = []
        input_nodes = [Node(i, NodeType.INPUT, self.input_activation)
                       for i in range(self.input_size)]
        if self.bias:
            input_nodes.append(Node(self.input_size, NodeType.BIAS, value=1.0))
        output_nodes = [Node(i+len(input_nodes), NodeType.OUTPUT,
                             self.output_activation)
                        for i in range(self.output_size)]
        for i in range(len(input_nodes)):
            input_node = input_nodes[i]
            for j in range(self.output_size):
                output_node = output_nodes[j]
                connections += [Connection(input_node, output_node,
                                           random.uniform(-1.0, 1.0))]
        return ContextGenome(input_nodes+output_nodes, connections)

    def create_population(self) -> None:
        population: List[ContextGenome] = []
        for _ in range(self.n_networks):
            population.append(self.random_genome())
        self.population = population

    def update_species(self):
        for species in self.species:
            species.reset()
        for genome in self.population:
            known = False
            for species in self.species:
                if genome.distance(
                        species.representer,
                        self.c1, self.c2, self.c3) < self.distance_threshold:
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

    def next_generation(self):
        self.update_species()
        self.adjust_fitness_scores()

        population: List[ContextGenome] = []

        sorted_population = sorted(
            self.population, key = lambda x: x.species.species_id)
        grouped_population = [list(it) for k, it in groupby(
            sorted_population, lambda x: x.species.species_id)]

        # Preserve champions in the species
        self.generation += 1
        for group in grouped_population:
            if len(group) > 5:
                best_genome = max(group, key = lambda x: x.fitness)
                for node in best_genome.nodes:
                    node.reset_values()
                best_genome.generation = self.generation
                population.append(best_genome)

        while len(population) < self.n_networks:
            species = self.get_random_species()
            genome_1 = self.get_random_genome(species)
            if random.random() < self.interspecies_mating_rate:
                species = self.get_random_species()
            genome_2 = self.get_random_genome(species)
            new_genome = genome_1.crossover(genome_2)
            if random.random() < self.weight_mutation_rate:
                new_genome.weight_mutation()
            if random.random() < self.node_mutation_rate:
                new_genome.add_node_mutation(activation = self.hidden_activation)
            if random.random() < self.connection_mutation_rate:
                new_genome.add_connection_mutation()
            new_genome.generation = self.generation
            population.append(new_genome)

        self.population = population

    def adjust_fitness_scores(self) -> None:
        for genome in self.population:
            genome.fitness = genome.fitness / len(genome.species.genomes)
            genome.species.total_adjusted_fitness += genome.fitness


    def get_random_species(self) -> Species:
        if len(self.species) == 1:
            return self.species[0]
        total = 0.0
        for species in self.species:
            if species.stegnant < self.stegnant_threshold:
                total += species.total_adjusted_fitness
        r = random.uniform(0, total)
        upto = 0.0
        for species in self.species:
            if species.stegnant < self.stegnant_threshold:
                if upto + species.total_adjusted_fitness >= r:
                    return species
                upto += species.total_adjusted_fitness
        assert False

    @staticmethod
    def get_random_genome(species: Species) -> ContextGenome:
        total = sum(genome.fitness for genome in species.genomes)
        r = random.uniform(0, total)
        upto = 0.0
        for genome in species.genomes:
            if upto + genome.fitness >= r:
                return genome
            upto += genome.fitness
        assert False


if __name__ == '__main__':
    # TESTING
    import matplotlib.pyplot as plt
    import snake
    import numpy as np
    import cv2

    # Settings
    SEED = 123
    VISION_RADIUS = 2
    MAP_SIZE = 6 # Size of the map including walls
    SNAKE_SIZE = 3 # Starting size
    INPUT_SIZE = (VISION_RADIUS * 2 + 1) ** 2 - 2
    N_NETWORKS = 200
    BEST_PERCENT = 0.1
    PLAYBACK = True

    np.random.seed(SEED)
    random.seed(SEED)

    assert SNAKE_SIZE >= MAP_SIZE-3

    OUTPUT_SIZE = 3 # If you change this you have to change L139
    ACTIONS = [-1, 0, 1] # Left, Straight, Right

    snake_ai = NEAT(n_networks = N_NETWORKS,
                    input_size = INPUT_SIZE,
                    output_size = OUTPUT_SIZE,
                    bias = True,
                    c1 = 1.0, c2 = 1.0, c3 = 0.4,
                    distance_threshold = 3.0,
                    weight_mutation_rate = 0.8,
                    node_mutation_rate = 0.03,
                    connection_mutation_rate = 0.05,
                    interspecies_mating_rate = 0.001,
                    disable_rate = 0.75,
                    stegnant_threshold = 15,
                    input_activation = steepened_sigmoid,
                    hidden_activation = steepened_sigmoid,
                    output_activation = steepened_sigmoid)

    while True:
        game = snake.Game(MAP_SIZE, SNAKE_SIZE)
        best_fitness = -float('Inf')
        for genome in snake_ai.population:
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
                if new_game.won:
                    screen = new_game.draw(50)
                    playback.append(screen)
                best_genome = genome
                best_game = playback
                best_fitness = score
        plt.cla()
        best_genome.draw(node_radius=2.0, vertical_distance = 5.0,
                         horizontal_distance = 50.0)
        plt.draw()
        plt.pause(0.001)

        if PLAYBACK:
            for screen in best_game:
                cv2.imshow('cvwindow', screen)
                key = cv2.waitKey(200)
                if key == 27:
                    break

        snake_ai.next_generation()
