#!/usr/bin/env python3
import random
import copy
from typing import Union, List, Callable, Tuple
from itertools import groupby
import functools
import math
import statistics
import pickle

from .genome import Genome
from .node import Node, NodeType
from .connection import Connection


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def steepened_sigmoid(x):
    return 1 / (1 + math.exp(-4.9 * x))

def relu(x):
    return max(x, 0)

def leaky_relu(x):
    return max(0.1*x, x)

def tanh(x):
    return math.tanh(x)


class ContextGenome(Genome):
    '''Genome class that holds data which depends on the context.'''
    def __init__(self, nodes: List[Node], connections: List[Connection]):
        self.fitness: float = 0.0
        self.generation: int = 0
        super().__init__(nodes, connections)

    def crossover(self, other: Genome) -> Genome:
        new_genome = super(ContextGenome, self).crossover(other)
        return ContextGenome(new_genome.nodes, new_genome.connections)


class NEATEST(object):
    def __init__(self, n_networks: int, input_size: int, output_size: int,
                 bias: bool,
                 node_mutation_rate: float,
                 connection_mutation_rate: float,
                 disable_connection_mutation_rate: float,
                 dominant_gene_rate: float,
                 stegnant_threshold: int = 15,
                 elite_rate: float = 0.10,
                 sigma: float = 0.1,
                 hidden_activation: Callable[[float], float]=lambda x: x,
                 output_activation: Callable[[float], float]=lambda x: x):

        self.n_networks = n_networks
        self.input_size = input_size
        self.output_size = output_size
        self.bias = bias
        self.sigma = sigma
        self.elite_rate = elite_rate
        self.disable_connection_mutation_rate = disable_connection_mutation_rate
        self.node_mutation_rate = node_mutation_rate
        self.connection_mutation_rate = connection_mutation_rate
        self.dominant_gene_rate = dominant_gene_rate
        self.stegnant_threshold = stegnant_threshold
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        self.generation = 0
        self.best_fitness: float = -float('inf')
        self.best_genome: ContextGenome

        self.create_population()

    def random_genome(self) -> ContextGenome:
        '''Create fc neural network without hidden layers with random weights.'''
        connections: List[Connection] = []
        input_nodes = [Node(i, NodeType.INPUT, depth = 0.0)
                       for i in range(self.input_size)]
        if self.bias:
            input_nodes.append(Node(self.input_size, NodeType.BIAS, value = 1.0,
                                    depth = 0.0))
        output_nodes = [Node(i+len(input_nodes), NodeType.OUTPUT,
                             self.output_activation, depth = 1.0)
                        for i in range(self.output_size)]
        for i in range(len(input_nodes)):
            input_node = input_nodes[i]
            for j in range(self.output_size):
                output_node = output_nodes[j]
                connections += [Connection(in_node=input_node, out_node=output_node,
                                           dominant_gene_rate=self.dominant_gene_rate,
                                           weight=random.gauss(0.0, self.sigma))]
        return ContextGenome(input_nodes+output_nodes, connections)


    def create_population(self) -> None:
        population: List[ContextGenome] = []
        for _ in range(self.n_networks):
            population.append(self.random_genome())
        self.population = population

    def next_generation(self, rewards : List[float]):
        for idx, reward in enumerate(rewards):
            self.population[idx].fitness = reward
            if reward > self.best_fitness:
                self.best_fitness = reward
                self.best_genome = self.population[idx]


        sorted_population = sorted(self.population,
                                   key = lambda x: x.fitness,
                                   reverse=True)
        population: List[ContextGenome]
        if self.elite_rate > 0.0:
            population = sorted_population[0:int(self.n_networks * self.elite_rate)]
        else:
            population = []

        self.generation += 1
        while len(population) < self.n_networks:
            genome_1 = self.get_random_genome(sorted_population)
            genome_2 = self.get_random_genome(sorted_population)
            new_genome = genome_1.crossover(genome_2)
            if random.random() < self.disable_connection_mutation_rate:
                new_genome.disable_connection_mutation()
            if random.random() < self.node_mutation_rate:
                new_genome.add_node_mutation(
                    dominant_gene_rate=self.dominant_gene_rate,
                    activation = self.hidden_activation)
            if random.random() < self.connection_mutation_rate:
                new_genome.add_connection_mutation(
                    sigma=self.sigma,
                    dominant_gene_rate=self.dominant_gene_rate)
            new_genome.generation = self.generation
            population.append(new_genome)

        self.population = population

    def get_random_genome(self, sorted_population: List[ContextGenome]) -> ContextGenome:
        """Return random genome from the population."""
        min_fitness = sorted_population[-1].fitness
        total: float = 0.0
        for genome in sorted_population:
            total += genome.fitness
        total += self.n_networks * (-min_fitness + 0.1)
        r = random.random()
        upto = 0.0
        for genome in sorted_population:
            score = (genome.fitness - min_fitness + 0.1) / total
            upto += score
            if upto >= r:
                return genome
        assert False

    def save_checkpoint(self):
        import time
        import pathlib
        import os
        import inspect
        frame = inspect.stack()[1]
        module = inspect.getmodule(frame[0])
        folder = pathlib.Path(
            f"{os.path.join(os.path.dirname(module.__file__), 'checkpoints')}")
        folder.mkdir(parents=True, exist_ok=True)
        filename = f'{int(time.time())}.checkpoint'
        save_path = os.path.abspath(os.path.join(folder, filename))
        print(f'Saving checkpoint: {save_path}')
        with open(os.path.join(folder, filename), 'wb') as output:
            pickle.dump(self.__dict__, output, -1)

    @classmethod
    def load_checkpoint(cls, filename):
        with open(filename, 'rb') as checkpoint:
            cls_dict = pickle.load(checkpoint)
        neat = cls.__new__(cls)
        neat.__dict__.update(cls_dict)
        return neat
