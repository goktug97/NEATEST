#!/usr/bin/env python3
import random
import copy
from typing import Union, List, Callable, Tuple, NewType
from itertools import groupby
import functools
import math
import statistics
import pickle
from abc import ABC, abstractmethod
import operator
import os
import sys

import numpy as np
from mpi4py import MPI #type: ignore

from .genome import Genome
from .node import Node, NodeType
from .node import sigmoid, steepened_sigmoid
from .node import relu, leaky_relu
from .node import tanh, passthrough
from .connection import Connection
from .optimizers import Optimizer

Array = Union[np.ndarray, np.generic]

@functools.lru_cache(maxsize=1)
def _center_function(population_size: int) -> Array:
    centers = np.arange(0, population_size)
    centers = centers / (population_size - 1)
    centers -= 0.5
    return centers

def _compute_ranks(rewards: Union[List[float], Array]) -> Array:
    rewards = np.array(rewards)
    ranks = np.empty(rewards.size, dtype=int)
    ranks[rewards.argsort()] = np.arange(rewards.size)
    return ranks

def rank_transformation(rewards: Union[List[float], Array]) -> Array:
    ranks = _compute_ranks(rewards)
    values = _center_function(len(rewards))
    return values[ranks] #type: ignore

class ContextGenome(Genome):
    '''Genome class that holds data which depends on the context.'''
    def __init__(self, nodes: List[Node], connections: List[Connection]):
        self.fitness: float = 0.0
        self.generation: int = 0
        super().__init__(nodes, connections)

    def crossover(self, other: 'ContextGenome') -> 'ContextGenome': #type: ignore
        new_genome = super(ContextGenome, self).crossover(other)
        return ContextGenome(new_genome.nodes, new_genome.connections)

    def copy(self) -> 'ContextGenome': #type: ignore
        new_genome = super(ContextGenome, self).copy()
        return ContextGenome(new_genome.nodes, new_genome.connections)



SortedContextGenomes = NewType('SortedContextGenomes', List[ContextGenome])


class Agent(ABC):
    @abstractmethod
    def rollout(self, genome: Genome) -> float:
        ...


class NEATEST(object):
    def __init__(self,
                 agent: Agent,
                 optimizer: Optimizer,
                 n_networks: int,
                 es_population: int,
                 input_size: int,
                 output_size: int,
                 bias: bool,
                 node_mutation_rate: float,
                 connection_mutation_rate: float,
                 disable_connection_mutation_rate: float,
                 dominant_gene_rate: float,
                 dominant_gene_delta: float,
                 seed: int,
                 hidden_layers: List[int] = [],
                 elite_rate: float = 0.0,
                 sigma: float = 0.01,
                 hidden_activation: Callable[[float], float]=passthrough,
                 output_activation: Callable[[float], float]=passthrough):

        self.comm = MPI.COMM_WORLD
        self.n_proc = self.comm.Get_size()
        assert not n_networks % self.n_proc
        assert not es_population % self.n_proc
        random.seed(seed)
        np.random.seed(seed)
        self.agent = agent
        self.optimizer = optimizer
        self.es_population = es_population
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
        self.dominant_gene_delta = dominant_gene_delta
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        self.generation: int = 0
        self.best_fitness: float = -float('inf')
        self.best_genome: ContextGenome
        self.population: List[ContextGenome]

        if not self.comm.rank == 0:
            f = open(os.devnull, 'w')
            sys.stdout = f

        self.create_population(hidden_layers)

    def random_genome(self, hidden_layers) -> ContextGenome:
        '''Create fc neural network without hidden layers with random weights.'''
        layers = []
        connections: List[Connection] = []

        # Input Nodes
        input_nodes = [Node(-1, NodeType.INPUT, depth = 0.0)
                       for _ in range(self.input_size)]
        if self.bias:
            input_nodes.append(Node(-1, NodeType.BIAS, value = 1.0,
                                    depth = 0.0))
        layers.append(input_nodes)

        # Hidden Nodes
        for idx, hidden_layer in enumerate(hidden_layers):
            depth = 1 / (len(hidden_layers) + 1) * (idx + 1)
            hidden_nodes = [Node(-1,
                                 NodeType.HIDDEN,
                                 self.hidden_activation,
                                 depth = depth)
                           for _ in range(hidden_layer)]
            if self.bias:
                hidden_nodes.append(
                    Node(-1,
                         NodeType.BIAS, value = 1.0,
                         depth = depth))
            layers.append(hidden_nodes)

        # Output Nodes
        output_nodes = [Node(-1, NodeType.OUTPUT,
                             self.output_activation, depth = 1.0)
                        for _ in range(self.output_size)]
        layers.append(output_nodes)

        for i in range(1, len(layers)):
            for j in range(len(layers[i-1])):
                input_node = layers[i-1][j]
                for k in range(len(layers[i])):
                    output_node = layers[i][k]
                    connections += [Connection(
                        in_node=input_node, out_node=output_node,
                        dominant_gene_rate=self.dominant_gene_rate,
                        weight=random.gauss(0.0, self.sigma))]

        nodes: List[Node] = functools.reduce(operator.iconcat, layers, [])
        return ContextGenome(nodes, connections)

    def create_population(self, hidden_layers) -> None:
        population: List[ContextGenome] = [self.random_genome(hidden_layers)]
        for _ in range(self.n_networks):
            population.append(population[0].copy())
        self.population = population

    def next_generation(self):

        sorted_population: SortedContextGenomes = self.sort_population(
                self.population)

        population: List[ContextGenome]
        if self.elite_rate > 0.0:
            population = sorted_population[0:int(self.n_networks * self.elite_rate)]
        else:
            population = []

        self.generation += 1
        while len(population) < self.n_networks:
            genome_1 = self.get_random_genome()
            genome_2 = self.get_random_genome()
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

    def calculate_grads(self, genome: ContextGenome):
        genome: ContextGenome = genome.copy() #type: ignore
        for i in reversed(range(len(genome.connections))):
            if not genome.connections[i].enabled:
                del genome.connections[i]
        weights: List[float] = [connection.weight for connection in genome.connections]
        weights_array: Array = np.array(weights)
        epsilon: Array = np.random.normal(0.0, self.sigma,
                                          (self.es_population//2,
                                           len(weights)))
        population_weights: Array = np.concatenate([weights_array + epsilon,
                                                    weights_array - epsilon])

        n_jobs = self.es_population // self.n_proc
        rewards: List[float] = []
        rewards_array: Array = np.zeros(self.es_population, dtype='d')
        for i in range(self.comm.rank*n_jobs, n_jobs * (self.comm.rank + 1)):
            for j, connection in enumerate(genome.connections):
                connection.weight = population_weights[i, j] #type: ignore
            genome.reset_values()
            rewards.append(self.agent.rollout(genome))
        self.comm.Allgather([np.array(rewards, dtype=np.float64), MPI.DOUBLE],
                            [rewards_array, MPI.DOUBLE])
        ranked_rewards: Array = rank_transformation(rewards_array)
        epsilon = np.concatenate([epsilon, -epsilon])
        grads: Array = (np.dot(ranked_rewards, epsilon) /
                        (self.es_population * self.sigma))
        grads = np.clip(grads, -1.0, 1.0)

        for i in range(len(Connection.dominant_gene_rates)):
            Connection.dominant_gene_rates[i] -= self.dominant_gene_delta

        for i, connection in enumerate(genome.connections):
            connection.weight = weights[i]
            connection.grad = -grads[i] #type: ignore
            connection.dominant_gene_rate += 2 * self.dominant_gene_delta

        for i in range(len(Connection.dominant_gene_rates)):
            rate = Connection.dominant_gene_rates[i]
            Connection.dominant_gene_rates[i] = min(0.6, max(0.4, rate))

    def train(self, n_steps: int) -> None:
        n_jobs = self.n_networks // self.n_proc
        for step in range(n_steps):
            rewards = []
            print(f'Generation: {self.generation}')

            for genome in self.population[
                    self.comm.rank*n_jobs: n_jobs * (self.comm.rank + 1)]:
                genome.reset_values()
                reward = self.agent.rollout(genome)
                rewards.append(reward)
            rewards = functools.reduce(
                operator.iconcat, self.comm.allgather(rewards), [])

            for idx, reward in enumerate(rewards):
                self.population[idx].fitness = reward
                if reward > self.best_fitness:
                    self.best_fitness = reward
                    self.best_genome = self.population[idx].copy().detach()

            self.train_genome(self.get_random_genome())

            self.next_generation()
            print(f'Max Reward Session: {self.best_fitness}')
            print(f'Max Reward Step: {max(rewards)}')

    def train_genome(self, genome: ContextGenome, n_steps: int = 1):
        for _ in range(n_steps):
            self.optimizer.zero_grad()
            self.calculate_grads(genome)
            self.optimizer.step()

    @staticmethod
    def sort_population(population: List[ContextGenome]) -> SortedContextGenomes:
        return SortedContextGenomes(sorted(population, key = lambda x: x.fitness,
                                           reverse=True))

    def get_random_genome(self) -> ContextGenome:
        """Return random genome from a sorted population."""
        rewards: Array = np.array([genome.fitness for genome in self.population])
        eps = np.finfo(float).eps
        normalized_rewards: Array = rewards - rewards.min() + eps
        probabilities = normalized_rewards / np.sum(normalized_rewards)
        return np.random.choice(self.population, p=probabilities)

    def save_checkpoint(self) -> None:
        import time
        import pathlib
        import os
        import inspect
        frame = inspect.stack()[1]
        module = inspect.getmodule(frame[0])
        if module is not None:
            folder = pathlib.Path(
                f"{os.path.join(os.path.dirname(module.__file__), 'checkpoints')}")
            folder.mkdir(parents=True, exist_ok=True)
            filename = f'{int(time.time())}.checkpoint'
            save_path = os.path.abspath(os.path.join(folder, filename))
            print(f'Saving checkpoint: {save_path}')
            with open(os.path.join(folder, filename), 'wb') as output:
                pickle.dump(self.__dict__, output, -1)

    @classmethod
    def load_checkpoint(cls, filename: str) -> 'NEATEST':
        with open(filename, 'rb') as checkpoint:
            cls_dict = pickle.load(checkpoint)
        neat = cls.__new__(cls)
        neat.__dict__.update(cls_dict)
        return neat
