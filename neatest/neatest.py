#!/usr/bin/env python3
import random
import copy
from typing import Union, List, Callable, Tuple, NewType, Dict, Type
import itertools
import functools
import math
import statistics
from abc import ABC, abstractmethod
import operator
import os
import sys
import time
import pathlib
import inspect

try:
    disable_mpi = os.environ.get('NEATEST_DISABLE_MPI')
    if disable_mpi and disable_mpi != '0':
        raise ImportError
    from mpi4py import MPI
except ImportError:
    from .MPI import MPI
    MPI = MPI()
import numpy as np
import cloudpickle #type: ignore

from .genome import Genome
from .node import Node, NodeType
from .node import sigmoid, steepened_sigmoid
from .node import relu, leaky_relu
from .node import tanh, passthrough
from .connection import Connection, Weight, GeneRate, DummyConnection, align_connections
from .optimizers import Optimizer
from .version import VERSION

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
        self.generation: int = 1
        super().__init__(nodes, connections)

    def copy(self) -> 'ContextGenome': #type: ignore
        new_genome = super(ContextGenome, self).copy()
        return ContextGenome(new_genome.nodes, new_genome.connections)

    def deepcopy(self) -> 'ContextGenome': #type: ignore
        new_genome = super(ContextGenome, self).deepcopy()
        return ContextGenome(new_genome.nodes, new_genome.connections)


SortedContextGenomes = NewType('SortedContextGenomes', List[ContextGenome])


class Agent(ABC):
    @abstractmethod
    def rollout(self, genome: Genome) -> float:
        ...


class NEATEST(object):
    def __init__(self,
                 agent: Agent,
                 optimizer: Type[Optimizer],
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
                 save_checkpoint_n: int = 50,
                 logdir: str = None,
                 optimizer_kwargs: dict = {},
                 hidden_layers: List[int] = [],
                 elite_rate: float = 0.0,
                 sigma: float = 0.01,
                 hidden_activation: Callable[[float], float]=passthrough,
                 output_activation: Callable[[float], float]=passthrough):

        comm = MPI.COMM_WORLD
        self.version = VERSION
        n_proc = comm.Get_size()
        self.n_proc = n_proc
        # assert not n_networks % n_proc
        # assert not es_population % n_proc
        self.seed = seed;
        self.random = random.Random(seed)
        self.np_random = np.random.RandomState(seed)
        self.logdir = logdir
        self.agent = agent
        self.es_population = es_population
        self.n_networks = n_networks
        self.input_size = input_size
        self.output_size = output_size
        self.bias = bias
        self.sigma = sigma
        self.save_checkpoint_n = save_checkpoint_n
        self.elite_rate = elite_rate
        self.disable_connection_mutation_rate = disable_connection_mutation_rate
        self.node_mutation_rate = node_mutation_rate
        self.connection_mutation_rate = connection_mutation_rate
        self.dominant_gene_rate = dominant_gene_rate
        self.dominant_gene_delta = dominant_gene_delta
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        self.weights: List[Weight] = []
        self.gene_rates: List[GeneRate] = []
        self.connections: Dict[Connection, int] = {}
        self.node_id_generator = itertools.count(0, 1)
        self.connection_id_generator = itertools.count(0, 1)

        self.optimizer: Optimizer = optimizer(self.weights, **optimizer_kwargs)

        self.generation: int = 1
        self.best_fitness: float = -float('inf')
        self.best_genome: ContextGenome
        self.population: List[ContextGenome]

        self.data: List[Tuple[int, str, int, float]] = []

        if not comm.rank == 0:
            f = open(os.devnull, 'w')
            sys.stdout = f
        else:
            if self.logdir:
                from pathlib import Path
                Path(self.logdir).mkdir(parents=True, exist_ok=True)

        self.create_population(hidden_layers)

    def random_genome(self, hidden_layers) -> ContextGenome:
        '''Create fc neural network with random weights.'''
        layers = []
        connections: List[Connection] = []

        # Input Nodes
        input_nodes = [Node(next(self.node_id_generator), NodeType.INPUT, depth = 0.0)
                       for _ in range(self.input_size)]
        if self.bias:
            input_nodes.append(Node(next(self.node_id_generator), NodeType.BIAS,
                                    value = 1.0,
                                    depth = 0.0))
        layers.append(input_nodes)

        # Hidden Nodes
        for idx, hidden_layer in enumerate(hidden_layers):
            depth = 1 / (len(hidden_layers) + 1) * (idx + 1)
            hidden_nodes = [Node(next(self.node_id_generator),
                                 NodeType.HIDDEN,
                                 self.hidden_activation,
                                 depth = depth)
                           for _ in range(hidden_layer)]
            if self.bias:
                hidden_nodes.append(
                    Node(next(self.node_id_generator),
                         NodeType.BIAS, value = 1.0,
                         depth = depth))
            layers.append(hidden_nodes)

        # Output Nodes
        output_nodes = [Node(next(self.node_id_generator), NodeType.OUTPUT,
                             self.output_activation, depth = 1.0)
                        for _ in range(self.output_size)]
        layers.append(output_nodes)

        for i in range(1, len(layers)):
            for j in range(len(layers[i-1])):
                input_node = layers[i-1][j]
                for k in range(len(layers[i])):
                    output_node = layers[i][k]
                    dummy_connection = DummyConnection(input_node, output_node)
                    innovation, weight, dominant_gene_rate = self.register_connection(
                        dummy_connection)

                    connections += [Connection(
                        in_node=input_node, out_node=output_node,
                        innovation = innovation,
                        dominant_gene_rate=dominant_gene_rate,
                        weight=weight)]

        nodes: List[Node] = functools.reduce(operator.iconcat, layers, [])
        return ContextGenome(nodes, connections)

    def create_population(self, hidden_layers) -> None:
        population: List[ContextGenome] = [self.random_genome(hidden_layers)]
        for _ in range(self.n_networks):
            population.append(population[0].copy())
        self.population = population

    def register_connection(self, dummy_connection: DummyConnection):
        if dummy_connection in self.connections:
            innovation = self.connections[dummy_connection]
            weight = self.weights[innovation]
            dominant_gene_rate = self.gene_rates[innovation]
        else:
            innovation = next(self.connection_id_generator)
            weight = Weight(self.random.gauss(0.0, self.sigma))
            dominant_gene_rate = GeneRate(self.dominant_gene_rate)
            self.weights.append(weight)
            self.gene_rates.append(dominant_gene_rate)
            self.connections[dummy_connection] = innovation
        return innovation, weight, dominant_gene_rate

    def add_connection_mutation(self, genome: Genome) -> None:
        '''Create new connection between two random non-connected nodes.'''
        def _add_connection_mutation(depth = 0):
            if depth > 20:
                return
            in_idx = self.random.randint(0, len(genome.nodes) - 1)
            in_node = genome.nodes[in_idx]
            out_idx = self.random.randint(0, len(genome.nodes) - 1)
            out_node = genome.nodes[out_idx]
            dummy_connection = DummyConnection(in_node, out_node)
            try:
                index = genome.connections.index(dummy_connection)
                if not genome.connections[index].enabled:
                    if (self.random.random() <=
                        genome.connections[index].dominant_gene_rate.value):
                        genome.connections[index].enabled = True
                        return
                else:
                    _add_connection_mutation(depth+1)
                    return
            except ValueError:
                pass

            if (out_node.type == NodeType.BIAS or
                out_node.type == NodeType.INPUT or
                in_node.type == NodeType.OUTPUT):
                _add_connection_mutation(depth+1)
                return
            innovation, weight, dominant_gene_rate = self.register_connection(
                dummy_connection)

            connection = Connection(in_node=in_node, out_node=out_node,
                                    innovation = innovation,
                                    dominant_gene_rate=dominant_gene_rate,
                                    weight=weight)
            genome.connections.append(connection)
        _add_connection_mutation()

    def add_node_mutation(self, genome: Genome,
                          activation: Callable[[float], float]=lambda x: x) -> None:
        '''Add a node to a random connection and split the connection.'''
        idx = self.random.randint(0, len(genome.connections)-1)
        genome.connections[idx].enabled = False

        # new_node = Node(next(self.node_id_generator), NodeType.HIDDEN, activation)
        new_node = Node(max(genome.nodes) + 1, NodeType.HIDDEN, activation)

        first_weight = Weight(1.0)
        first_gene_rate = GeneRate(self.dominant_gene_rate)
        first_innovation = next(self.connection_id_generator)
        first_connection = Connection(in_node=genome.connections[idx].in_node,
                                      out_node=new_node,
                                      innovation = first_innovation,
                                      dominant_gene_rate=first_gene_rate,
                                      weight=first_weight)
        self.weights.append(first_weight)
        self.gene_rates.append(first_gene_rate)
        self.connections[first_connection] = first_innovation

        second_weight = Weight(genome.connections[idx].weight.value)
        second_gene_rate = GeneRate(self.dominant_gene_rate)
        second_innovation = next(self.connection_id_generator)
        second_connection = Connection(in_node=new_node,
                                       out_node=genome.connections[idx].out_node,
                                       innovation=second_innovation,
                                       dominant_gene_rate=second_gene_rate,
                                       weight=second_weight)
        self.weights.append(second_weight)
        self.gene_rates.append(second_gene_rate)
        self.connections[second_connection] = second_innovation

        genome.connections.append(first_connection)
        genome.connections.append(second_connection)
        new_node.depth = (first_connection.in_node.depth +
                          second_connection.out_node.depth) / 2
        genome.nodes.append(new_node)

    def disable_connection_mutation(self, genome: Genome) -> None:
        def _disable_connection_mutation(depth = 0):
            if depth > 20:
                return
            idx = self.random.randint(0, len(genome.connections)-1)
            if (genome.connections[idx].out_node.type == NodeType.OUTPUT or
                genome.connections[idx].in_node.type == NodeType.INPUT or
                genome.connections[idx].in_node.type == NodeType.BIAS):
                _disable_connection_mutation(depth + 1)
                return
            else:
                if not genome.connections[idx].enabled:
                    _disable_connection_mutation(depth + 1)
                    return
                else:
                    genome.connections[idx].enabled = False
                    return
        _disable_connection_mutation()

    def next_generation(self, sorted_population: SortedContextGenomes):
        population: List[ContextGenome]
        if self.elite_rate > 0.0:
            population = sorted_population[0:int(self.n_networks * self.elite_rate)]
        else:
            population = []

        self.generation += 1
        while len(population) < self.n_networks:
            genome_1 = self.get_random_genome()
            genome_2 = self.get_random_genome()
            new_genome = self.crossover(genome_1, genome_2)
            if self.random.random() < self.disable_connection_mutation_rate:
                self.disable_connection_mutation(new_genome)
            if self.random.random() < self.node_mutation_rate:
                self.add_node_mutation(
                    new_genome,
                    activation = self.hidden_activation)
            if self.random.random() < self.connection_mutation_rate:
                self.add_connection_mutation(
                    new_genome)
            new_genome.generation = self.generation
            population.append(new_genome)

        self.population = population

    def calculate_grads(self, genome: ContextGenome):
        comm = MPI.COMM_WORLD
        cp_genome: ContextGenome = genome.deepcopy() #type: ignore
        for i in reversed(range(len(cp_genome.connections))):
            if not cp_genome.connections[i].enabled:
                del cp_genome.connections[i]
        weights: List[float] = [connection.weight.value
                                for connection in cp_genome.connections]
        weights_array: Array = np.array(weights)
        epsilon: Array = self.np_random.normal(0.0, self.sigma,
                                          (self.es_population//2,
                                           len(weights)))
        population_weights: Array = np.concatenate([weights_array + epsilon,
                                                    weights_array - epsilon])

        n_jobs = self.es_population // comm.Get_size()
        rewards: List[float] = []
        rewards_array: Array = np.zeros(self.es_population, dtype='d')
        for i in range(comm.rank*n_jobs, n_jobs * (comm.rank + 1)):
            for j, connection in enumerate(cp_genome.connections):
                connection.weight = Weight(population_weights[i, j]) #type: ignore
            rewards.append(self.agent.rollout(cp_genome))
        comm.Allgatherv([np.array(rewards, dtype=np.float64), MPI.DOUBLE],
                            [rewards_array, MPI.DOUBLE])
        ranked_rewards: Array = rank_transformation(rewards_array)
        epsilon = np.concatenate([epsilon, -epsilon])
        grads: Array = (np.dot(ranked_rewards, epsilon) /
                        (self.es_population * self.sigma))
        grads = np.clip(grads, -1.0, 1.0)

        for gene_rate in self.gene_rates:
            gene_rate.value -= self.dominant_gene_delta

        idx = 0
        for connection in genome.connections:
            if connection.enabled:
                connection.weight.grad = -grads[idx] #type: ignore
                connection.dominant_gene_rate.value += 2 * self.dominant_gene_delta
                idx += 1

        for gene_rate in self.gene_rates:
            gene_rate.value = min(0.6, max(0.4, gene_rate.value))

    def train(self, n_steps: int) -> None:
        comm = MPI.COMM_WORLD
        n_jobs = self.n_networks // comm.Get_size()
        for step in range(n_steps):
            rewards = []
            for genome in self.population[
                    comm.rank*n_jobs: n_jobs * (comm.rank + 1)]:
                reward = self.agent.rollout(genome)
                rewards.append(reward)
            rewards = functools.reduce(
                operator.iconcat, comm.allgatherv(rewards), [])

            for idx, reward in enumerate(rewards):
                self.population[idx].fitness = reward

            self.train_genome(self.get_random_genome())

            sorted_population: SortedContextGenomes = self.sort_population(
                self.population)

            reward = self.agent.rollout(sorted_population[0])
            self.data.append((int(self.generation), 'NEATEST', self.seed, reward))
            if reward >= self.best_fitness:
                self.best_fitness = reward
                self.best_genome = sorted_population[0].deepcopy()

            print(f'Generation: {self.generation}')
            print(f'Rollout Reward: {reward}')
            print(f'Max Reward Session: {self.best_fitness}')
            print(f'Max Reward in Population: {max(rewards)}')

            if self.generation and not self.generation % self.save_checkpoint_n:
                self.save_checkpoint()

            self.next_generation(sorted_population)

        if self.logdir:
            if MPI.COMM_WORLD.rank == 0:
                self.save_logs()

    def train_genome(self, genome: ContextGenome, n_steps: int = 1):
        for _ in range(n_steps):
            self.optimizer.zero_grad()
            self.calculate_grads(genome)
            self.optimizer.step()

    def crossover(self, genome_1:Genome, genome_2:Genome) -> ContextGenome:
        '''Crossover two genomes by aligning their innovation numbers.'''
        connections: List[Connection] = []
        nodes: List[Node] = []
        connections_1, connections_2 = align_connections(
            genome_1.connections, genome_2.connections)

        for idx in range(len(connections_1)):
            connection_1 = connections_1[idx]
            connection_2 = connections_2[idx]
            enabled: bool
            connection: Connection
            if connection_1.dummy or connection_2.dummy:
                if connection_1.dummy:
                    connection = connection_2
                else:
                    connection = connection_1
                if connection.enabled:
                    if self.random.random() <= connection.dominant_gene_rate.value:
                        enabled = True
                    else:
                        enabled = False
                else:
                    continue
            else:
                connection = connection_1
                if connection_1.enabled and connection_2.enabled:
                    enabled = True
                elif connection_1.enabled ^ connection_2.enabled:
                    enabled = (self.random.random() <=
                               connection.dominant_gene_rate.value)
                else:
                    if self.random.random() <= connection.dominant_gene_rate.value:
                        enabled = False
                    else:
                        continue

            in_node = Node(connection.in_node.id, connection.in_node.type,
                           connection.in_node.activation,
                           depth = connection.in_node.depth)
            out_node = Node(connection.out_node.id, connection.out_node.type,
                            connection.out_node.activation,
                            depth = connection.out_node.depth)

            nodes_dict = dict(zip(nodes, range(len(nodes))))
            if in_node not in nodes_dict:
                nodes.append(in_node)
                nodes_dict[in_node] = len(nodes)-1
            if out_node not in nodes_dict:
                nodes.append(out_node)
                nodes_dict[out_node] = len(nodes)-1
            connection = Connection(nodes[nodes_dict[in_node]],
                                    nodes[nodes_dict[out_node]],
                                    innovation = connection.innovation,
                                    weight = connection.weight,
                                    dominant_gene_rate = connection.dominant_gene_rate)
            connection.enabled = enabled
            connections.append(connection)
        new_genome = ContextGenome(nodes, connections)
        return new_genome

    def save_logs(self):
        import pandas as pd #type: ignore
        data = pd.DataFrame(
            self.data, columns=['Generation', 'Algorithm', 'Seed', 'Reward'])
        data = data.astype(
            {"Generation": int, "Algorithm": str, 'Seed': int, 'Reward': float})
        file = os.path.join(self.logdir, f'{time.strftime("%Y%m%d-%H%M%S")}.csv')
        data.to_csv(file, index=False)

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
        return self.np_random.choice(self.population, p=probabilities)

    def save_checkpoint(self) -> None:
        if MPI.COMM_WORLD.rank == 0:
            frame = inspect.stack()[2]
            module = inspect.getmodule(frame[0])
            if module is not None:
                folder = pathlib.Path(
                    f"{os.path.join(os.path.dirname(module.__file__), 'checkpoints')}")
                folder.mkdir(parents=True, exist_ok=True)
                filename = f'{int(time.time())}.checkpoint'
                save_path = os.path.abspath(os.path.join(folder, filename))
                print(f"\033[33;1mCheckpoint: {save_path}\033[0m")
                with open(save_path, 'wb') as output:
                    cloudpickle.dump(self, output)

    @classmethod
    def load_checkpoint(cls, filename: str) -> 'NEATEST':
        comm = MPI.COMM_WORLD
        if not comm.rank == 0:
            f = open(os.devnull, 'w')
            sys.stdout = f
        print(f"\033[33;1mLoading: {filename}\033[0m")
        with open(filename, 'rb') as checkpoint:
            neatest = cloudpickle.load(checkpoint)
        if comm.rank == 0:
            if neatest.logdir:
                from pathlib import Path
                Path(neatest.logdir).mkdir(parents=True, exist_ok=True)
        n_proc = comm.Get_size()
        # assert not neatest.n_networks % n_proc
        # assert not neatest.es_population % n_proc
        if neatest.n_proc != n_proc:
            print("\033[31;1mWarning: Number of process mismatch\033[0m")
        if neatest.version != VERSION:
            print("\033[31;1mWarning: Checkpoint version mismatch!\n"
                  f"Current Version: {VERSION.major}.{VERSION.minor}.{VERSION.patch}\n"
                  "Checkpoint Version:"
                  f" {neatest.version.major}.{neatest.version.minor}."
                  f"{neatest.version.patch}\n\033[0m")
        return neatest
