#!/usr/bin/env python3
import random
import copy
from typing import Union, List, Callable, Tuple
from itertools import groupby
import functools
import math
import statistics

from genome import Genome
from node import Node, NodeType
from connection import Connection


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def steepened_sigmoid(x):
    return 1 / (1 + math.exp(-4.9 * x))

def relu(x):
    return max(x, 0)

def leaky_relu(x):
    return max(0.1*x, x)

def tanh(x):
    e_2x = math.exp(2*x)
    return (e_2x - 1) / (e_2x + 1)


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
                 random_range: Tuple[float, float] = (-1.0, 1.0),
                 noise_magnitude: float = 0.001,
                 input_activation: Callable[[float], float]=steepened_sigmoid,
                 hidden_activation: Callable[[float], float]=steepened_sigmoid,
                 output_activation: Callable[[float], float]=steepened_sigmoid):

        self.n_networks = n_networks
        self.input_size = input_size
        self.output_size = output_size
        self.bias = bias
        self.c1, self.c2, self.c3 = c1, c2, c3
        self.random_range = random_range
        self.noise_magnitude = noise_magnitude
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
        input_nodes = [Node(i, NodeType.INPUT, self.input_activation,
                            depth = 0.0)
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
                connections += [Connection(input_node, output_node,
                                           random.uniform(*self.random_range))]
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

    def next_generation(self):
        self.update_species()

        # Remove the worst from each species and delete empty species
        for idx in reversed(range(len(self.species))):
            species = self.species[idx]
            if len(species.genomes):
                # Species are sorted by fitness scores in reverse
                species.genomes.sort(key = lambda x: x.fitness, reverse=True)
                species.genomes.pop()
            if not len(species.genomes):
                self.species.pop(idx)
                
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
                new_genome.weight_mutation(self.noise_magnitude, self.random_range)
            if random.random() < self.node_mutation_rate:
                new_genome.add_node_mutation(activation = self.hidden_activation)
            if random.random() < self.connection_mutation_rate:
                new_genome.add_connection_mutation(self.random_range)
            new_genome.generation = self.generation
            population.append(new_genome)

        self.population = population

    def adjust_fitness_scores(self) -> None:
        for species in self.species:
            for genome in species.genomes:
                genome.fitness = genome.fitness / len(species.genomes)
                species.total_adjusted_fitness += genome.fitness

    def get_random_species(self) -> Species:
        if len(self.species) == 1:
            return self.species[0]
        total = 0.0
        length = 0
        min_fitness = float('inf')
        for species in self.species:
            if species.stegnant < self.stegnant_threshold:
                length += 1
                total += species.total_adjusted_fitness
                if species.total_adjusted_fitness < min_fitness:
                    min_fitness = species.total_adjusted_fitness
        total += length * (-min_fitness + 0.1)
        r = random.random()
        upto = 0.0
        for species in self.species:
            if species.stegnant < self.stegnant_threshold:
                score = (species.total_adjusted_fitness - min_fitness + 0.1) / total
                upto += score
                if upto >= r:
                    return species
        assert False

    @staticmethod
    def get_random_genome(species: Species) -> ContextGenome:
        if len(species.genomes) == 1:
            return species.genomes[0]
        # Assume genomes are sorted by fitness, max first
        min_fitness = species.genomes[-1].fitness
        total = species.total_adjusted_fitness
        total += len(species.genomes) * (-min_fitness + 0.1)
        r = random.random()
        upto = 0.0
        for genome in species.genomes:
            score = (genome.fitness - min_fitness + 0.1) / total
            upto += score
            if upto >= r:
                return genome
        assert False
