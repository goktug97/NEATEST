#!/usr/bin/env python3
import random
import copy
from typing import Union, List
import functools
import math
import statistics

import genome
from genome import Genome

def create_population(size: int, input_size: int, output_size: int) -> List[Genome]:
    population: List[Genome] = []
    for _ in range(size):
        population.append(genome.random_genome(input_size, output_size))
    return population

class NEAT(object):
    def __init__(self, size: int, input_size: int, output_size: int):
        self.population = create_population(size, input_size, output_size)
        self.generation = 0

    def forward(self, inputs: List[List[float]]) -> List[List[float]]:
        outputs: List[List[float]] = []
        for input, genome in zip(inputs, self.population):
            outputs.append(genome(input))
        return outputs

if __name__ == '__main__':
    # TESTING
    import matplotlib.pyplot as plt
    # population = create_population(100, 3, 2)
    # print(population)
    new_genome_1 = genome.random_genome(4, 2, True)
    new_genome_1.add_node_mutation()
    new_genome_1.add_node_mutation()
    new_genome_1.add_node_mutation()
    new_genome_1.add_connection_mutation()
    new_genome_1.add_node_mutation()
    new_genome_1.add_connection_mutation()
    new_genome_1.add_node_mutation()
    new_genome_1.add_connection_mutation()
    new_genome_1.add_node_mutation()
    new_genome_1.add_connection_mutation()
    new_genome_1.weight_mutation()
    print(new_genome_1)

    new_genome_2 = genome.random_genome(4, 2, True)
    new_genome_2.add_node_mutation()
    new_genome_2.add_connection_mutation()
    new_genome_2.add_node_mutation()
    new_genome_2.add_connection_mutation()
    new_genome_2.add_node_mutation()
    new_genome_2.add_connection_mutation()
    new_genome_2.add_node_mutation()
    new_genome_2.add_connection_mutation()
    new_genome_2.add_connection_mutation()
    new_genome_2.add_node_mutation()
    new_genome_2.add_connection_mutation()
    new_genome_2.add_node_mutation()
    new_genome_2.add_node_mutation()
    new_genome_2.add_connection_mutation()
    new_genome_2.weight_mutation()
    print(new_genome_2)

    new_genome_3 = new_genome_1.crossover(new_genome_2)
    new_genome_3.add_node_mutation()
    new_genome_3.add_node_mutation()
    new_genome_3.add_node_mutation()
    new_genome_3.weight_mutation()
    new_genome_3([1.1, 1.2, 2.3, 3.1])
    new_genome_3([1.1, 1.2, 2.3, 3.1])
    new_genome_3([1.1, 1.2, 2.3, 3.1])
    new_genome_3([1.1, 1.2, 2.3, 3.1])
    print(new_genome_3)
    new_genome_3.draw()
    plt.show()
