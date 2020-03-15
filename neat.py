#!/usr/bin/env python3
import random
from enum import Enum
import copy
from typing import Union, List, Set
from itertools import chain, repeat, islice

def pad_list(iterable, size, padding=None):
   return list(islice(chain(iterable, repeat(padding)), size))

class Connection(object):
    def __init__(self, in_node, out_node, weight):
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.enabled = True
        self.innovation = NEAT.add_connection(self)

    def __eq__(self, other):
        if ((self.in_node == other.in_node) and
            (self.out_node == other.out_node)):
            return True
        else:
            return False

    def __str__(self):
        string = f'{self.in_node} -> {self.out_node} '
        string = f'{string}Weight: {self.weight:.3f} '
        string = f'{string}Innovation No: {self.innovation} '
        string = f'{string}Disabled: {not self.enabled}'
        return string

    def __repr__(self):
        return str(self.innovation)


class NodeType(Enum):
    INPUT = 1
    HIDDEN = 2
    OUTPUT = 3


class Node(object):
    def __init__(self, id, type):
        self.id = id
        self.type = type

    def __eq__(self, other):
        if self.id == other.id:
            return True
        else:
            return False

    def __add__(self, other: Union["Node", int]) -> int:
        if isinstance(other, Node):
            return self.id + other.id
        else:
            return self.id + other

    def __hash__(self):
        return hash(self.id)

    def __str__(self):
        return str(self.id)


class Genome(object):
    def __init__(self, nodes, connections):
        self.connections = copy.deepcopy(connections)
        self.nodes = copy.deepcopy(nodes)

    def add_connection_mutation(self) -> None:
        while True:
            in_node = random.choice(self.nodes)
            # Do we want this?
            while True:
                out_node = random.choice(self.nodes)
                if in_node != out_node:
                    break
            new_connection = Connection(in_node, out_node, random.random())
            unique = True
            for connection in self.connections:
                if new_connection == connection:
                    unique = False
            if unique:
                self.connections.append(new_connection)
                break

    def add_node_mutation(self) -> None:
        new_node = Node(max(self.nodes, key=lambda x: x.id)+1, NodeType.HIDDEN)
        self.nodes.append(new_node)
        idx = random.randint(0, len(self.connections)-1)
        self.connections[idx].enabled = False
        first_connection = Connection(self.connections[idx].in_node, new_node, 1)
        weight = self.connections[idx].weight
        second_connection = Connection(new_node, self.connections[idx].out_node, weight)
        self.connections.append(first_connection)
        self.connections.append(second_connection)

    def __str__(self):
        string = ''
        string = f'{string}NODES:\n'
        for node in self.nodes:
            string = f'{string}{node.id}:{node.type} ' 
        string = f'{string}\n\nCONNECTIONS:\n'
        for connection in self.connections:
            string = f'{string}{connection}\n'
        return string


class NEAT(object):
    connections: List[Connection] = []
    global_innovation = 0 

    @staticmethod
    def random_genome(input_size: int, output_size: int) -> Genome:
        connections: List[Connection] = []
        nodes: Set[Node] = set()
        for i in range(input_size):
            for j in range(output_size):
                input_node = Node(i, NodeType.INPUT)
                output_node = Node(j+output_size, NodeType.OUTPUT)
                nodes.add(input_node)
                nodes.add(output_node)
                connections += [Connection(input_node, output_node, random.random())]
        return Genome(list(nodes), connections)

    @classmethod
    def add_connection(cls, new_connection:Connection) -> int:
        for connection in cls.connections:
            if new_connection == connection:
                return connection.innovation
        cls.connections.append(new_connection)
        cls.global_innovation += 1
        return cls.global_innovation

    @staticmethod
    def crossover(genome_1:Genome, genome_2:Genome) -> Genome:
        connections: List[Connection] = []
        connections_1 = copy.deepcopy(genome_1.connections)
        connections_2 = copy.deepcopy(genome_2.connections)
        for i in range(min(len(connections_1), len(connections_2))):
            if connections_1[i].innovation > connections_2[i].innovation:
                connections_1.insert(i, None)
            elif connections_1[i].innovation < connections_2[i].innovation:
                connections_2.insert(i, None)
            else:
                pass

        max_length = max(len(connections_1), len(connections_2))
        connections_1 = pad_list(connections_1, max_length)
        connections_2 = pad_list(connections_2, max_length)

        for idx in range(max_length):
            connection_1 = connections_1[idx]
            connection_2 = connections_2[idx]
            if connection_1 is None:
                connection = connection_2
            elif connection_2 is None:
                connection = connection_1
            else:
                connection = random.choice([connection_1, connection_2])
            connections.append(connection)

        nodes = list(set(genome_1.nodes + genome_2.nodes))
        return Genome(nodes, connections)
        

if __name__ == '__main__':
    new_genome_1 = NEAT.random_genome(1, 1)
    new_genome_1.add_node_mutation()
    new_genome_1.add_connection_mutation()
    new_genome_1.add_node_mutation()
    new_genome_1.add_connection_mutation()
    print(new_genome_1)

    new_genome_2 = NEAT.random_genome(1, 1)
    new_genome_2.add_node_mutation()
    new_genome_2.add_connection_mutation()
    new_genome_2.add_node_mutation()
    new_genome_2.add_connection_mutation()
    new_genome_2.add_node_mutation()
    new_genome_2.add_connection_mutation()
    print(new_genome_2)

    print(NEAT.crossover(new_genome_1, new_genome_2))

    
