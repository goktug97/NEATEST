#!/usr/bin/env python3
import random
from enum import Enum


class NEAT(object):
    connections = []
    global_innovation = 0 

    @classmethod
    def add_connection(cls, new_connection):
        for connection in cls.connections:
            if new_connection == connection:
                return connection.innovation
        cls.connections.append(new_connection)
        cls.global_innovation += 1
        return cls.global_innovation


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

    def __add__(self, other):
        if isinstance(other, Node):
            return self.id + other.id
        else:
            return self.id + other

    def __str__(self):
        return str(self.id)

class Genome(object):
    def __init__(self, nodes, connections):
        self.connections = connections
        self.nodes = nodes

    def add_connection(self):
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

    def add_node(self):
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


if __name__ == '__main__':
    nodes = [Node(0, NodeType.INPUT), Node(1, NodeType.OUTPUT)]
    connections = [Connection(nodes[0], nodes[1], random.random())]
    new_genome = Genome(nodes, connections)
    new_genome.add_node()
    new_genome.add_connection()
    new_genome.add_node()
    new_genome.add_connection()
    print(new_genome)
    
