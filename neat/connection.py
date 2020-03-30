import copy
from typing import List, Union, Tuple, Dict
import statistics
from itertools import chain, repeat, islice 

from .node import Node, NodeType


def pad_list(iterable, size, padding=None):
   return list(islice(chain(iterable, repeat(padding)), size))


class Connection(object):
    connections: Dict['Connection', int] = {}
    global_innovation: int = 0 

    def __init__(self, in_node: Node, out_node: Node, weight: float = 1.0,
                 dummy: bool = False):
        self.in_node = in_node
        self.out_node = out_node
        self.enabled = True
        self.dummy = dummy
        if dummy: return
        self.weight = weight
        self.innovation = Connection.register_connection(self)

        self.out_node.inputs.append(self)

    @classmethod
    def register_connection(cls, new_connection:'Connection') -> int:
        if new_connection in cls.connections:
            return cls.connections[new_connection]
        else:
            cls.global_innovation += 1
            cls.connections[new_connection] = cls.global_innovation
            return cls.global_innovation

    def __hash__(self):
        return hash(str(self.in_node.id)+str(self.out_node.id))

    def __eq__(self, other):
        if isinstance(other, Connection):
            return ((self.in_node == other.in_node) and
                    (self.out_node == other.out_node))
        else:
            raise ValueError(f'Value type should be Connection, got {type(other)}')

    def copy(self):
        return copy.deepcopy(self)

    def __str__(self):
        string = f'{self.in_node.id} -> {self.out_node.id} '
        string = f'{string}Weight: {self.weight:.3f} '
        string = f'{string}Innovation No: {self.innovation} '
        string = f'{string}Disabled: {not self.enabled} '
        return string

    def __repr__(self):
        return str(self.innovation)

def allign_connections(
        connections_1: List[Connection],
        connections_2: List[Connection]) -> Tuple[
            List[Connection],
            List[Connection],
            int, int, float]:
    '''Destructively allign connections by their innovation number.'''
    dummy_node = Node(0, NodeType.HIDDEN)
    dummy_connection = Connection(dummy_node, dummy_node, dummy=True)
    connections_1 = sorted(connections_1, key = lambda x: x.innovation)
    connections_2 = sorted(connections_2, key = lambda x: x.innovation)
    weights = []
    disjoint = 0
    for i in range(min(len(connections_1), len(connections_2))):
        if connections_1[i].innovation > connections_2[i].innovation:
            connections_1.insert(i, dummy_connection)
            disjoint +=1
        elif connections_1[i].innovation < connections_2[i].innovation:
            connections_2.insert(i, dummy_connection)
            disjoint +=1
        else:
            weights.append(abs(connections_1[i].weight - connections_2[i].weight))

    avarage_weight_difference = statistics.mean(weights)
    max_length = max(len(connections_1), len(connections_2))
    excess = max_length - min(len(connections_1), len(connections_2))
    connections_1 = pad_list(connections_1, max_length, padding=dummy_connection)
    connections_2 = pad_list(connections_2, max_length, padding=dummy_connection)
    return connections_1, connections_2, disjoint, excess, avarage_weight_difference
