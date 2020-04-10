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

    def __gt__(self, other):
        return self.innovation > other.innovation

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
        if self.dummy:
            string = f'{string}Innovation No: Dummy '
        else:
            string = f'{string}Innovation No: {self.innovation} '
        string = f'{string}Disabled: {not self.enabled} '
        return string

    def __repr__(self):
        if self.dummy:
            return 'Dummy'
        else:
            return str(self.innovation)

def align_connections(
        connections_1: List[Connection],
        connections_2: List[Connection]) -> Tuple[
            List[Connection],
            List[Connection],
            int, int, float]:
    dummy_node = Node(0, NodeType.HIDDEN)
    dummy_connection = Connection(dummy_node, dummy_node, dummy=True)
    end = dummy_connection
    iterators = [chain(i, [end]) for i in [sorted(connections_1),
        sorted(connections_2)]]
    values = [next(i) for i in iterators]
    connections_1 = []
    connections_2 = []
    weights = []
    excess = 0
    disjoint = 0
    while not all(v is end for v in values):
        smallest = min(v for v in values if v is not end)
        alignment = []
        match = True
        for v in values:
            if v == smallest:
                alignment.append(v)
            else:
                match = False
                alignment.append(dummy_connection)
                if values[0] is end or values[1] is end:
                    excess += 1
                else:
                    disjoint += 1
        connection_1, connection_2 = alignment
        if match:
            weights.append(abs(connection_1.weight - connection_2.weight))
        connections_1.append(connection_1)
        connections_2.append(connection_2)
        values = [next(i) if v == smallest else v
                  for i, v in zip(iterators, values)]
    avarage_weight_difference = statistics.mean(weights)
    return connections_1, connections_2, disjoint, excess, avarage_weight_difference
