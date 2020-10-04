import copy
from typing import List, Union, Tuple, Dict
import statistics
from itertools import chain, repeat, islice

from .node import Node, NodeType


class Weight():
    def __init__(self, value):
        self.value = value
        self.grad = 0.0


class GeneRate():
    def __init__(self, value):
        self.value = value


class Connection(object):

    def __init__(self, in_node: Node, out_node: Node, innovation: int = -1,
                 dominant_gene_rate: GeneRate = GeneRate(0.0),
                 weight: Weight = Weight(1.0), dummy: bool = False):
        self.in_node = in_node
        self.out_node = out_node
        self.enabled = True
        self.dummy = dummy
        self.innovation = innovation
        self.dominant_gene_rate = dominant_gene_rate
        self.weight = weight
        if dummy: return
        self.out_node.inputs.append(self)

    def __gt__(self, other):
        return self.innovation > other.innovation

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
        string = f'{string}Weight: {self.weight.value:.3f} '
        string = f'{string}Dominant Gene Rate: {self.dominant_gene_rate.value:.3f} '
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
            List[Connection]]:
    dummy_node = Node(0, NodeType.HIDDEN)
    dummy_connection = DummyConnection(dummy_node, dummy_node)
    end = dummy_connection
    iterators = [chain(i, [end]) for i in [sorted(connections_1),
        sorted(connections_2)]]
    values = [next(i) for i in iterators]
    connections_1 = []
    connections_2 = []
    while not all(v is end for v in values):
        smallest = min(v for v in values if v is not end)
        alignment = []
        match = True
        for v in values:
            if v.innovation == smallest.innovation:
                alignment.append(v)
            else:
                match = False
                alignment.append(dummy_connection)
        connection_1, connection_2 = alignment
        connections_1.append(connection_1)
        connections_2.append(connection_2)
        values = [next(i) if v.innovation == smallest.innovation else v
                  for i, v in zip(iterators, values)]
    return connections_1, connections_2

class DummyConnection(Connection):
    def __init__(self, in_node: Node, out_node: Node):
        super().__init__(in_node, out_node, dummy=True)
