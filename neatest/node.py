import functools
from itertools import groupby, count
from typing import Callable, Union, List, TYPE_CHECKING
import copy
from enum import Enum

import numpy as np

def passthrough(x):
    return x

def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))

def steepened_sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-4.9 * x))

def relu(x: float) -> float:
    return max(x, 0.0)

def leaky_relu(x: float) -> float:
    return max(0.1*x, x)

def tanh(x: float) -> float:
    return float(np.tanh(x))


@functools.total_ordering
class NodeType(Enum):
    INPUT = 1
    BIAS = 2
    HIDDEN = 3
    OUTPUT = 4

    def __gt__(self, other) -> bool:
        return self.value > other.value

@functools.total_ordering
class Node(object):
    def __init__(self, id: int, type: NodeType,
                 activation: Callable[[float], float] = passthrough,
                 value: float = 0.0,
                 depth: float = 0.0):
        self.type = type
        self.id = id
        self.activation = activation
        self.value = value
        self.depth = depth

        if TYPE_CHECKING:
            from .connection import Connection
        self.inputs: List[Connection] = []

    def __hash__(self):
        return hash(str(self.id))

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.id == other.id
        else:
            raise ValueError(f'Value type should be Node, got {type(other)}')

    def __lt__(self, other):
        if isinstance(other, Node):
            return self.id < other.id
        else:
            raise ValueError(f'Value type should be Node, got {type(other)}')

    def __add__(self, other):
        if isinstance(other, Node):
            return self.id + other.id
        else:
            return self.id + other

    def copy(self):
        return copy.deepcopy(self)

    def __str__(self):
        return f'{self.id}:{self.type}'

    def __repr__(self):
        return str(self.id)


def group_nodes(nodes: List[Node], by) -> List[List[Node]]:
    sorted_nodes = sorted(nodes, key = lambda x: getattr(x, by))
    grouped_nodes = [list(it) for k, it in groupby(
        sorted_nodes, lambda x: getattr(x, by))]
    return grouped_nodes
