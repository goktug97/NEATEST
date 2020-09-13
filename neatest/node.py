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

    def __gt__(self, other):
        return self.value > other.value


class Node(object):
    id_generator = count(0, 1)
    def __init__(self, id: int, type: NodeType,
                 activation: Callable[[float], float] = passthrough,
                 value: Union[float, None] = None,
                 depth: float = 0.0):
        if id == -1:
            self.id = next(self.id_generator)
        else:
            self.id = id
        self.type = type
        self.activation = activation
        self._value = value
        self.old_value = None
        self.depth = depth
        self.visited = False

        if TYPE_CHECKING:
            from .connection import Connection
        self.inputs: List[Connection] = []

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self.old_value, self._value = self._value, value

    def reset_values(self):
        self._value = None
        self.old_value = None

    def __hash__(self):
        return hash(str(self.id))

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.id == other.id
        else:
            raise ValueError(f'Value type should be Node, got {type(other)}')

    def __add__(self, other: Union['Node', int]) -> int:
        if isinstance(other, Node):
            return self.id + other.id
        elif isinstance(other, int):
            return self.id + other
        else:
            raise ValueError(f'Value type should be Node or int, got {type(other)}')

    def copy(self):
        return copy.deepcopy(self)

    def __str__(self):
        return f'{self.id}:{self.type}'

    def __repr__(self):
        return str(self.id)

def group_nodes_by_type(nodes: List[Node]) -> List[List[Node]]:
    sorted_nodes = sorted(nodes, key = lambda x: x.type)
    grouped_nodes = [list(it) for k, it in groupby(sorted_nodes, lambda x: x.type)]
    return grouped_nodes

def group_nodes_by_depth(nodes: List[Node]) -> List[List[Node]]:
    sorted_nodes = sorted(nodes, key = lambda x: x.depth)
    grouped_nodes = [list(it) for k, it in groupby(sorted_nodes, lambda x: x.depth)]
    return grouped_nodes
