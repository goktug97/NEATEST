from abc import ABC, abstractmethod
from typing import Union, List, Callable, Tuple, NewType, Type
from enum import Enum
import functools

import numpy as np

Array = Union[np.ndarray, np.generic]

@functools.lru_cache(maxsize=1)
def _center_function(population_size: int) -> Array:
    ...

def _compute_ranks(rewards: Union[List[float], Array]) -> Array:
    ...

def rank_transformation(rewards: Union[List[float], Array]) -> Array:
    ...

@functools.total_ordering
class NodeType(Enum):
    def __gt__(self, other: NodeType) -> bool:
        return self.value > other.value

def passthrough(x: float) -> float:
    ...

def sigmoid(x: float) -> float:
    ...

def steepened_sigmoid(x: float) -> float:
    ...

def relu(x: float) -> float:
    ...

def leaky_relu(x: float) -> float:
    ...

def tanh(x: float) -> float:
    ...

class Node():
    def __init__(self, id: int, type: NodeType,
                 activation: Callable[[float], float] = ...,
                 value: Union[float, None] = ...,
                 depth: float = ...):
        ...

    @property
    def value(self) -> float:
        ...

    @value.setter
    def value(self, value: float) -> None:
        ...

    def reset_values(self) -> None:
        ...

    def __hash__(self) -> int:
        ...

    def __eq__(self, other: object) -> bool:
        ...

    def __add__(self, other: Union['Node', int]) -> int:
        ...

    def copy(self) -> 'Node':
        ...

    def __str__(self) -> str:
        ...

    def __repr__(self) -> str:
        ...

def group_nodes_by_type(nodes: List[Node]) -> List[List[Node]]:
    ...

def group_nodes_by_depth(nodes: List[Node]) -> List[List[Node]]:
    ...

class Connection(object):
    def __init__(self, in_node: Node, out_node: Node, dominant_gene_rate: float = ...,
                 weight: float = ..., dummy: bool = ...):
        ...

    def __gt__(self, other: 'Connection') -> bool:
        ...

    @property
    def weight(self) -> float:
        ...

    @weight.setter
    def weight(self, value: float) -> None:
        ...

    @property
    def grad(self) -> float:
        ...

    @grad.setter
    def grad(self, value: float) -> None:
        ...

    @property
    def dominant_gene_rate(self) -> float:
        ...

    @dominant_gene_rate.setter
    def dominant_gene_rate(self, value: float) -> None:
        ...

    @classmethod
    def register_connection(cls, new_connection:'Connection', weight: float,
                            dominant_gene_rate: float) -> int:
        ...

    def __hash__(self) -> int:
        ...

    def __eq__(self, other: object) -> bool:
        ...

    def copy(self) -> 'Connection':
        ...

    def __str__(self) -> str:
        ...

    def __repr__(self) -> str:
        ...

def align_connections(
        connections_1: List[Connection],
        connections_2: List[Connection]) -> Tuple[
            List[Connection],
            List[Connection]]:
    ...

class Genome(object):
    connections: List[Connection]
    nodes: List[Node]

    def __init__(self, nodes: List['Node'], connections: List[Connection]):
        ...

    def add_connection_mutation(self, sigma: float, dominant_gene_rate: float) -> None:
        ...

    def add_node_mutation(self,
                          dominant_gene_rate: float,
                          activation: Callable[[float], float]=...) -> None:
        ...

    def disable_connection_mutation(self) -> None:
        ...

    def __call__(self, inputs: List[float]) -> List[float]:
        ...

    @property
    def size(self) -> int:
        ...

    def copy(self) -> 'Genome':
        ...

    def crossover(self, other: 'Genome') -> 'Genome':
        ...

    def draw(self, node_radius: float = ...,
             vertical_distance: float = ...,
             horizontal_distance: float = ...) -> None:
        ...

    def save(self, filename: str) -> None:
        ...

    @classmethod
    def load(cls, filename: str) -> 'Genome':
        ...

    def __str__(self) -> str:
        ...

def crossover(genome_1:Genome, genome_2:Genome) -> Genome:
    ...

def draw_genome(genome: Genome,
                node_radius: float = ...,
                vertical_distance: float = ...,
                horizontal_distance: float = ...) -> None:
    ...

class ContextGenome(Genome):
    def __init__(self, nodes: List[Node], connections: List[Connection]) -> None:
        ...

    def crossover(self, other: ContextGenome) -> ContextGenome: #type: ignore
        ...

SortedContextGenomes = NewType('SortedContextGenomes', List[ContextGenome])

class Optimizer(ABC):

    @staticmethod
    def zero_grad() -> None:
        ...

    @abstractmethod
    def step(self) -> None:
        ...

class Agent(ABC):
    @abstractmethod
    def rollout(self, genome: Genome) -> float:
        ...

class Adam(Optimizer):
    def __init__(self, lr: float, beta_1: float = ..., beta_2:
                 float = ..., epsilon: float = ...):
        ...

    def step(self) -> None:
        ...

class NEATEST(object):
    population: List[ContextGenome]
    generation: int
    best_fitness: float
    best_genome: ContextGenome

    def __init__(self,
                 agent: Agent,
                 optimizer: Optimizer,
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
                 elite_rate: float = ...,
                 sigma: float = ...,
                 hidden_activation: Callable[[float], float]=...,
                 output_activation: Callable[[float], float]=...):

        ...


    def random_genome(self) -> ContextGenome:
        ...

    def create_population(self) -> None:
        ...

    def next_generation(self, sorted_population: SortedContextGenomes) -> None:
        ...

    def calculate_grads(self, genome: ContextGenome) -> None:
        ...

    def train(self, n_steps: int) -> None:
        ...

    @staticmethod
    def sort_population(population: List[ContextGenome]) -> SortedContextGenomes:
        ...

    def get_random_genome(self, sorted_population: SortedContextGenomes) -> ContextGenome:
        ...

    def save_checkpoint(self) -> None:
        ...

    @classmethod
    def load_checkpoint(cls, filename: str) -> 'NEATEST':
        ...
