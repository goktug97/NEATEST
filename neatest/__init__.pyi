from abc import ABC, abstractmethod
from typing import Union, List, Callable, Tuple, NewType, Type, Dict
from enum import Enum
import functools
import random

import numpy as np

Array = Union[np.ndarray, np.generic]

class Version():
    major: int
    minor: int
    patch: int
    def __init__(self):
        ...

    def __eq__(self, other) -> bool:
        ...

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
        ...

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
                 value: float = ...,
                 depth: float = ...):
        ...

    def __hash__(self) -> int:
        ...

    def __eq__(self, other: object) -> bool:
        ...

    def copy(self) -> 'Node':
        ...

    def __str__(self) -> str:
        ...

    def __repr__(self) -> str:
        ...

def group_nodes(nodes: List[Node], by: str) -> List[List[Node]]:
    ...

class Weight():
    grad: float
    def __init__(self, value: float):
        ...

class GeneRate():
    def __init__(self, value: float):
        ...

class Connection(object):
    def __init__(self, in_node: Node, out_node: Node, dominant_gene_rate: GeneRate = ...,
                 weight: Weight = ..., dummy: bool = ...):
        ...

    def __gt__(self, other: 'Connection') -> bool:
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

class DummyConnection(Connection):
    def __init__(self, in_node: Node, out_node: Node):
        ...

class Genome(object):
    input_size: int
    output_size: int
    outputs: List[List[Node]]
    version: Version

    def __init__(self, nodes: List['Node'], connections: List[Connection]):
        ...

    def __call__(self, inputs: List[float]) -> List[float]:
        ...

    @property
    def size(self) -> int:
        ...

    def deepcopy(self) -> 'Genome':
        ...

    def copy(self) -> 'Genome':
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

def draw_genome(genome: Genome,
                node_radius: float = ...,
                vertical_distance: float = ...,
                horizontal_distance: float = ...) -> None:
    ...

class ContextGenome(Genome):
    def __init__(self, nodes: List[Node], connections: List[Connection]) -> None:
        ...

    def copy(self) -> 'ContextGenome': #type: ignore
        ...

    def deepcopy(self) -> 'ContextGenome': #type: ignore
        ...

SortedContextGenomes = NewType('SortedContextGenomes', List[ContextGenome])

class Optimizer(ABC):
    def __init__(self, weights: List[Weight], **kwargs):
        ...

    def zero_grad(self) -> None:
        ...

    @abstractmethod
    def step(self) -> None:
        ...

class Agent(ABC):
    @abstractmethod
    def rollout(self, genome: Genome) -> float:
        ...

class Adam(Optimizer):
    m: Array
    v: Array
    t: int
    def __init__(self, weights: List[Weight], lr: float, beta_1: float = ...,
                 beta_2: float = ..., epsilon: float = ...):
        ...

    def step(self) -> None:
        ...

class NEATEST(object):
    population: List[ContextGenome]
    generation: int
    connections: Dict[Connection, int]
    best_fitness: float
    best_genome: ContextGenome
    version: Version
    n_proc: int
    random: random.Random
    np_random: np.random.mtrand.RandomState
    weights: List[Weight]
    gene_rates: List[GeneRate]
    optimizer: Optimizer
    data: List[Tuple[int, str, int, float]]

    def __init__(self,
                 agent: Agent,
                 optimizer: Type[Optimizer],
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
                 seed: int,
                 save_checkpoint_n: int = ...,
                 logdir: str = ...,
                 optimizer_kwargs: dict = ...,
                 hidden_layers: List[int] = ...,
                 elite_rate: float = ...,
                 sigma: float = ...,
                 hidden_activation: Callable[[float], float]=...,
                 output_activation: Callable[[float], float]=...):

        ...


    def register_connection(self, dummy_connection: DummyConnection):
        ...

    def add_connection_mutation(self, genome: Genome) -> None:
        ...

    def add_node_mutation(self, genome: Genome,
                          activation: Callable[[float], float]=...) -> None:
        ...

    def disable_connection_mutation(self, genome: Genome) -> None:
        ...

    def crossover(self, genome_1:Genome, genome_2:Genome) -> Genome:
        ...

    def random_genome(self) -> ContextGenome:
        ...

    def create_population(self) -> None:
        ...

    def save_logs(self) -> None:
        ...
 
    def next_generation(self, sorted_population: SortedContextGenomes) -> None:
        ...

    def calculate_grads(self, genome: ContextGenome) -> None:
        ...

    def train_genome(self, genome: ContextGenome, n_steps: int = ...):
        ...

    def reset_values(self) -> None:
        ...

    def train(self, n_steps: int) -> None:
        ...

    @staticmethod
    def sort_population(population: List[ContextGenome]) -> SortedContextGenomes:
        ...

    def get_random_genome(self) -> ContextGenome:
        ...

    def save_checkpoint(self) -> None:
        ...

    @classmethod
    def load_checkpoint(cls, filename: str) -> 'NEATEST':
        ...
