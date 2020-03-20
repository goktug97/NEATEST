#!/usr/bin/env python3
import random
from enum import Enum
import copy
from typing import Union, List, Set, Any, Tuple, Dict, Callable
from itertools import chain, repeat, islice, groupby
import functools
import math
import statistics


def pad_list(iterable, size, padding=None):
   return list(islice(chain(iterable, repeat(padding)), size))


@functools.total_ordering
class NodeType(Enum):
    INPUT = 1
    HIDDEN = 2
    OUTPUT = 3

    def __gt__(self, other):
        return self.value > other.value


class Node(object):
    def __init__(self, id: int, type: NodeType,
                 activation: Callable[[float], float] = lambda x: x):
        self.id = id
        self.type = type
        self.activation = activation
        self._value = None
        self.old_value = None
        self.depth = float('inf')
        self.visited = False

        self.inputs: List[Node] = []
        self.outputs: List[Node] = []

    def calculate_depth(self, connections: List['Connection'], depth:int=0) -> float:
        '''Calculate minimum distance to a input node, infinite if node is an output.'''
        min_depth = float('inf')
        if self.type == NodeType.OUTPUT:
            return float('inf')
        connections_dict = dict(zip(connections, range(NEAT.global_innovation)))
        if self.type == NodeType.INPUT:
            self.visited = False
            return depth
        self.visited = True
        for node in self.inputs:
            if node.visited:
                continue
            idx = connections_dict[Connection(node, self, dummy=True)]
            connection = connections[idx]
            if connection.enabled:
                tmp =  node.calculate_depth(connections, depth+1)
                if tmp < min_depth:
                    min_depth = tmp
        self.visited = False
        return min_depth

    def update_depth(self, connections: List['Connection']) -> None:
        self.depth = self.calculate_depth(connections)

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


class Connection(object):
    def __init__(self, in_node: Node, out_node: Node, weight: float = 1.0,
                 dummy: bool = False):
        self.in_node = in_node
        self.out_node = out_node
        if dummy: return
        self.weight = weight
        self.enabled = True
        self.innovation = NEAT.add_connection(self)

        self.in_node.outputs.append(out_node)
        self.out_node.inputs.append(in_node)

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


class Genome(object):
    def __init__(self, nodes: List[Node], connections: List[Connection]):
        self.connections = connections
        self.nodes = nodes

        grouped_nodes = NEAT.group_nodes_by_type(self.nodes)
        self.input_size = len(grouped_nodes[0])
        self.output_size = len(grouped_nodes[2])
        self.outputs = grouped_nodes[-1]

        for node in self.nodes:
            node.update_depth(self.connections)

    def weight_mutation(self) -> None:
        '''Apply weight mutation to connection weights.'''
        for connection in self.connections:
            if random.random() <= 0.1:
                connection.weight = random.random()
            else:
                connection.weight += random.uniform(-0.1, 0.1)

    def add_connection_mutation(self) -> None:
        '''Create new connection between two random non-connected nodes.'''
        in_idx = random.randint(0, len(self.nodes)-1)
        in_node = self.nodes[in_idx]
        out_idx= random.randint(0, len(self.nodes)-1)
        out_node = self.nodes[out_idx]
        connection = Connection(in_node, out_node, dummy=True)
        if connection in self.connections:
            return
        connection = Connection(in_node, out_node, random.random())
        self.connections.append(connection)
        for node in self.nodes:
            node.update_depth(self.connections)

    def add_node_mutation(self) -> None:
        '''Add a node to a random connection and split the connection.'''
        new_node = Node(max(self.nodes, key=lambda x: x.id)+1, NodeType.HIDDEN)
        idx = random.randint(0, len(self.connections)-1)
        self.connections[idx].enabled = False
        first_connection = Connection(self.connections[idx].in_node, new_node, 1)
        weight = self.connections[idx].weight
        second_connection = Connection(new_node, self.connections[idx].out_node, weight)
        self.connections.append(first_connection)
        self.connections.append(second_connection)
        self.nodes.append(new_node)
        for node in self.nodes:
            node.update_depth(self.connections)

    def __call__(self, inputs: List[float]) -> List[float]:
        self.nodes.sort(key=lambda x: x.depth)
        connections = dict(zip(self.connections, range(NEAT.global_innovation)))
        for node in self.nodes:
            value = 0.0
            if node.type == NodeType.INPUT:
                value += inputs[node.id]
            for input_node in node.inputs:
                idx = connections[Connection(input_node, node, dummy=True)]
                connection = self.connections[idx]
                if connection.enabled:
                    if input_node.depth >= node.depth:
                        if input_node.old_value is not None:
                            value += input_node.old_value * connection.weight
                    else:
                        value += input_node.value * connection.weight
            node.value = node.activation(value)
        return [node.value for node in self.outputs]
        
    def copy(self):
        return copy.deepcopy(self)

    def crossover(self, other: 'Genome') -> 'Genome':
        '''Crossover the genome with the other genome.'''
        return NEAT.crossover(self, other)

    def draw(self) -> None:
        NEAT.draw_genome(self)

    def distance(self, other: 'Genome',
                 c1: float, c2: float, c3: float, N: int) -> float:
        return NEAT.distance(self, other, c1, c2, c3, N)

    def __str__(self):
        string = ''
        string = f'{string}NODES:\n'
        for node in self.nodes:
            string = f'{string}{node}\n' 
        string = f'{string}\n\nCONNECTIONS:\n'
        for connection in self.connections:
            string = f'{string}{connection}\n'
        return string


class NEAT(object):
    connections: Dict[Connection, int] = {}
    global_innovation: int = 0 

    def __init__(self, size: int, input_size: int, output_size: int):
        self.population = self.create_population(size, input_size, output_size)
        self.generation = 0

    def forward(self, inputs: List[List[float]]) -> List[List[float]]:
        outputs: List[List[float]] = []
        for input, genome in zip(inputs, self.population):
            outputs.append(genome(input))
        return outputs

    @staticmethod
    def create_population(size: int, input_size: int, output_size: int) -> List[Genome]:
        population: List[Genome] = []
        for _ in range(size):
            population.append(NEAT.random_genome(input_size, output_size))
        return population

    @staticmethod
    def random_genome(input_size: int, output_size: int) -> Genome:
        '''Create fc neural network without hidden layers with random weights.'''
        connections: List[Connection] = []
        input_nodes = [Node(i, NodeType.INPUT) for i in range(input_size)]
        output_nodes = [Node(i+input_size, NodeType.OUTPUT) for i in range(output_size)]
        for i in range(input_size):
            input_node = input_nodes[i]
            for j in range(output_size):
                output_node = output_nodes[j]
                connections += [Connection(input_node, output_node, random.random())]
        return Genome(input_nodes+output_nodes, connections)

    @staticmethod
    def group_nodes_by_type(nodes: List[Node]) -> List[List[Node]]:
        sorted_nodes = sorted(nodes, key = lambda x: x.type)
        grouped_nodes = [list(it) for k, it in groupby(sorted_nodes, lambda x: x.type)]
        if len(grouped_nodes) == 2:
            grouped_nodes.insert(1, [])
        return grouped_nodes

    @staticmethod
    def group_nodes_by_depth(nodes: List[Node]) -> List[List[Node]]:
        sorted_nodes = sorted(nodes, key = lambda x: x.depth)
        grouped_nodes = [list(it) for k, it in groupby(sorted_nodes, lambda x: x.depth)]
        return grouped_nodes

    @classmethod
    def add_connection(cls, new_connection:Connection) -> int:
        if new_connection in cls.connections:
            return cls.connections[new_connection]
        else:
            cls.global_innovation += 1
            cls.connections[new_connection] = cls.global_innovation
            return cls.global_innovation

    @staticmethod
    def allign_connections(
            connections_1: List[Union[Connection, Any]],
            connections_2: List[Union[Connection, Any]]) -> Tuple[
                List[Union[Connection, None]],
                List[Union[Connection, None]],
                int, int, float]:
        '''Destructively allign connections by their innovation number.'''
        connections_1.sort(key = lambda x: x.innovation)
        connections_2.sort(key = lambda x: x.innovation)
        weights = []
        disjoint = 0
        for i in range(min(len(connections_1), len(connections_2))):
            if connections_1[i].innovation > connections_2[i].innovation:
                connections_1.insert(i, None)
                disjoint +=1
            elif connections_1[i].innovation < connections_2[i].innovation:
                connections_2.insert(i, None)
                disjoint +=1
            else:
                weights.append(abs(connections_1[i].weight - connections_2[i].weight))

        avarage_weight_difference = statistics.mean(weights)
        max_length = max(len(connections_1), len(connections_2))
        excess = max_length - min(len(connections_1), len(connections_2))
        connections_1 = pad_list(connections_1, max_length)
        connections_2 = pad_list(connections_2, max_length)
        return connections_1, connections_2, disjoint, excess, avarage_weight_difference

    @staticmethod
    def distance(genome_1: Genome, genome_2: Genome,
                 c1: float, c2: float, c3: float, N: int) -> float:
        connections_1: Any = copy.deepcopy(genome_1.connections)
        connections_2: Any = copy.deepcopy(genome_2.connections)
        _, _, disjoint, excess, avarage_weight_difference = NEAT.allign_connections(
            connections_1, connections_2)
        return excess*c1/N + disjoint*c2/N + avarage_weight_difference*c3

    @staticmethod
    def merge_nodes(nodes_1: List[Node], nodes_2: List[Node]) -> List[Node]:
        nodes = nodes_1 + nodes_2
        nodes = sorted(nodes, key = lambda x: x.id)
        grouped_nodes = [list(it) for k, it in groupby(nodes, lambda x: x.id)]
        nodes = []
        for group in grouped_nodes:
            new_node = group[0].copy()
            if len(group) > 1:
                new_node.inputs = list(set(new_node.inputs + group[1].inputs))
                new_node.outputs = list(set(new_node.outputs + group[1].outputs))
            nodes.append(new_node)
        return nodes

    @staticmethod
    def crossover(genome_1:Genome, genome_2:Genome) -> Genome:
        '''Crossover two genomes by aligning their innovation numbers.'''
        connections: List[Connection] = []
        nodes: List[Node] = []
        connections_1: Any = copy.deepcopy(genome_1.connections)
        connections_2: Any = copy.deepcopy(genome_2.connections)
        connections_1, connections_2, _, _, _ = NEAT.allign_connections(
            connections_1, connections_2)

        for idx in range(len(connections_1)):
            connection_1 = connections_1[idx]
            connection_2 = connections_2[idx]
            if connection_1 is None:
                connection = connection_2.copy()
                if not connection_2.enabled:
                    connection.enabled = random.random() > 0.75
            elif connection_2 is None:
                connection = connection_1.copy()
                if not connection_1.enabled:
                    connection.enabled = random.random() > 0.75
            else:
                connection = random.choice([connection_1, connection_2]).copy()
                if not (connection_1.enabled and connection_2.enabled):
                    connection.enabled = random.random() > 0.75
            in_node = Node(connection.in_node.id, connection.in_node.type,
                           connection.in_node.activation)
            out_node = Node(connection.out_node.id, connection.out_node.type,
                           connection.out_node.activation)

            nodes_dict = dict(zip(nodes, range(len(nodes))))
            if in_node not in nodes_dict:
                nodes.append(in_node)
                nodes_dict[in_node] = len(nodes)-1
            if out_node not in nodes_dict:
                nodes.append(out_node)
                nodes_dict[out_node] = len(nodes)-1
            connection = Connection(nodes[nodes_dict[in_node]],
                                    nodes[nodes_dict[out_node]],
                                    connection.weight)
            connections.append(connection)
        return Genome(nodes, connections)

    @staticmethod
    def draw_genome(genome: Genome, hidden_nodes_per_layer: int = 4,
                    node_radius: float = 0.05,
                    distance: float = 0.25) -> None:
        '''Draw the genome to a matplotlib figure but do not show it.'''
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        plt.gcf().canvas.set_window_title('float')

        positions = {}
        node_groups = NEAT.group_nodes_by_depth(genome.nodes)
        for group_idx, nodes in enumerate(node_groups):
            y_position = -distance * (len(nodes)-1)/2
            for i, node in enumerate(nodes):
                positions[f'{node.id}'] = (group_idx * distance,
                                           y_position + i*distance)

        for node in genome.nodes:
            circle = plt.Circle(positions[f'{node.id}'],
                                node_radius, color='r', fill=False)
            text_x, text_y = positions[f'{node.id}']
            plt.gcf().gca().text(*positions[f'{node.id}'], node.id,
                                 horizontalalignment='center',
                                 verticalalignment='center',
                                 fontsize=10.0)
            plt.gcf().gca().add_artist(circle)

        kw = dict(arrowstyle="Simple,tail_width=0.5,head_width=4,head_length=8",
                  color="k", antialiased=True)
        for connection in genome.connections:
            if connection.enabled:
                node1_x = positions[f'{connection.in_node.id}'][0]
                node2_x = positions[f'{connection.out_node.id}'][0]
                node1_y = positions[f'{connection.in_node.id}'][1]
                node2_y = positions[f'{connection.out_node.id}'][1]
                angle = math.atan2(node2_x - node1_x, node2_y - node1_y)
                x_adjustment = node_radius * math.sin(angle)
                y_adjustment = node_radius * math.cos(angle)
                connectionstyle = 'arc3'
                arrow = patches.FancyArrowPatch((node1_x + x_adjustment,
                                                 node1_y + y_adjustment),
                                                (node2_x - x_adjustment,
                                                 node2_y - y_adjustment),
                                                connectionstyle=connectionstyle,
                                                **kw)
                plt.gcf().gca().add_patch(arrow)
        plt.axis('scaled')

        
if __name__ == '__main__':
    # TESTING
    import matplotlib.pyplot as plt
    # population = NEAT.create_population(100, 3, 2)
    # print(population)
    new_genome_1 = NEAT.random_genome(4, 2)
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

    new_genome_2 = NEAT.random_genome(4, 2)
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

    new_genome_3 = NEAT.crossover(new_genome_1, new_genome_2)
    new_genome_3.add_node_mutation()
    new_genome_3.add_node_mutation()
    new_genome_3.add_node_mutation()
    new_genome_3.weight_mutation()
    new_genome_3([1.1, 1.2, 2.3, 3.1])
    print(new_genome_3)
    new_genome_3.draw()
    plt.show()
