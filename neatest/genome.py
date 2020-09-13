from typing import List, Any, Callable, Tuple
import math
import copy
import random
import pickle

from .connection import Connection, align_connections
from .node import Node, NodeType, group_nodes_by_depth, group_nodes_by_type


class Genome(object):
    def __init__(self, nodes: List['Node'], connections: List[Connection]):
        self.connections = connections
        self.nodes = nodes

        grouped_nodes = group_nodes_by_type(self.nodes)
        self.input_size = len(grouped_nodes[0])
        self.output_size = len(grouped_nodes[-1])
        self.outputs = grouped_nodes[-1]

    def add_connection_mutation(self, sigma: float, dominant_gene_rate: float) -> None:
        '''Create new connection between two random non-connected nodes.'''
        def _add_connection_mutation(depth = 0):
            if depth > 20:
                return
            in_idx = random.randint(0, len(self.nodes) - 1)
            in_node = self.nodes[in_idx]
            out_idx = random.randint(0, len(self.nodes) - 1)
            out_node = self.nodes[out_idx]
            connection = Connection(in_node, out_node, dummy=True)
            try:
                index = self.connections.index(connection)
                if not self.connections[index].enabled:
                    if random.random() <= self.connections[index].dominant_gene_rate:
                        self.connections[index].enabled = True
                        return
                else:
                    _add_connection_mutation(depth+1)
                    return
            except ValueError:
                pass

            if (out_node.type == NodeType.BIAS or
                out_node.type == NodeType.INPUT or
                in_node.type == NodeType.OUTPUT):
                _add_connection_mutation(depth+1)
                return
            connection = Connection(in_node=in_node, out_node=out_node,
                                    dominant_gene_rate=dominant_gene_rate,
                                    weight=random.gauss(0.0, sigma))
            self.connections.append(connection)
        _add_connection_mutation()

    def add_node_mutation(self,
                          dominant_gene_rate: float,
                          activation: Callable[[float], float]=lambda x: x) -> None:
        '''Add a node to a random connection and split the connection.'''
        idx = random.randint(0, len(self.connections)-1)
        self.connections[idx].enabled = False
        new_node = Node(-1,
                        NodeType.HIDDEN, activation)
        first_connection = Connection(in_node=self.connections[idx].in_node,
                                      out_node=new_node,
                                      dominant_gene_rate=dominant_gene_rate,
                                      weight=1)
        weight = self.connections[idx].weight
        second_connection = Connection(in_node=new_node,
                                       out_node=self.connections[idx].out_node,
                                       dominant_gene_rate=dominant_gene_rate,
                                       weight=weight)
        self.connections.append(first_connection)
        self.connections.append(second_connection)
        new_node.depth = (first_connection.in_node.depth +
                          second_connection.out_node.depth) / 2
        self.nodes.append(new_node)

    def disable_connection_mutation(self) -> None:
        def _disable_connection_mutation(depth = 0):
            if depth > 20:
                return
            idx = random.randint(0, len(self.connections)-1)
            if (self.connections[idx].out_node.type == NodeType.OUTPUT or
                self.connections[idx].in_node.type == NodeType.INPUT or
                self.connections[idx].in_node.type == NodeType.BIAS):
                _disable_connection_mutation(depth + 1)
                return
            else:
                if not self.connections[idx].enabled:
                    _disable_connection_mutation(depth + 1)
                    return
                else:
                    self.connections[idx].enabled = False
                    return
        _disable_connection_mutation()

    def __call__(self, inputs: List[float]) -> List[float]:
        self.nodes.sort(key=lambda x: x.depth)
        for node in self.nodes:
            value = 0.0
            if node.type == NodeType.INPUT:
                value += inputs[node.id]
            for connection in node.inputs:
                if connection.enabled:
                    if (connection.in_node.depth >= node.depth):
                        if connection.in_node.old_value is not None:
                            value += connection.in_node.old_value * connection.weight
                    else:
                        value += connection.in_node.value * connection.weight
            node.value = node.activation(value)
        return [node.value for node in self.outputs]

    @property
    def size(self) -> int:
        return len(list(filter(lambda x: x.enabled, self.connections)))

    def reset_values(self) -> None:
        for node in self.nodes:
            node.value = 0.0

    def copy(self):
        connections: List[Connection] = []
        nodes: List[Node] = []
        for idx in range(len(self.connections)):
            connection = self.connections[idx]
            in_node = Node(connection.in_node.id, connection.in_node.type,
                           connection.in_node.activation,
                           depth = connection.in_node.depth)
            out_node = Node(connection.out_node.id, connection.out_node.type,
                            connection.out_node.activation,
                            depth = connection.out_node.depth)
            nodes_dict = dict(zip(nodes, range(len(nodes))))
            if in_node not in nodes_dict:
                nodes.append(in_node)
                nodes_dict[in_node] = len(nodes)-1
            if out_node not in nodes_dict:
                nodes.append(out_node)
                nodes_dict[out_node] = len(nodes)-1
            new_connection = Connection(nodes[nodes_dict[in_node]],
                                    nodes[nodes_dict[out_node]])
            new_connection.enabled = connection.enabled
            connections.append(new_connection)
        new_genome = Genome(nodes, connections)
        return new_genome

    def crossover(self, other: 'Genome'):
        '''Crossover the genome with the other genome.'''
        return crossover(self, other)

    def draw(self, node_radius: float = 0.05,
             vertical_distance: float = 0.25,
             horizontal_distance: float = 0.25) -> None:
        draw_genome(self, node_radius, vertical_distance, horizontal_distance)

    def save(self, filename: str) -> None:
        with open(filename, 'wb') as output:
            pickle.dump(self.__dict__, output, -1)

    @classmethod
    def load(cls, filename: str) -> 'Genome':
        with open(filename, 'rb') as f:
            genome_dict = pickle.load(f)
        genome: 'Genome' = cls.__new__(cls)
        genome.__dict__.update(genome_dict)
        return genome

    def __str__(self):
        string = ''
        string = f'{string}NODES:\n'
        for node in self.nodes:
            string = f'{string}{node}\n'
        string = f'{string}\n\nCONNECTIONS:\n'
        for connection in self.connections:
            string = f'{string}{connection}\n'
        return string

def crossover(genome_1:Genome, genome_2:Genome) -> Genome:
    '''Crossover two genomes by aligning their innovation numbers.'''
    connections: List[Connection] = []
    nodes: List[Node] = []
    connections_1, connections_2 = align_connections(
        genome_1.connections, genome_2.connections)

    for idx in range(len(connections_1)):
        connection_1 = connections_1[idx]
        connection_2 = connections_2[idx]
        enabled: bool
        connection: Connection
        if connection_1.dummy or connection_2.dummy:
            if connection_1.dummy:
                connection = connection_2
            else:
                connection = connection_1
            if connection.enabled:
                if random.random() <= connection.dominant_gene_rate:
                    enabled = True
                else:
                    enabled = False
            else:
                continue
        else:
            connection = connection_1
            if connection_1.enabled and connection_2.enabled:
                enabled = True
            elif connection_1.enabled ^ connection_2.enabled:
                enabled = random.random() <= connection.dominant_gene_rate
            else:
                if random.random() <= connection.dominant_gene_rate:
                    enabled = False
                else:
                    continue

        in_node = Node(connection.in_node.id, connection.in_node.type,
                       connection.in_node.activation,
                       depth = connection.in_node.depth)
        out_node = Node(connection.out_node.id, connection.out_node.type,
                        connection.out_node.activation,
                        depth = connection.out_node.depth)

        nodes_dict = dict(zip(nodes, range(len(nodes))))
        if in_node not in nodes_dict:
            nodes.append(in_node)
            nodes_dict[in_node] = len(nodes)-1
        if out_node not in nodes_dict:
            nodes.append(out_node)
            nodes_dict[out_node] = len(nodes)-1
        connection = Connection(nodes[nodes_dict[in_node]],
                                nodes[nodes_dict[out_node]])
        connection.enabled = enabled
        connections.append(connection)
    new_genome = Genome(nodes, connections)
    return new_genome

def draw_genome(genome: Genome,
                node_radius: float = 0.05,
                vertical_distance: float = 0.25,
                horizontal_distance: float = 0.25) -> None:
    '''Draw the genome to a matplotlib figure but do not show it.'''
    import matplotlib.pyplot as plt #type: ignore
    import matplotlib.patches as patches #type: ignore
    plt.gcf().canvas.set_window_title('float')

    positions = {}
    node_groups = group_nodes_by_depth(genome.nodes)
    for group_idx, nodes in enumerate(node_groups):
        y_position = -vertical_distance * (len(nodes)-1)/2
        for i, node in enumerate(nodes):
            positions[f'{node.id}'] = (group_idx * horizontal_distance,
                                       y_position + i*vertical_distance)
            circle = plt.Circle(positions[f'{node.id}'],
                                node_radius, color='r', fill=False)
            plt.gcf().gca().text(*positions[f'{node.id}'], node.id,
                                 horizontalalignment='center',
                                 verticalalignment='center',
                                 fontsize=10.0)
            plt.gcf().gca().add_artist(circle)

    for connection in genome.connections:
        if connection.enabled:
            node1_x = positions[f'{connection.in_node.id}'][0]
            node2_x = positions[f'{connection.out_node.id}'][0]
            node1_y = positions[f'{connection.in_node.id}'][1]
            node2_y = positions[f'{connection.out_node.id}'][1]
            angle = math.atan2(node2_x - node1_x, node2_y - node1_y)
            x_adjustment = node_radius * math.sin(angle)
            y_adjustment = node_radius * math.cos(angle)
            arrow = patches.FancyArrowPatch(
                (node1_x + x_adjustment,
                 node1_y + y_adjustment),
                (node2_x - x_adjustment,
                 node2_y - y_adjustment),
                arrowstyle="Simple,tail_width=0.5,head_width=3,head_length=5",
                color="k", antialiased=True)
            plt.gcf().gca().add_patch(arrow)
    plt.axis('scaled')
