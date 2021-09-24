from typing import List, Any, Callable, Tuple
import math
import os

from .connection import Connection, GeneRate, Weight
from .node import Node, NodeType, group_nodes
from .version import VERSION

import cloudpickle #type: ignore
try:
    disable_mpi = os.environ.get('NEATEST_DISABLE_MPI')
    if disable_mpi and disable_mpi != '0':
        raise ImportError
    from mpi4py import MPI
except ImportError:
    from .MPI import MPI
    MPI = MPI()


class Genome(object):
    def __init__(self, nodes: List['Node'], connections: List[Connection]):
        self.connections = connections
        self.nodes = nodes
        self.version = VERSION

        grouped_nodes = group_nodes(self.nodes, 'type')
        self.input_size = len(grouped_nodes[0])
        self.output_size = len(grouped_nodes[-1])
        self.outputs = grouped_nodes[-1]

    def __call__(self, inputs: List[float]) -> List[float]:
        self.nodes.sort(key=lambda x: x.depth)
        for node in self.nodes:
            value = 0.0
            if node.type == NodeType.INPUT:
                value += inputs[node.id]
            elif node.type == NodeType.BIAS:
                continue
            for connection in node.inputs:
                if connection.enabled:
                    value += connection.in_node.value * connection.weight.value
            node.value = node.activation(value)
        return [node.value for node in self.outputs]

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
                                        nodes[nodes_dict[out_node]],
                                        innovation = connection.innovation,
                                        dominant_gene_rate =
                                        connection.dominant_gene_rate,
                                        weight = connection.weight)
            new_connection.enabled = connection.enabled
            connections.append(new_connection)
        new_genome = Genome(nodes, connections)
        return new_genome

    def deepcopy(self):
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
                                        nodes[nodes_dict[out_node]],
                                        innovation = connection.innovation,
                                        dominant_gene_rate =
                                        GeneRate(connection.dominant_gene_rate.value),
                                        weight = Weight(connection.weight.value))
            new_connection.enabled = connection.enabled
            connections.append(new_connection)
        new_genome = Genome(nodes, connections)
        return new_genome

    def draw(self, node_radius: float = 0.05,
             vertical_distance: float = 0.25,
             horizontal_distance: float = 0.25) -> None:
        draw_genome(self, node_radius, vertical_distance, horizontal_distance)

    def save(self, filename: str) -> None:
        if MPI.COMM_WORLD.rank == 0:
            with open(filename, 'wb') as output:
                cloudpickle.dump(self, output)

    @classmethod
    def load(cls, filename: str) -> 'Genome':
        print(f"\033[33;1mLoading: {filename}\033[0m")
        with open(filename, 'rb') as f:
            genome = cloudpickle.load(f)
        if genome.version != VERSION:
            print("\033[31;1mWarning: Genome version mismatch!\n"
                  f"Current Version: {VERSION.major}.{VERSION.minor}.{VERSION.patch}\n"
                  "Checkpoint Version:"
                  f" {genome.version.major}.{genome.version.minor}."
                  f"{genome.version.patch}\033[0m")
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

def draw_genome(genome: Genome,
                node_radius: float = 0.05,
                vertical_distance: float = 0.25,
                horizontal_distance: float = 0.25) -> None:
    '''Draw the genome to a matplotlib figure but do not show it.'''
    import matplotlib.pyplot as plt #type: ignore
    import matplotlib.patches as patches #type: ignore
    plt.gcf().canvas.set_window_title('float')

    positions = {}
    node_groups = group_nodes(genome.nodes, 'depth')
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
