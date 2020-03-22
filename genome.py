from typing import List, Any, Callable
import math
import copy
import random

from connection import Connection, allign_connections
from node import Node, NodeType, group_nodes_by_depth, group_nodes_by_type


class Genome(object):
    def __init__(self, nodes: List['Node'], connections: List[Connection]):
        self.connections = connections
        self.nodes = nodes

        grouped_nodes = group_nodes_by_type(self.nodes)
        self.input_size = len(grouped_nodes[0])
        self.output_size = len(grouped_nodes[-1])
        self.outputs = grouped_nodes[-1]

        for node in self.nodes:
            node.update_depth()

    def weight_mutation(self) -> None:
        '''Apply weight mutation to connection weights.'''
        for connection in self.connections:
            if random.random() <= 0.1:
                connection.weight = random.random()
            else:
                connection.weight += random.uniform(-0.1, 0.1)

    def add_connection_mutation(self) -> None:
        '''Create new connection between two random non-connected nodes.'''
        in_idx = random.randint(0, len(self.nodes) - 1)
        in_node = self.nodes[in_idx]
        out_idx = random.randint(0, len(self.nodes) - 1)
        out_node = self.nodes[out_idx]
        connection = Connection(in_node, out_node, dummy=True)
        if connection in out_node.inputs or out_node.type == NodeType.BIAS:
            self.add_connection_mutation()
            return
        connection = Connection(in_node, out_node, random.random())
        self.connections.append(connection)
        for node in self.nodes:
            node.update_depth()

    def add_node_mutation(self,
                          activation: Callable[[float], float]=lambda x: x) -> None:
        '''Add a node to a random connection and split the connection.'''
        new_node = Node(max(self.nodes, key=lambda x: x.id)+1,
                        NodeType.HIDDEN, activation)
        idx = random.randint(0, len(self.connections)-1)
        self.connections[idx].enabled = False
        first_connection = Connection(self.connections[idx].in_node, new_node, 1)
        weight = self.connections[idx].weight
        second_connection = Connection(new_node, self.connections[idx].out_node, weight)
        self.connections.append(first_connection)
        self.connections.append(second_connection)
        self.nodes.append(new_node)
        for node in self.nodes:
            node.update_depth()

    def __call__(self, inputs: List[float]) -> List[float]:
        self.nodes.sort(key=lambda x: x.depth)
        for node in self.nodes:
            value = 0.0
            if node.type == NodeType.INPUT:
                value += inputs[node.id]
            for connection in node.inputs:
                if connection.enabled:
                    if connection.in_node.depth >= node.depth:
                        if connection.in_node.old_value is not None:
                            value += connection.in_node.old_value * connection.weight
                    else:
                        value += connection.in_node.value * connection.weight
            node.value = node.activation(value)
        return [node.value for node in self.outputs]

    @property
    def size(self):
        return len(list(filter(lambda x: x.enabled, self.connections)))
        
    def copy(self):
        return copy.deepcopy(self)

    def crossover(self, other: 'Genome'):
        '''Crossover the genome with the other genome.'''
        return crossover(self, other)

    def draw(self) -> None:
        draw_genome(self)

    def distance(self, other: 'Genome',
                 c1: float, c2: float, c3: float) -> float:
        return distance(self, other, c1, c2, c3)

    def __str__(self):
        string = ''
        string = f'{string}NODES:\n'
        for node in self.nodes:
            string = f'{string}{node}\n' 
        string = f'{string}\n\nCONNECTIONS:\n'
        for connection in self.connections:
            string = f'{string}{connection}\n'
        return string

def distance(genome_1: Genome, genome_2: Genome,
             c1: float, c2: float, c3: float) -> float:
    N = (1 if (genome_1.size < 20 and genome_2.size < 20)
         else max(genome_1.size, genome_2.size))
    connections_1: Any = copy.deepcopy(genome_1.connections)
    connections_2: Any = copy.deepcopy(genome_2.connections)
    _, _, disjoint, excess, avarage_weight_difference = allign_connections(
        connections_1, connections_2)
    return excess*c1/N + disjoint*c2/N + avarage_weight_difference*c3

def crossover(genome_1:Genome, genome_2:Genome) -> Genome:
    '''Crossover two genomes by aligning their innovation numbers.'''
    connections: List[Connection] = []
    nodes: List[Node] = []
    connections_1: Any = copy.deepcopy(genome_1.connections)
    connections_2: Any = copy.deepcopy(genome_2.connections)
    connections_1, connections_2, _, _, _ = allign_connections(
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

def draw_genome(genome: Genome,
                node_radius: float = 0.05,
                distance: float = 0.25) -> None:
    '''Draw the genome to a matplotlib figure but do not show it.'''
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    plt.gcf().canvas.set_window_title('float')

    positions = {}
    node_groups = group_nodes_by_depth(genome.nodes)
    for group_idx, nodes in enumerate(node_groups):
        y_position = -distance * (len(nodes)-1)/2
        for i, node in enumerate(nodes):
            positions[f'{node.id}'] = (group_idx * distance,
                                      y_position + i*distance)
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
