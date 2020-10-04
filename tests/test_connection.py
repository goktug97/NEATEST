import unittest

from neatest import Connection, DummyConnection, Weight, GeneRate, align_connections
from neatest import Node, NodeType

class TestConnection(unittest.TestCase):
    def test_eq(self):
        connection_1 = DummyConnection(Node(0, NodeType.INPUT),
                                       Node(1, NodeType.OUTPUT))
        connection_2 = DummyConnection(Node(0, NodeType.INPUT),
                                       Node(1, NodeType.OUTPUT))
        self.assertEqual(connection_1, connection_2)

    def test_hash(self):
        in_node = Node(0, NodeType.INPUT)
        out_node = Node(1, NodeType.OUTPUT)
        connection = DummyConnection(in_node, out_node)
        self.assertEqual(
            hash(str(in_node.id)+str(out_node.id)),
            hash(connection))

    def test_out_node(self):
        in_node = Node(0, NodeType.INPUT)
        out_node = Node(1, NodeType.OUTPUT)
        connection = Connection(in_node, out_node)
        self.assertEqual(out_node.inputs[0],
                         connection)

    def test_shared(self):
        weight = Weight(0.0)
        gene_rate = GeneRate(0.0)
        in_node = Node(0, NodeType.INPUT)
        out_node = Node(1, NodeType.OUTPUT)
        connection_1 = Connection(in_node, out_node, 0, gene_rate, weight)
        connection_2 = Connection(in_node, out_node, 1, gene_rate, weight)
        connection_1.weight.value = 1.0
        connection_1.dominant_gene_rate.value = 1.0
        self.assertEqual(connection_1.weight.value, connection_2.weight.value)
        self.assertEqual(connection_1.dominant_gene_rate.value,
                         connection_2.dominant_gene_rate.value)

    def test_align(self):
        in_node = Node(0, NodeType.INPUT)
        out_node = Node(1, NodeType.OUTPUT)
        a = Connection(in_node, out_node, 0)
        b = Connection(in_node, out_node, 1)
        c = Connection(in_node, out_node, 2)
        d = Connection(in_node, out_node, 3)
        e = Connection(in_node, out_node, 4)
        f = Connection(in_node, out_node, 5)
        connections_1, connections_2 = align_connections([a, b, c, f], [c, d, e])

        dummy_node = Node(0, NodeType.HIDDEN)
        dummy_connection = Connection(dummy_node, dummy_node, dummy=True)

        self.assertEqual(
            [a, b, c, dummy_connection, dummy_connection, f],
            connections_1)
        self.assertEqual(
            [dummy_connection, dummy_connection, c, d, e, dummy_connection],
            connections_2)

if __name__ == '__main__':
    unittest.main()
