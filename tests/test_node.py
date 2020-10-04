import unittest

from neatest import Node, NodeType, group_nodes

class TestNode(unittest.TestCase):
    def test_eq(self):
        a = Node(0, NodeType.INPUT)
        b = Node(0, NodeType.INPUT)
        self.assertEqual(a, b)

    def test_hash(self):
        self.assertEqual(
            hash(Node(0, NodeType.INPUT)),
            hash(str(0)))

    def test_group_by_type(self):
        a = Node(0, NodeType.INPUT)
        b = Node(1, NodeType.BIAS)
        c = Node(2, NodeType.OUTPUT)
        d = Node(3, NodeType.OUTPUT)
        e = Node(4, NodeType.HIDDEN)
        nodes = [a, b, c, d, e]
        self.assertEqual(group_nodes(nodes, by='type'),
                         [[a], [b], [e], [c, d]])

    def test_group_by_depth(self):
        a = Node(0, NodeType.INPUT, depth=0.0)
        b = Node(1, NodeType.BIAS, depth=0.0)
        c = Node(2, NodeType.OUTPUT, depth=1.0)
        d = Node(3, NodeType.OUTPUT, depth=1.0)
        e = Node(4, NodeType.HIDDEN, depth=0.5)
        nodes = [a, b, c, d, e]
        self.assertEqual(group_nodes(nodes, by='depth'),
                         [[a, b], [e], [c, d]])

if __name__ == '__main__':
    unittest.main()
