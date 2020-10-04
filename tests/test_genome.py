import unittest

from neatest import Connection, Weight, GeneRate
from neatest import Node, NodeType
from neatest import Genome

class TestGenome(unittest.TestCase):
    def setUp(self):
        na = Node(0, NodeType.INPUT, depth=0.0)
        nb = Node(1, NodeType.INPUT, depth=0.0)
        nc = Node(2, NodeType.BIAS, value=1.0, depth=0.0)
        nd = Node(3, NodeType.OUTPUT, depth=1.0)
        ne = Node(4, NodeType.OUTPUT, depth=1.0)
        nf = Node(5, NodeType.HIDDEN, depth=0.5)
        ca = Connection(na, nd, 0, GeneRate(0.0), Weight(1.0))
        cb = Connection(na, ne, 1, GeneRate(0.0), Weight(1.0))
        cc = Connection(nb, nd, 2, GeneRate(0.0), Weight(1.0))
        cd = Connection(nb, ne, 3, GeneRate(0.0), Weight(1.0))
        ce = Connection(nc, nd, 4, GeneRate(0.0), Weight(1.0))
        cf = Connection(nc, ne, 5, GeneRate(0.0), Weight(1.0))
        cg = Connection(ne, nf, 6, GeneRate(0.0), Weight(1.0))
        ch = Connection(nf, ne, 7, GeneRate(0.0), Weight(1.0))
        self.genome = Genome([na, nb, nc, nd, ne],
                             [ca, cb, cc, cd, ce, cf])
        self.recursive_genome = Genome([na, nb, nc, nd, ne, nf],
                                       [ca, cb, cc, cd, ce, cf, cg, ch])
        
    def test_init(self):
        self.assertEqual(self.genome.input_size, 2)
        self.assertEqual(self.genome.output_size, 2)

    def test_call(self):
        self.assertEqual(self.genome([1.0, 1.0]), [3.0, 3.0])
        self.assertEqual(self.genome([1.0, 1.0]), [3.0, 3.0])
        self.assertEqual(self.genome([1.0, 1.0]), [3.0, 3.0])
        self.assertEqual(self.genome([1.0, 1.0]), [3.0, 3.0])

    def test_recursive_call(self):
        self.assertEqual(self.recursive_genome([1.0, 1.0]), [3.0, 3.0])
        self.assertEqual(self.recursive_genome([1.0, 1.0]), [3.0, 6.0])
        self.assertEqual(self.recursive_genome([1.0, 1.0]), [3.0, 9.0])
        self.assertEqual(self.recursive_genome([1.0, 1.0]), [3.0, 12.0])

    def test_copy(self):
        genome = self.genome.copy()
        genome.connections[0].weight.value = 2.0
        genome.connections[0].dominant_gene_rate.value = 2.0
        self.assertEqual(self.genome.connections[0].weight.value, 2.0)
        self.assertEqual(self.genome.connections[0].dominant_gene_rate.value, 2.0)
        self.assertIs(self.genome.connections[0].weight, genome.connections[0].weight)
        self.assertIs(self.genome.connections[0].dominant_gene_rate,
                      genome.connections[0].dominant_gene_rate)

    def test_deepcopy(self):
        genome = self.genome.deepcopy()
        genome.connections[1].weight.value = 2.0
        genome.connections[1].dominant_gene_rate.value = 2.0
        self.assertEqual(self.genome.connections[1].weight.value, 1.0)
        self.assertEqual(self.genome.connections[1].dominant_gene_rate.value, 0.0)
        self.assertIsNot(self.genome.connections[1].weight, genome.connections[1].weight)
        self.assertIsNot(self.genome.connections[1].dominant_gene_rate,
                         genome.connections[1].dominant_gene_rate)

if __name__ == '__main__':
    unittest.main()
