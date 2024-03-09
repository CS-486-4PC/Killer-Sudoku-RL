import unittest
from unittest.mock import patch

from src.sudoku import KSudoku, CageGenerator, Cage

""" seed = 42
[4, 8, 5, 2, 7, 6, 3, 9, 1]
[7, 6, 9, 3, 4, 1, 8, 2, 5]
[2, 3, 1, 9, 5, 8, 4, 7, 6]
[9, 5, 7, 8, 3, 2, 1, 6, 4]
[8, 2, 6, 1, 9, 4, 5, 3, 7]
[1, 4, 3, 7, 6, 5, 9, 8, 2]
[5, 7, 2, 4, 8, 3, 6, 1, 9]
[6, 9, 8, 5, 1, 7, 2, 4, 3]
[3, 1, 4, 6, 2, 9, 7, 5, 8]
"""

class TestKSudoku(unittest.TestCase):
    def setUp(self):
        self.sudoku = KSudoku(seed=42)
        self.grid = self.sudoku.getGrid()
        self.base = self.sudoku.getBase()

    def test_getCell(self):
        cell = self.sudoku.getCell(0, 0)
        self.assertIsInstance(cell, int)
        self.assertEqual(cell, 4)
        cell = self.sudoku.getCell(8, 8)
        self.assertIsInstance(cell, int)
        self.assertEqual(cell, 8)


class TestCageGenerator(unittest.TestCase):
    def setUp(self):
        self.sudoku = KSudoku()
        self.grid = self.sudoku.getGrid()
        self.base = self.sudoku.getBase()
        self.cg = CageGenerator(self.base)

    def test_init(self):
        self.assertIsInstance(self.cg.cages, list)
        self.assertEqual(len(self.cg.cages), 0)
        self.assertEqual(len(self.cg.unusedCells), 81)

    def test_selectStartingCell(self):
        cell = self.cg._selectStartingCell()
        self.assertIsInstance(cell, tuple)
        self.assertEqual(len(cell), 2)
        self.assertTrue(cell in self.cg.unusedCells)

    @patch('envs.sudoku.CageGenerator._getUnusedNeighbors')
    def test_selectSize_4neighbors(self, mock_getUnusedNeighbors):
        cell = (1, 1)
        mock_getUnusedNeighbors.return_value = (4, [(0, 1), (1, 0), (1, 2), (2, 1)])
        size = self.cg._selectSize(cell)
        self.assertEqual(size, 1)

    def test_getUnusedNeighbors(self):
        size, neighbors = self.cg._getUnusedNeighbors((0, 0))
        self.assertEqual(size, 2)
        self.assertCountEqual(neighbors, [(0, 1), (1, 0)])

    def test_getUnusedNeighbors_1(self):
        size, neighbors = self.cg._getUnusedNeighbors((0, 1))
        self.assertEqual(size, 3)
        self.assertCountEqual(neighbors, [(0, 0), (1, 1), (0, 2)])

    def test_getUnusedNeighbors_2(self):
        size, neighbors = self.cg._getUnusedNeighbors((1, 1))
        self.assertEqual(size, 4)
        self.assertCountEqual(neighbors, [(0, 1), (1, 0), (1, 2), (2, 1)])

    def test_makeOneCage(self):
        cell = (0, 0)
        size = 3
        cage: Cage = self.cg._makeOneCage(cell, size)
        self.assertCountEqual(cage.cells, [(0, 0), (0, 1), (1, 0)])
        self.assertEqual(cage.getSize(), 3)
