import unittest
from envs.sudoku import KSudoku, CageGenerator


class TestCageGenerator(unittest.TestCase):
    def setUp(self):
        self.sudoku = KSudoku()
        self.grid = self.sudoku.getGrid()
        self.base = self.sudoku.getBase()

    def test_init(self):
        cage_generator = CageGenerator(self.base)
        self.assertIsInstance(cage_generator.cages, list)
        self.assertEqual(len(cage_generator.cages), 0)
        self.assertEqual(len(cage_generator.unusedCells), 81)

    def test_selectStartingCell(self):
        cage_generator = CageGenerator(self.base)
        cell = cage_generator._selectStartingCell()
        self.assertIsInstance(cell, tuple)
        self.assertEqual(len(cell), 2)
        self.assertTrue(cell in cage_generator.unusedCells)
