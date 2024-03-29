import random
from typing import Tuple, List

import numpy as np

SEED = 42
RATE = 0.5
# random.seed(SEED)  # for reproducibility

# Grid = List[List[int]]  # Old definition
Grid = np.ndarray  # New definition


def is_valid(board, row, col, num):
    # Check if the number is not repeated in the current row/column/3x3 square
    for x in range(9):
        if board[row][x] == num or board[x][col] == num or board[row - row % 3 + x // 3][col - col % 3 + x % 3] == num:
            return False
    return True


def solve_sudoku(board):
    empty = find_empty_location(board)
    if not empty:
        return True  # no more empty spaces, puzzle solved
    row, col = empty

    numbers = list(range(1, 10))
    random.shuffle(numbers)  # randomize numbers for unique solutions

    for num in numbers:
        if is_valid(board, row, col, num):
            board[row][col] = num

            if solve_sudoku(board):
                return True

            board[row][col] = 0  # backtrack
    return False


def find_empty_location(board):
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                return i, j
    return None


def generate_sudoku():
    board = np.zeros((9, 9), dtype=int)
    solve_sudoku(board)
    return board


class KSudoku:
    def __init__(self, seed=SEED, mask_rate=RATE):
        self.seed = seed
        self.mask_rate = mask_rate
        self.base: Grid = self._generateCompleteGrid()  # solution
        self.grid: Grid = self._generateGrid(self.base)

    def getGrid(self) -> Grid:
        return self.grid

    def getBase(self) -> Grid:
        return self.base

    def getCell(self, r: int, c: int) -> int:
        return self.base[r][c]

    def _generateGrid(self, baseGrid: Grid) -> Grid:
        return self._mask(baseGrid)

    def _mask(self, grid: Grid) -> Grid:
        """
        Return a masked grid (0 for masked cells, original value for the rest) with the given mask rate.
        :return: the masked grid
        :rtype: Grid
        """
        if self.mask_rate > 1.:
            raise ValueError("mask rate should less or equal to 1")
        if self.seed is not None:
            random.seed(self.seed)

        # Old copying method
        # m_grid = copy.deepcopy(grid)

        # New copying method
        m_grid = grid.copy()
        if self.mask_rate <= 0.:
            return m_grid

        h = len(m_grid)
        w = len(m_grid[0])
        n = h * w
        masked_n = int(n * self.mask_rate)
        mask_array = [True] * masked_n + [False] * (n - masked_n)
        random.shuffle(mask_array)
        for r in range(h):
            for c in range(w):
                if mask_array[r * w + c]:
                    m_grid[r][c] = 0

        return m_grid

    def _generateCompleteGrid(self) -> Grid:
        """
        Generate a complete sudoku grid
        :return: the complete sudoku grid
        :rtype: Grid
        """
        return generate_sudoku()


class Cage:
    def __init__(self, base: Grid):
        self.cells: List[Tuple[int, int]] = []  # list of (row, col)
        self.base = base

    def addCell(self, cell: Tuple[int, int]) -> None:
        self.cells.append(cell)

    def removeCell(self, cell: Tuple[int, int]) -> None:
        self.cells.remove(cell)

    def getCells(self) -> List[Tuple[int, int]]:
        return self.cells

    def getValue(self) -> int:
        """
        Compute the sum of all cells in the cage.
        :return: the sum of all cells in the cage
        :rtype: int
        """
        value = 0
        for cell in self.cells:
            row, col = cell
            value += self.base[row][col]
        return value

    def getSize(self) -> int:
        """
        Get the size of the cage.
        :return: the size of the cage
        :rtype: int
        """
        return len(self.cells)


class CageGenerator:
    """
    Generate cages in a given grid.
    """

    def __init__(self, baseGrid: Grid, seed=SEED):
        self.seed = None
        self.grid = baseGrid
        self.cages: List[Cage] = []
        # initialize the list of unused cells
        # will be updated as cages are generated
        self.unusedCells = [(r, c) for r in range(9) for c in range(9)]

    def generateCages(self) -> List[Cage]:
        """
        Generate cages in the grid.
        :return: the list of cages
        :rtype: List[Cage]
        """

        while self.unusedCells:
            cell = self._selectStartingCell()
            size = self._selectSize(cell)
            cage = self._makeOneCage(cell, size)
            if self._isCageOK(cage):
                self.cages.append(cage)
            else:
                self._backUp(cage)

        return self.cages

    def _selectStartingCell(self) -> Tuple[int, int]:
        """
        Select a cell to start a new cage.
        Top priority for a starting cell is that it is surrounded on 3 or 4 sides,
        otherwise the cell is chosen at random from the list of unused cells
        :return: the selected cell
        :rtype: Tuple[int, int]
        """
        random.seed(self.seed)

        if not self.unusedCells:
            raise ValueError("no unused cells")

        for cell in self.unusedCells:
            numNeighbors, _ = self._getUnusedNeighbors(cell)
            if numNeighbors == 3 or numNeighbors == 4:
                return cell

        return random.choice(self.unusedCells)

    def _getUnusedNeighbors(self, cell: Tuple[int, int]) -> Tuple[int, List[Tuple[int, int]]]:
        """
        Get the unused neighbours of the given cell.
        :param cell: the given cell
        :type cell: Tuple[int, int]
        :return: the number and the list of unused neighbours
        :rtype: Tuple[int, List[Tuple[int, int]]]
        """
        row, col = cell
        unusedNeighbors = []

        # check the 4 neighbours: up, down, left, right
        for deltaRow, deltaCol in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (row + deltaRow, col + deltaCol)

            if self._isValidCell(neighbor) and neighbor in self.unusedCells:
                unusedNeighbors.append(neighbor)

        return len(unusedNeighbors), unusedNeighbors

    @staticmethod
    def _isValidCell(cell: Tuple[int, int]) -> bool:
        """
        Check if the given cell is valid, i.e. within the grid boundaries.
        :param cell: the given cell
        :type cell: Tuple[int, int]
        :return: whether the cell is valid
        :rtype: bool
        """
        row, col = cell
        return 0 <= row <= 8 and 0 <= col <= 8

    def _selectSize(self, cell: Tuple[int, int]) -> int:
        """
        Select the size of a new cage.
        A cell surrounded on 4 sides must become a single-cell cage.
        Choosing a cell surrounded on three sides allows the cage to occupy and
        grow out of tight corners, avoiding an excess of small and single-cell cages.
        The chosen size is a random number from 2 to biggest cage possible, 9.
        :param cell: the starting cell
        :type cell: Tuple[int, int]
        :return: the selected size
        :rtype: int
        """
        random.seed(self.seed)

        numNeighbors, _ = self._getUnusedNeighbors(cell)
        if numNeighbors == 4:
            return 1

        # random size from 2 to the maximum possible size, 9
        return random.randint(2, 5)

    def _makeOneCage(self, cell: Tuple[int, int], size: int) -> Cage:
        """
        Make a cage of the required size, starting from the given cell.
        Keep adding unused neighbouring cells until the required size is reached
        Update the lists of used cells and neighbours as it goes
        :param cell: the starting cell
        :type cell: Tuple[int, int]
        :param size: the required size
        :type size: int
        :return: the cage
        :rtype: Cage
        """
        currentCell = cell

        cage = Cage(self.grid)
        cage.addCell(currentCell)
        self._removeUsedCell(currentCell)
        _, unusedNeighbors = self._getUnusedNeighbors(cell)

        while cage.getSize() < size and unusedNeighbors:
            currentCell = self._selectNextCell(unusedNeighbors)
            cage.addCell(currentCell)
            # update the list of un-used cells
            self._removeUsedCell(currentCell)
            unusedNeighbors.remove(currentCell)
            # update the list of un-used neighbours
            _, newNeighbors = self._getUnusedNeighbors(currentCell)
            unusedNeighbors.extend(newNeighbors)
            # remove duplicates
            unusedNeighbors = list(set(unusedNeighbors))

        return cage

    def _selectNextCell(self, neighbors) -> Tuple[int, int]:
        """
        Select the next cell to add to the cage.
        The next cell is selected from the unused neighbours.
        If a neighbour has 4 neighbours of its own, it is chosen.
        Otherwise, a random neighbour is chosen.
        :param neighbors: the list of unused neighbours
        :type neighbors: List[Tuple[int, int]]
        :return: the next cell
        :rtype: Tuple[int, int]
        """
        random.seed(self.seed)

        for neighbor in neighbors:
            numNeighbours, _ = self._getUnusedNeighbors(neighbor)
            # the cell I just came from is a neighbor but was removed from the list
            # so technically I have 4 neighbors
            if numNeighbours == 3:
                return neighbor

        return random.choice(neighbors)

    def _isCageOK(self, cage: Cage) -> bool:
        """
        Check if the cage is valid.
        :param cage: the cage
        :type cage: Cage
        :return: whether the cage is valid
        :rtype: bool
        """
        # TODO:
        return True

    def _backUp(self, cage: Cage) -> None:
        """
        Back up the cage.
        :param cage: the cage
        :type cage: Cage
        """
        for cell in cage.cells:
            self.unusedCells.append(cell)
        self.cages.remove(cage)

    def _removeUsedCell(self, cell: Tuple[int, int]) -> None:
        """
        Remove the used cell from the unused cells.
        :param cell: the cell
        :type cell: Tuple[int, int]
        """
        self.unusedCells.remove(cell)

    def visualize(self):
        """
        Visualize the generated cages on the grid.
        """
        grid = [[' ' for _ in range(9)] for _ in range(9)]

        # Mark the cells with their corresponding cage values
        for cage in self.cages:
            cage_value = str(cage.getValue())
            for cell in cage.cells:
                row, col = cell
                grid[row][col] = cage_value

        # Print the grid with cages
        print("KS Grid with Cages:")
        for row in grid:
            print(" ".join(f"{value:>3}" for value in row))


def to_array(cages: List[Cage]) -> np.ndarray:
    """
    Visualize the generated cages on the grid and return as an ndarray.
    :return: ndarray representing the grid with cages.
    """
    grid = np.zeros((9, 9), dtype=int)

    for cage in cages:
        cage_value = cage.getValue()
        for cell in cage.cells:
            row, col = cell
            grid[row, col] = cage_value

    return grid


if __name__ == "__main__":
    ks = KSudoku()
    g = ks.getGrid()
    # for r in g:
    #     print(r)
    # print()
    b = ks.getBase()
    # for r in b:
    #     print(r)

    cg = CageGenerator(b)
    # print(cg._selectStartingCell())
    cages = cg.generateCages()
    cg.visualize()
    print(b)
