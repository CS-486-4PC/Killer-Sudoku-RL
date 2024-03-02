from typing import Tuple, List
import numpy as np
import logging
import copy
import random

Grid = List[List[int]]
SEED = 42
RATE = 0.5

class KSudoku:
    def __init__(self, seed=SEED, mask_rate=RATE):
        self.seed = seed
        self.mask_rate = mask_rate
        self.base = self._generateCompleteGrid()
        self.grid = self._generateGrid(self.base)

    def getGrid(self) -> Grid:
        return self.grid

    def getBase(self) -> Grid:
        return self.base

    def _generateGrid(self, base: Grid) -> Grid:
        return self._mask(base)

    def _mask(self, grid) -> Grid:
        """
        Return a masked grid (0 for masked cells, original value for the rest) with the given mask rate.
        :return: the masked grid
        :rtype: Grid
        """
        if self.mask_rate > 1.:
            raise ValueError("mask rate should less or equal to 1")
        if self.seed is not None:
            random.seed(self.seed)

        m_grid = copy.deepcopy(grid)
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
        attempt = 1
        if self.seed is not None:
            np.random.seed(self.seed)

        while True:
            n = 9
            g = np.zeros((n, n), np.uint)
            rg = np.arange(1, n + 1)
            g[0, :] = np.random.choice(rg, n, replace=False)

            try:
                for r in range(1, n):
                    for c in range(n):
                        col_rest = np.setdiff1d(rg, g[:r, c])
                        row_rest = np.setdiff1d(rg, g[r, :c])
                        avb1 = np.intersect1d(col_rest, row_rest)
                        sub_r, sub_c = r // 3, c // 3
                        avb2 = np.setdiff1d(np.arange(0, n + 1),
                                            g[sub_r * 3:(sub_r + 1) * 3, sub_c * 3:(sub_c + 1) * 3].ravel())
                        avb = np.intersect1d(avb1, avb2)
                        g[r, c] = np.random.choice(avb, size=1)
                break
            except ValueError:
                attempt += 1

        g_list: Grid = g.tolist()
        if attempt > 1:
            logging.debug(f"generate by np_union attempt {attempt}")

        return g_list


if __name__ == "__main__":
    ks = KSudoku()
    grid = ks.getGrid()
    for row in grid:
        print(row)

    print()

    base = ks.getBase()
    for row in base:
        print(row)
