from typing import Tuple, List
import numpy as np
import logging
import copy
import random

Grid = List[List[int]]
SEED = 42

def mask(grid: Grid, rate: float = 0.5, seed=None) -> Grid:
    """
    Return a masked grid (0 for masked cells, original value for the rest) with the given mask rate.
    :param grid: the original grid
    :type grid: Grid
    :param rate: the proportion of cells to be masked
    :type rate: float
    :param seed: the random seed
    :type seed: int
    :return: the masked grid
    :rtype: Grid
    """
    if rate > 1.:
        raise ValueError("mask rate should less or equal to 1")
    if seed is not None:
        random.seed(seed)

    grid = copy.deepcopy(grid)
    if rate <= 0.:
        return grid

    h = len(grid)
    w = len(grid[0])
    n = h * w
    masked_n = int(n * rate)
    mask_array = [True] * masked_n + [False] * (n - masked_n)
    random.shuffle(mask_array)
    for r in range(h):
        for c in range(w):
            if mask_array[r * w + c]:
                grid[r][c] = 0

    return grid


def generateGrid(mask_rate=0.5, seed=None) -> Tuple[Grid, Grid]:
    """
    Generate a new sudoku grid with the given mask rate.
    :param mask_rate: the proportion of cells to be masked
    :type mask_rate: float
    :param seed: the random seed
    :type seed: int
    :return: the masked (unsolved) grid and the original (solved) grid
    :rtype: Tuple[Grid, Grid]
    """
    attempt = 1
    if seed is not None:
        np.random.seed(seed)

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

    return mask(g_list, mask_rate), g_list
