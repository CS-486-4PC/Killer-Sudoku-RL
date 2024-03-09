import typing

import gymnasium as gym
import numpy as np
import tensorflow as tf
from gymnasium.spaces import Box, MultiDiscrete

from sudoku import Grid, KSudoku

print(tf.__version__)

Action = typing.Tuple[int, int, int]


class KillerSudokuEnv(gym.Env):
    def __init__(self):
        self.size = 9
        self.seed = 42  # larger value means more cells are masked, hence more difficult
        self.difficulty = 0.5
        # Define the action space as 3-dimensional MultiDiscrete for row, column, and number
        self.action_space = MultiDiscrete([9, 9, 9])

        # number of actions: row * column * number = 9x9x9
        self.num_actions = self.size ** 3
        # observation_space: 9x9 grid, each cell has a value between 0-9 (0 = empty)
        self.observation_space = Box(low=0, high=9, shape=(self.size, self.size * 2), dtype=int)
        self._setup_board(self.seed, self.difficulty)

    def step(self, action: Action) -> typing.Tuple[Grid, float, bool, dict]:
        row, col, number = action

        # self.grid[row][col] = number  # Update the state with the action
        self.grid[row, col] = number  # NumPy indexing

        if self._is_game_completed():
            reward = self._calculate_reward()
            done = True
        else:
            reward = 0
            done = False

        return self.grid, reward, done, {}

    def reset(self, seed=None, options=None) -> np.ndarray:
        self._setup_board(self.seed, self.difficulty)
        return self.grid

    def render(self, mode='human'):
        for row in self.grid:
            print(' '.join(map(str, row)))

    def seed(self, seed):
        self.seed = seed

    def _setup_board(self, seed, difficulty):
        ks = KSudoku(seed, difficulty)
        self.grid = ks.getGrid()  # state
        self.base = ks.getBase()

    def _are_all_cells_filled(self) -> bool:
        # Check if the game is finished (all cells are filled)
        for row in self.grid:
            if 0 in row:
                return False
        return True

    def _is_puzzle_valid(self) -> bool:
        def is_unit_valid(unit: np.ndarray) -> bool:
            # Filter out zeros and check for duplicates
            filtered_unit = unit[unit > 0]
            return len(np.unique(filtered_unit)) == len(filtered_unit)

        # Check rows
        for row in range(9):
            if not is_unit_valid(self.grid[row, :]):
                return False

        # Check columns
        for col in range(9):
            if not is_unit_valid(self.grid[:, col]):
                return False

        # Check 3x3 squares
        for row in range(0, 9, 3):
            for col in range(0, 9, 3):
                square = self.grid[row:row + 3, col:col + 3].flatten()
                if not is_unit_valid(square):
                    return False

        return True

    def _is_game_completed(self) -> bool:
        return self._are_all_cells_filled() and self._is_puzzle_valid()

    def _calculate_reward(self):
        # Calculate the number of correctly filled cells
        return self._count_correct_cells()

    def _count_correct_cells(self) -> int:
        correct_count = 0
        for row in range(9):
            for col in range(9):
                if self.grid[row][col] == self.base[row][col]:
                    correct_count += 1
        return correct_count
