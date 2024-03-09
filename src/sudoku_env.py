import typing

import gymnasium as gym
import numpy as np
import tensorflow as tf
from gymnasium.spaces import Box, MultiDiscrete

from sudoku import KSudoku, CageGenerator, to_array

print(tf.__version__)

Action = typing.Tuple[int, int, int]


class KillerSudokuEnv(gym.Env):
    def __init__(self):
        self.render_mode = "human"

        self.size = 9
        self.seed = 42  # larger value means more cells are masked, hence more difficult
        self.difficulty = 0.5
        # Define the action space as 3-dimensional MultiDiscrete for row, column, and number
        self.action_space = MultiDiscrete([9, 9, 9])

        # number of actions: row * column * number = 9x9x9
        self.num_actions = self.size ** 3
        # observation_space: 9x9 grid, each cell has a value between 0-9 (0 = empty)
        self.observation_space = Box(low=0, high=9, shape=(self.size, self.size * 2), dtype=np.int32)

        self._setup_board(self.seed, self.difficulty)

    def step(self, action: Action) -> tuple[np.ndarray, float, bool, bool, dict]:
        row, col, number = action
        correct_number = self.base[row, col]  # The correct number in the solution

        # Check if the action places the correct number
        if number == correct_number:
            self.grid[row, col] = number  # Update the state with the correct action
            reward = 1  # Positive reward for correct placement
        else:
            reward = -1  # Negative reward for incorrect placement

        terminated = False
        truncated = False
        info = {}  # Additional information, empty for now

        if self._is_game_completed():
            terminated = True
        else:
            reward += 0  # No additional reward if the game is not completed

        observation = np.concatenate((self.grid, self.cages), axis=1)  # Combine grid and cages

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self._setup_board(self.seed, self.difficulty)
        observation = np.concatenate((self.grid, self.cages), axis=1)
        return observation, {}

    def render(self, mode='human'):
        if mode != 'human':
            raise NotImplementedError("Only 'human' render mode is supported.")

        # Create a visual representation of the grid and cages
        grid_str = ""
        for r in range(self.size):
            if r % 3 == 0 and r != 0:  # Add a horizontal divider every 3 rows
                grid_str += "-" * 21 + "\n"

            for c in range(self.size):
                if c % 3 == 0 and c != 0:  # Add a vertical divider every 3 columns
                    grid_str += "| "

                cell_value = self.grid[r, c]
                cell_str = str(cell_value) if cell_value != 0 else '.'
                grid_str += cell_str + " "

            grid_str += "\n"

        # Print cage information every few lines if needed
        for r in range(0, self.size, 3):
            cage_str = ""
            for c in range(self.size):
                cage_index = r // 3 * 3 + c // 3
                sum_of_cage = sum(self.cages[cage_index])
                cage_str += f"Cage {cage_index}: Sum = {sum_of_cage}\t"
            grid_str += cage_str + "\n"

        print(grid_str)

    def seed(self, seed):
        self.seed = seed

    def _setup_board(self, seed, difficulty):
        ks = KSudoku(seed, difficulty)
        self.grid = ks.getGrid()  # state
        self.base = ks.getBase()

        cg = CageGenerator(self.base)
        # print(cg._selectStartingCell())
        self.cages = to_array(cg.generateCages())

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
