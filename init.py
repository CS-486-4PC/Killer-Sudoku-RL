from random import randint
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy


def is_valid(board: np.ndarray, row: int, col: int, num: int) -> bool:
    # Check if 'num' is not already placed in current row, current column and current 3x3 box
    box_row = row - row % 3
    box_col = col - col % 3

    if num in board[row]:
        return False
    if num in [board[i][col] for i in range(9)]:
        return False
    if num in [board[i][j] for i in range(box_row, box_row + 3) for j in range(box_col, box_col + 3)]:
        return False

    return True


def solve_sudoku(board: np.ndarray) -> bool:
    empty_cell = find_empty_location(board)
    if not empty_cell:
        return True  # Puzzle solved
    row, col = empty_cell

    for num in range(1, 10):
        if is_valid(board, row, col, num):
            board[row][col] = num
            if solve_sudoku(board):
                return True
            board[row][col] = 0  # Backtrack

    return False  # Triggers backtracking


def find_empty_location(board: np.ndarray) -> Optional[Tuple[int, int]]:
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                return i, j
    return None


def generate_sudoku():
    board = np.zeros((9, 9), dtype=np.int32)

    # Fill the diagonal with SRN x SRN matrices
    fill_diagonal(board)

    # Fill remaining blocks
    solve_sudoku(board)

    # Remove Randomly 30-60 numbers to create puzzles
    remove_numbers(board)

    return board


def fill_diagonal(board: np.ndarray) -> None:
    for i in range(0, 9, 3):
        fill_box(board, i, i)


def fill_box(board: np.ndarray, row: int, col: int) -> None:
    num = 0
    for i in range(3):
        for j in range(3):
            while True:
                num = randint(1, 9)
                if num not in [board[row + x][col + y] for x in range(3) for y in range(3)]:
                    board[row + i][col + j] = num
                    break


def remove_numbers(board: np.ndarray) -> None:
    count = randint(30, 60)
    while count != 0:
        i = randint(0, 8)
        j = randint(0, 8)
        if board[i][j] != 0:
            count -= 1
            board[i][j] = 0


# Step 1: Define the Killer Sudoku Environment
class KillerSudokuEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(KillerSudokuEnv, self).__init__()
        self.action_space = spaces.Discrete(729)  # 9*9*9 possible actions
        self.observation_space = spaces.Box(low=0, high=9, shape=(9, 9), dtype=np.int32)
        self._state = None
        self.solution = None

    def seed(self, seed=None):
        # If you are using random number generators in your environment,
        # set their seed here. For example:
        # self.np_random, _ = seeding.np_random(seed)

        # If your environment does not use a random generator, just pass
        pass

    def reset(self, **kwargs):
        # If you are using the 'seed' in your environment, you can extract and use it here:
        # seed = kwargs.get('seed', None)
        # if seed is not None:
        #     # Set your random number generator's seed
        #     ...

        self._state = self._generate_random_puzzle()
        return self._state

    def step(self, action):
        row = action // 81
        col = (action % 81) // 9
        number = (action % 81) % 9 + 1

        self._state[row][col] = number

        if self._is_game_completed():
            reward = self._calculate_reward()
            done = True
        else:
            reward = 0
            done = False

        return self._state, reward, done, {}

    def render(self, mode='human', close=False):
        if close:
            return

    def _generate_random_puzzle(self) -> np.ndarray:
        generated = generate_sudoku()
        self.solution = generated
        return generated

    def _is_game_completed(self):
        return self._are_all_cells_filled() and self._is_puzzle_valid()

    def _are_all_cells_filled(self):
        return not np.any(self._state == 0)

    def _is_puzzle_valid(self):
        for i in range(9):
            if not self._is_unique(self._state[i, :]):  # Check rows
                return False
            if not self._is_unique(self._state[:, i]):  # Check columns
                return False

        for i in range(0, 9, 3):
            for j in range(0, 9, 3):
                if not self._is_unique(self._state[i:i + 3, j:j + 3].flatten()):  # Check subgrids
                    return False

        return True

    def _is_unique(self, arr):
        return len(arr) == len(np.unique(arr))

    def _calculate_reward(self) -> int:
        # Calculate the number of correctly filled cells
        return self._count_correct_cells()

    def _count_correct_cells(self) -> int:
        correct_count = 0
        for row in range(9):
            for col in range(9):
                if self._state[row, col] == self.solution[row, col]:
                    correct_count += 1
        return correct_count


# Define the device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the environment and the agent
env = KillerSudokuEnv()
env = make_vec_env(lambda: env, n_envs=1)

# Create and train the DQN model
model = DQN("MlpPolicy", env, verbose=1, buffer_size=100000, learning_rate=1e-3, batch_size=64)
model.learn(total_timesteps=int(1e5))

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

# Save the model
model.save("killer_sudoku_dqn_model")
