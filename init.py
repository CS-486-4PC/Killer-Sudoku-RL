from random import randint
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import py_environment, tf_py_environment
from tf_agents.networks import q_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import array_spec, BoundedArraySpec
from tf_agents.trajectories import time_step, TimeStep
from tf_agents.utils import common


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
class KillerSudokuEnv(py_environment.PyEnvironment):
    _state: np.ndarray
    solution: np.ndarray
    _action_spec: BoundedArraySpec
    _observation_spec: BoundedArraySpec

    def __init__(self):
        super().__init__()
        self._state = np.zeros((9, 9), dtype=np.int32)

        # Here we define what an action could be.
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(3,), dtype=np.int32, minimum=[0, 0, 1], maximum=[8, 8, 9], name='action')

        # Here we define what the grid should look like.
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=self._state.shape, dtype=np.int32, minimum=0, maximum=9, name='observation')

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self) -> TimeStep:
        self._state = self._generate_random_puzzle()  # Initialize with a new puzzle
        self._episode_ended = False
        return time_step.restart(self._state)

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

    def _step(self, action: [int, int, int]) -> TimeStep:
        # Extract row, column, and number from the action
        row, col, number = action

        # Apply the action to the Sudoku board
        self._state[row][col] = number

        # Check if the game is completed (solved or unsolvable)
        if self._is_game_completed():
            reward = self._calculate_reward()  # Calculate the reward
            return time_step.termination(self._state, reward)
        else:
            # If game is not completed, continue without a reward
            return time_step.transition(self._state, reward=0.0)

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


# Step 2: Create the Neural Network Model
train_env = KillerSudokuEnv()
train_env = tf_py_environment.TFPyEnvironment(train_env)
q_net = q_network.QNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=(100,))

# Step 3: Setup the DQN Agent
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
train_step_counter = tf.Variable(0)
agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter)
agent.initialize()

# Step 4: Training the Agent
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=100000)

driver = dynamic_step_driver.DynamicStepDriver(
    train_env,
    agent.collect_policy,
    observers=[replay_buffer.add_batch],
    num_steps=1)  # collect a step with each driver.run()

# Run the driver to collect experience
for _ in range(10000):  # number of steps to collect
    driver.run()

# Sample a batch of data from the buffer and train the agent
for _ in range(5000):  # number of training iterations
    experience, _ = replay_buffer.get_next(sample_batch_size=64)
    train_loss = agent.train(experience).loss

# Evaluate the agent's performance on a separate Killer Sudoku puzzle
