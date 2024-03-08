import typing

import gymnasium as gym
from gym.spaces import Discrete, Box, MultiDiscrete
import tensorflow as tf
from tensorflow import keras

from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy

from sudoku import Grid, KSudoku

print(tf.__version__)

Action = typing.Tuple[int, int, int]


class KillerSudokuEnv(gym.Env):
    def __init__(self):
        self.size = 9
        self.seed = 42  # larger value means more cells are masked, hence more difficult
        self.difficulty = 0.5
        # Define the action space as 3-dimensional MultiDiscrete for row, column, and number
        # self.action_space = MultiDiscrete([9, 9, 9])
        self.action_space = gym.spaces.Tuple((
            Discrete(self.size),            # row: 0-8
            Discrete(self.size),            # column: 0-8
            Discrete(self.size, start=1)    # value: 1-9
        ))
        # number of actions: row * column * number = 9x9x9
        self.num_actions = self.size ** 3
        # observation_space: 9x9 grid, each cell has a value between 0-9 (0 = empty)
        self.observation_space = Box(low=0, high=9, shape=(self.size, self.size), dtype=int)
        self._setup_board(self.seed, self.difficulty)

    def step(self, action: Action) -> typing.Tuple[Grid, float, bool, dict]:
        row, col, number = action
        self.grid[row][col] = number  # Update the state with the action

        done = self._check_done()
        reward = self._get_reward(row, col, number)
        info = {}

        return self.grid, reward, done, info

    def reset(self, seed=None, options=None):
        self._setup_board(self.seed, self.difficulty)
        return self.grid

    def render(self, mode='human'):
        for row in self.grid:
            print(row)

    def seed(self, seed):
        self.seed = seed

    def _setup_board(self, seed, difficulty):
        ks = KSudoku(seed, difficulty)
        self.grid = ks.getGrid()
        self.base = ks.getBase()

    def _check_done(self) -> bool:
        # Check if the game is finished (all cells are filled)
        for row in self.grid:
            if 0 in row:
                return False

    def _get_reward(self, row: int, col: int, number: int) -> float:
        # Define reward logic for the game
        return 0.0  # TODO: Placeholder reward, implement game-specific logic


def main():
    # Create the environment
    env = KillerSudokuEnv()

    # Neural Network model for Deep Q Learning
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=(1, 9, 9)))  # Reshape for compatibility
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(env.num_actions, activation='linear'))  # Output layer

    # Configure and compile the DQN agent
    memory = SequentialMemory(limit=50000, window_length=1)
    policy = EpsGreedyQPolicy()
    dqn = DQNAgent(model=model, nb_actions=env.num_actions, memory=memory, nb_steps_warmup=10,
                   target_model_update=1e-2, policy=policy)
    dqn.compile(keras.optimizers.Adam(lr=1e-3), metrics=['mae'])

    # Train the agent
    dqn.fit(env, nb_steps=50000, visualize=False, verbose=2)

    # Test the agent
    dqn.test(env, nb_episodes=5, visualize=False)


if __name__ == '__main__':
    main()

