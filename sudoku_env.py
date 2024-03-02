import gym
from gym.spaces import Discrete, Box, Tuple
import random
import typing

from sudoku import KSudoku

Action = typing.Tuple[int, int, int]


class KillerSudokuEnv(gym.Env):
    def __init__(self):
        self.size = 9
        self.seed = 42  # larger value means more cells are masked, hence more difficult
        self.difficulty = 0.5
        # action_space: (row, column, value)
        self.action_space = gym.spaces.Tuple((
            Discrete(self.size),            # row: 0-8
            Discrete(self.size),            # column: 0-8
            Discrete(self.size, start=1)    # value: 1-9
        ))
        # observation_space: 9x9 grid, each cell has a value between 0-9 (0 = empty)
        self.observation_space = Box(low=0, high=9, shape=(self.size, self.size), dtype=int)
        self._setup_board(self.seed, self.difficulty)

    def step(self, action: Action):
        pass

    def reset(self, seed=None, options=None):
        self._setup_board(self.seed, self.difficulty)
        return self.grid

    def render(self, mode='human'):
        pass

    def _setup_board(self, seed, difficulty):
        ks = KSudoku(seed, difficulty)
        self.grid = ks.getGrid()
        self.base = ks.getBase()


if __name__ == "__main__":
    env = KillerSudokuEnv()
    env.reset()
    assert env.grid is not None
    for row in env.grid:
        print(row)

