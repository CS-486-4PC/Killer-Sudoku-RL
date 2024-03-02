import gym
from gym.spaces import Discrete, Box
import random

from sudoku import Grid, generateGrid


N_BOXES = 9  # 9 3x3 boxes
SIZE = 9  # 9x9 grid
SEED = 42

class KillerSudokuEnv(gym.Env):
    def __init__(self, size=SIZE):
        self.size = size
        # action_space: 9 actions, 1-9
        self.action_space = Discrete(9, start=1)
        # observation_space: 9x9 grid, each cell has a value between 0-9 (0 = empty)
        self.observation_space = Box(low=0, high=9, shape=(SIZE, SIZE), dtype=int)
        self._setup_board(SEED)

    def step(self, action: int):
        pass

    def reset(self, seed=None, options=None):
        self._setup_board(seed)
        return self.grid

    def render(self, mode='human'):
        pass

    def _setup_board(self, seed):
        # TODO: difficulty (mask rate) can be a parameter
        random.seed(seed)
        mask_rate = random.uniform(0.1, 0.7)
        self.grid, self.base = generateGrid(mask_rate, seed)
