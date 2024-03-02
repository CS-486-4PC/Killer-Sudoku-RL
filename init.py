from typing import Tuple

import gym
import numpy as np
import tensorflow as tf
from gym import spaces

print(tf.__version__)

from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy
from tensorflow import keras

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


class KillerSudokuEnv(gym.Env):
    """
    Custom environment for Killer Sudoku that follows the gym interface.
    """

    def __init__(self):
        super(KillerSudokuEnv, self).__init__()
        # Define the action space as 3-dimensional MultiDiscrete for row, column, and number
        self.action_space = spaces.MultiDiscrete([9, 9, 9])
        # Define the observation space as a 9x9 grid with values ranging from 0 to 9
        self.observation_space = spaces.Box(low=0, high=9, shape=(9, 9), dtype=np.int32)
        # Initialize the state as a 9x9 grid of zeros
        self.state = np.zeros((9, 9), dtype=np.int32)

    def step(self, action: Tuple[int, int, int]) -> Tuple[np.ndarray, float, bool, dict]:
        row, col, number = action
        self.state[row, col] = number  # Update the state with the action

        done = self._check_done()
        reward = self._get_reward(row, col, number)
        info = {}

        return self.state, reward, done, info

    def reset(self) -> np.ndarray:
        self.state = np.zeros((9, 9), dtype=np.int32)
        self._generate_random_puzzle()
        return self.state

    def _generate_random_puzzle(self):
        # For simplicity, add a few random numbers to the board
        for _ in range(np.random.randint(10, 20)):  # Randomly decide how many numbers to place
            while True:
                row, col, num = np.random.randint(0, 9), np.random.randint(0, 9), np.random.randint(1, 10)
                if self._can_place_number(row, col, num):
                    self.state[row, col] = num
                    break

    def _can_place_number(self, row: int, col: int, num: int) -> bool:
        # Check if a number can be placed without violating Sudoku rules
        if self.state[row, col] != 0:
            return False  # Cell already has a number

        # Check row and column
        if num in self.state[row, :] or num in self.state[:, col]:
            return False

        # Check 3x3 square
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        if num in self.state[start_row:start_row + 3, start_col:start_col + 3]:
            return False

        return True

    def render(self, mode='human', close=False):
        # Render the current state of the environment
        if mode == 'human':
            print(self.state)  # For simplicity, we print the state

    def _check_done(self) -> bool:
        # Check if the game is finished (all cells are filled)
        return np.all(self.state != 0)

    def _get_reward(self, row: int, col: int, number: int) -> float:
        # Define reward logic for the game
        return 0.0  # Placeholder reward, implement game-specific logic


# class DQNAgent:
#     """
#     DQN Agent that learns to solve Killer Sudoku.
#     """
#
#     def __init__(self, env: KillerSudokuEnv):
#         self.env = env
#         self.model = self._create_model()
#         self.memory = deque(maxlen=2000)  # Experience replay buffer
#         self.gamma = 0.95  # Discount factor
#         self.epsilon = 1.0  # Exploration rate
#         self.epsilon_min = 0.01
#         self.epsilon_decay = 0.995
#         self.learning_rate = 0.001
#
#     def _create_model(self) -> keras.models.Sequential:
#         # Create a neural network for DQN
#         model = keras.Sequential([
#             keras.layers.Flatten(input_shape=(9, 9)),
#             keras.layers.Dense(128, activation='relu'),
#             keras.layers.Dense(128, activation='relu'),
#             keras.layers.Dense(81, activation='linear')  # Output for each cell
#         ])
#         model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
#         return model
#
#     def remember(self, state: np.ndarray, action: Tuple[int, int, int], reward: float, next_state: np.ndarray,
#                  done: bool):
#         # Store experience in the replay buffer
#         self.memory.append((state, action, reward, next_state, done))
#
#     def act(self, state: np.ndarray) -> Tuple[int, int, int]:
#         # Epsilon-greedy action selection
#         if np.random.rand() <= self.epsilon:
#             return self.env.action_space.sample()
#         act_values = self.model.predict(state)
#         return np.unravel_index(np.argmax(act_values), (9, 9, 9))  # Convert flat index to 3D action
#
#     def replay(self, batch_size: int):
#         # Train the model using experiences from the memory
#         if len(self.memory) < batch_size:
#             return
#
#         minibatch = random.sample(self.memory, batch_size)
#         for state, action, reward, next_state, done in minibatch:
#             target = reward if done else reward + self.gamma * np.amax(self.model.predict(next_state)[0])
#             target_f = self.model.predict(state)
#             target_f[0][np.ravel_multi_index(action, (9, 9, 9))] = target
#             self.model.fit(state, target_f, epochs=1, verbose=0)
#
#         # Update exploration rate
#         if self.epsilon > self.epsilon_min:
#             self.epsilon *= self.epsilon_decay
#
#     def learn(self, episodes: int, batch_size: int):
#         for e in range(episodes):
#             state = self.env.reset()
#             state = np.reshape(state, [1, 9, 9])
#             for _ in range(500):  # Maximum steps per episode
#                 action = self.act(state)
#                 next_state, reward, done, _ = self.env.step(action)
#                 next_state = np.reshape(next_state, [1, 9, 9])
#                 self.remember(state, action, reward, next_state, done)
#                 state = next_state
#                 if done or len(self.memory) > batch_size:
#                     self.replay(batch_size)
#                     if done:
#                         break


def main():
    # Create the environment
    env = KillerSudokuEnv()

    # Neural Network model for Deep Q Learning
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=(1, 9, 9)))  # Reshape for compatibility
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(env.action_space.n, activation='linear'))  # Output layer

    # Configure and compile the DQN agent
    memory = SequentialMemory(limit=50000, window_length=1)
    policy = EpsGreedyQPolicy()
    dqn = DQNAgent(model=model, nb_actions=env.action_space.n, memory=memory, nb_steps_warmup=10,
                   target_model_update=1e-2, policy=policy)
    dqn.compile(keras.optimizers.Adam(lr=1e-3), metrics=['mae'])

    # Train the agent
    dqn.fit(env, nb_steps=50000, visualize=False, verbose=2)

    # Test the agent
    dqn.test(env, nb_episodes=5, visualize=False)


if __name__ == '__main__':
    main()
