import os
import subprocess

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from numpy import ndarray
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from sudoku_env import KillerSudokuEnv

print(f'PyTorch version: {torch.__version__}')
print('*' * 10)
print(f'_CUDA version: ')
subprocess.run(["nvcc", "--version"])
print('*' * 10)
print(f'CUDNN version: {torch.backends.cudnn.version()}')
print(f'Available GPU devices: {torch.cuda.device_count()}')
print(f'Device Name: {torch.cuda.get_device_name()}')


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model every set number of steps.
    """

    def __init__(self, check_freq, save_path, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.num_timesteps % self.check_freq == 0:
            self.model.save(self.save_path + str(self.num_timesteps))
        return True


class KSNetwork(nn.Module):
    def __init__(self, observation_space, action_space):
        super(KSNetwork, self).__init__()
        # Define your network architecture here
        # Example architecture:
        num_inputs = observation_space.shape[0] * observation_space.shape[1]
        num_outputs = action_space.n
        self.net = nn.Sequential(
            nn.Linear(num_inputs, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_outputs)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        return self.net(x)


class KSExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=1458):
        super(KSExtractor, self).__init__(observation_space, features_dim)
        self._custom_network = KSNetwork(observation_space, gym.spaces.Discrete(features_dim))

    def forward(self, observations):
        return self._custom_network(observations)


policy_kwargs = dict(
    features_extractor_class=KSExtractor,
    features_extractor_kwargs=dict(features_dim=1458)
)

# Create environment
env = KillerSudokuEnv()

# Instantiate the agent
model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, learning_rate=0.0005)

# Train the agent and display a progress bar
save_dir = "./models/"
os.makedirs(save_dir, exist_ok=True)

# Create your callbacks
save_callback = SaveOnBestTrainingRewardCallback(check_freq=10000, save_path=save_dir)

# Combine them into a CallbackList
callback_list = CallbackList([save_callback])

# Train the agent with the callback list
model.learn(total_timesteps=int(2e3), callback=callback_list)
# # #
model.save("killer_sudoku")
del model  # delete trained model to demonstrate loading

# Load the trained agent
# NOTE: if you have loading issue, you can pass `print_system_info=True`
# to compare the system on which the model was trained vs the current one
# model = DQN.load("dqn_lunar", env=env, print_system_info=True)
model = PPO.load("killer_sudoku", env=env)


# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

def _are_all_cells_filled(obs: ndarray) -> bool:
    # Check if the game is finished (all cells are filled)
    array_1, _ = np.split(obs, 2, axis=2)

    for row in array_1:
        if 0 in row:
            return False
    return True


vec_env = model.get_env()
for i in range(0, 100):
    obs = vec_env.reset()
    print(obs)
    action, _states = model.predict(obs, deterministic=True)
    print(action)

# Enjoy trained agent
# vec_env = model.get_env()
# obs = vec_env.reset()
# while not _are_all_cells_filled(obs):
#     print(obs)

    # action, _states = model.predict(obs, deterministic=True)
    # print(action)
    #
    # obs, rewards, dones, info = vec_env.step(action)
    # vec_env.render("human")
