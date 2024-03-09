import subprocess

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from src.sudoku_env import KillerSudokuEnv

print(f'PyTorch version: {torch.__version__}')
print('*' * 10)
print(f'_CUDA version: ')
subprocess.run(["nvcc", "--version"])
print('*' * 10)
print(f'CUDNN version: {torch.backends.cudnn.version()}')
print(f'Available GPU devices: {torch.cuda.device_count()}')
print(f'Device Name: {torch.cuda.get_device_name()}')


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
model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

# Train the agent and display a progress bar
model.learn(total_timesteps=int(2e5), progress_bar=True)

model.save("killer_sudoku")
# del model  # delete trained model to demonstrate loading

# Load the trained agent
# NOTE: if you have loading issue, you can pass `print_system_info=True`
# to compare the system on which the model was trained vs the current one
# model = DQN.load("dqn_lunar", env=env, print_system_info=True)
model = PPO.load("killer_sudoku", env=env)

# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
