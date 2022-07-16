"""
Snake RL
Cameron Wolfe 7/15/2022
"""
import gym
from gym import logger, spaces
import numpy as np
from collections import namedtuple
from tensorboard import SummaryWriter
import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim

BOARD_WIDTH = 20
BOARD_HEIGHT = 16
FIRST_LAYER_SIZE = 128
SECOND_LAYER_SIZE = 64
BATCH_SIZE = 100
CUTOFF = .7

class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.linear(hidden_size, n_actions)
        )
    
    def forward(self, x):
        return self.net(x)

class SnakeEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    def __init__(self, width, height):
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high = 1, shape=[height, width], dtype=np.float32)
        
        self.width = width
        self.height = height

        self.fruit_pos = None
        self.snake_pos = None
        self.length = None
        
        self.steps_beyond_terminated = None



Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('Step', field_names=['observation', 'action'])

def iterate_batches(env, net, batch_size):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()
    sm = nn.Softmax(dim=1)

    while True:
        obs_t = torch.floatTensor([obs])
        act_probs_t = sm(net(obs_t))
        act_probs = act_probs_t.data.numpy()[0]
        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, is_done, _ = env.step(action)
        episode_reward += reward
        episode_steps.append(EpisodeStep(observation=obs, action=action))
        if is_done:
            batch.append(Episode(reward=episode_reward, steps=episode_steps))
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs

def filter_batch(batch, percentile):
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    train_obs = []
    train_act = []

    for example in batch:
        if example.reward < reward_bound:
            continue
        train_obs.extend(map(lambda step: step.observation, example.steps))
        train_act.extend(map(lambda step: step.action, example.steps))
    
    train_obs_t = torch.FloatTensor(train_obs)
    train_act_t = torch.LongTensor(train_act)
    return train_obs_t, train_act_t, reward_bound, reward_mean

if __name__ == "__main__":
    boop = 0

