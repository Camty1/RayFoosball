import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Normal
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-b', action="store_false", help="Flag for running on Babylon")
args = parser.parse_args()

device = torch.device("cpu")

if args.b:

    if (torch.cuda.is_available()):
        device = torch.device('cuda')
        torch.cuda.empty_cache()

    print("Device: " + str(torch.cuda.get_device_name(device)))


class Buffer():
    
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lambda_GAE=0.95):
        self.observations  = np.zeros((size, obs_dim), dtype=np.float32)
        self.actions       = np.zeros((size, act_dim), dtype=np.float32)
        
        self.rewards       = np.zeros(size, dtype=np.float32)
        self.values        = np.zeros(size, dtype=np.float32)
        self.log_probs     = np.zeros(size, dtype=np.float32)
        self.advantages    = np.zeros(size, dtype=np.float32)
        self.rewards_to_go = np.zeros(size, dtype=np.float32)
        self.gamma = gamma
        self.lambda_GAE = lambda_GAE

        self.ptr = 0
        self.traj_start = 0
        self.size = size

    def store(self, observation, action, reward, value, log_prob):

        assert self.ptr < self.size

        self.observations[self.ptr] = observation
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.ptr += 1

    def GAE(self, last_value=0.0):
        indices = slice(self.traj_start, self.ptr)
        rewards = np.append(self.rewards[indices], last_value)
        values = np.append(self.values[indices], last_value)
        td = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        advantages = np.zeros_like(self.advantages[indices])
        rewards_to_go = np.zeros_like(self.rewards_to_go[indices])
        
        for i in reversed(range(len(td))):
            if i == len(td)-1:
                advantages[i] = td[i]
                rewards_to_go[i] = rewards[-2] + self.gamma * rewards[-1]
            else:
                advantages[i] = td[i] + self.gamma * self.lambda_GAE * advantages[i+1]
                rewards_to_go[i] = rewards[i] + self.gamma * rewards_to_go[i+1]

        self.advantages[indices] = advantages
        self.rewards_to_go[indices] = rewards_to_go
        self.traj_start = self.ptr

    def get(self):
        assert self.ptr == self.size

        self.ptr, self.traj_start = 0, 0
        advantage_mean = np.mean(self.advantages)
        advantage_std = np.std(self.advantages)
        

        self.advantages = (self.advantages - advantage_mean) / advantage_std
        
        data = {
            "observations": torch.as_tensor(self.observations, dtype=torch.float32),
            "actions": torch.as_tensor(self.actions, dtype=torch.float32),
            "rewards": torch.as_tensor(self.rewards, dtype=torch.float32),
            "values": torch.as_tensor(self.values, dtype=torch.float32),
            "log_probs": torch.as_tensor(self.log_probs, dtype=torch.float32),
            "advantages": torch.as_tensor(self.advantages, dtype=torch.float32),
            "rewards_to_go": torch.as_tensor(self.rewards_to_go, dtype=torch.float32),
        }

        return data


class Actor(nn.Module):

    def __init__(self,
                 num_observations,
                 num_actions,
                 log_std_init=-0.5,
                 activation=nn.Tanh,
                 action_activation=nn.Identity):
        super().__init__()
        log_std = log_std_init * np.ones(num_actions, dtype=np.float32) 
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

        self.actor = nn.Sequential(
            nn.Linear(num_observations, 64),
            activation(),
            nn.Linear(64, 64),
            activation(),
            nn.Linear(64, num_actions),
            action_activation()
        )

    def forward(self, obs, action=None):

        dist = self._distribution(obs)
        log_prob = None
        if action is not None:
            log_prob = self._log_prob_from_dist(dist, action)
        return dist, log_prob

    def _distribution(self, obs):
        
        mu = self.actor(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_dist(self, dist, action):
        return dist.log_prob(action).sum(axis=-1)

class Critic(nn.Module):

    def __init__(self, num_observations, activation=nn.Tanh):
        super().__init__()    
        self.critic = nn.Sequential(
            nn.Linear(num_observations, 64),
            activation(),
            nn.Linear(64, 64),
            activation(),
            nn.Linear(64, 1),
            nn.Identity()
        )
    
    def forward(self, obs):
        return torch.squeeze(self.critic(obs), -1)

class ActorCritic(nn.Module):

    def __init__(self,
                 num_observations,
                 num_actions,
                 log_std_init=-0.5,
                 activation=nn.Tanh,
                 action_activation=nn.Identity):
        super().__init__()

        self.actor = Actor(num_observations, num_actions, log_std_init, activation, action_activation)

        self.critic = Critic(num_observations, activation)

    def step(self, obs):
        with torch.no_grad():
            dist = self.actor._distribution(obs)
            action = dist.sample()
            log_prob = self.actor._log_prob_from_dist(dist, action)
            value = self.critic(obs)

        return action.numpy(), value.numpy(), log_prob.numpy()

    def act(self, obs):
        return self.step(obs)[0]


class PPO():

    def __init__(self,
                 num_observations,
                 num_actions,
                 timesteps_per_epoch,
                 actor_lr=.0003,
                 critic_lr=.001,
                 k_epochs=80,
                 gamma=0.99,
                 lambda_GAE=0.95,
                 epsilon=0.2,
                 num_teams=1,
                 log_std_init=-0.5,
                 activation=nn.Tanh,
                 action_activation=nn.Identity):

        self.buffer = Buffer(num_observations, num_actions, timesteps_per_epoch, gamma, lambda_GAE)
        self.policy = ActorCritic(num_observations, num_actions, log_std_init, activation, action_activation)
        self.k_epochs = k_epochs
        self.epsilon = epsilon

        self.actor_optimizer = Adam(self.policy.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.policy.critic.parameters(), lr=critic_lr)

    def compute_loss_actor(self, data):
        obs, act, advantages, old_log_probs = data["observations"], data["actions"], data["advantages"], data["log_probs"]

        dist, log_probs = self.policy.actor(obs, act)
        ratio = torch.exp(log_probs - old_log_probs)

        clip_advantage = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * advantages
        loss_actor = -(torch.min(ratio * advantages, clip_advantage)).mean()
        
        approx_kl = (old_log_probs - log_probs).mean().item()
        entropy = dist.entropy().mean().item()

        extra_info = {"kl": approx_kl, "entropy": entropy}

        return loss_actor, extra_info

    def compute_loss_critic(self, data):
        observations, rewards_to_go = data["observations"], data["rewards_to_go"]

        return ((self.policy.critic(observations) - rewards_to_go)**2).mean()

    def update(self):
        data = self.buffer.get()

        for _ in range(self.k_epochs):
            self.actor_optimizer.zero_grad()
            loss, info = self.compute_loss_actor(data)

            loss.backward()
            self.actor_optimizer.step()

        for _ in range(self.k_epochs):
            self.critic_optimizer.zero_grad()
            loss = self.compute_loss_critic(data)

            loss.backward()
            self.critic_optimizer.step()

    def save(self, checkpoint_path):
        torch.save(self.policy.state_dict(), checkpoint_path)

class PPO_2T(PPO):

    def __init__(self,
                 num_observations,
                 num_actions,
                 timesteps_per_epoch,
                 actor_lr=.0003,
                 critic_lr=.001,
                 k_epochs=80,
                 gamma=0.99,
                 lambda_GAE=0.95,
                 epsilon=0.2,
                 num_teams=1,
                 log_std_init=-0.5,
                 activation=nn.Tanh,
                 action_activation=nn.Identity):
        
        super().__init__(self, num_observations, num_actions, timesteps_per_epoch, actor_lr, critic_lr, k_epochs, gamma, lambda_GAE, epsilon, num_teams, log_std_init, activation, action_activation)

        self.buffer_t1 = Buffer(num_observations, num_actions, timesteps_per_epoch, gamma, lambda_GAE)
        self.buffer_t2 = Buffer(num_observations, num_actions, timesteps_per_epoch, gamma, lambda_GAE)


    def update(self):
        data_t1 = self.buffer_t1.get()
        data_t2 = self.buffer_t2.get()

        data = {
            "observations": torch.concatenate((data_t1["observations"], data_t2["observations"])),
            "actions": torch.concatenate((data_t1["actions"], data_t2["actions"])),
            "rewards": torch.concatenate((data_t1["rewards"], data_t2["rewards"])),
            "values": torch.concatenate((data_t1["values"], data_t2["values"])),
            "log_probs": torch.concatenate((data_t1["log_probs"], data_t2["log_probs"])),
            "advantages": torch.concatenate((data_t1["advantages"], data_t2["advantages"])),
            "rewards_to_go": torch.concatenate((data_t1["rewards_to_go"], data_t2["rewards_to_go"])),
        }

        for _ in range(self.k_epochs):
            self.actor_optimizer.zero_grad()
            loss, info = self.compute_loss_actor(data)

            loss.backward()
            self.actor_optimizer.step()

        for _ in range(self.k_epochs):
            self.critic_optimizer.zero_grad()
            loss = self.compute_loss_critic(data)

            loss.backward()
            self.critic_optimizer.step()
