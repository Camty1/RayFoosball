import torch
import torch.nn as nn
import torch.optim as optim
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

class Buffer:
    def __init__(self):
        self.actions = []
        self.observations = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.is_terminal = []
        self.terminal_values = []
        self.terminal_count = 0

    def clear(self):
        del self.actions[:]
        del self.observations[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.values[:]
        del self.is_terminal[:]
        del self.terminal_values[:]
        self.terminal_count = 0
        
class ActorCritic(nn.Module):
    def __init__(self,
                 observation_dim,
                 action_dim,
                 action_std_init,
                 action_out_layer="sigmoid"):

        super(ActorCritic, self).__init__()

        assert action_out_layer in {"sigmoid", "tanh", "linear"}, "Invalid action output layer type"

        self.action_dim = action_dim
        self.action_std = action_std_init
        if action_out_layer == "sigmoid": 
            self.actor = nn.Sequential(
                nn.Linear(observation_dim, 64),
                nn.Tanh(),
                nn.Linear(64,64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Sigmoid()
            )
        elif action_out_layer == "linear":
            self.actor = nn.Sequential(
                nn.Linear(observation_dim, 64),
                nn.Tanh(),
                nn.Linear(64,64),
                nn.Tanh(),
                nn.Linear(64, action_dim)
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(observation_dim, 64),
                nn.Tanh(),
                nn.Linear(64,64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Tanh()
            )

    
        self.critic = nn.Sequential(
            nn.Linear(observation_dim, 64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,1)
        )


    def set_action_std(self, new_action_std):
        self.action_std = new_action_std

    def forward(self):
        raise NotImplementedError

    def act(self, obs):
        
        # Run obs through actor network
        action_mean = self.actor(obs)

        # Create a covariance matrix for normal dist
        std = self.action_std

        # Create and sample distribution for action
        dist = Normal(action_mean, std)
        action = dist.sample()

        # Get log prob (used later for PPO clipping)
        action_log_prob = dist.log_prob(action).sum(axis=-1)

        # Get value for current obs
        val = self.critic(obs)

        return action.detach(), action_log_prob.detach(), val.detach()

    def evaluate(self, obs, action):

        action_mean = self.actor(obs)

        std = self.action_std

        dist = Normal(action_mean, std)

        action_log_prob = dist.log_prob(action)
        dist_entropy = dist.entropy()
        value = self.critic(obs)

        return action_log_prob, value, dist_entropy

class PPO:
    def __init__(self, 
                 obs_dim, 
                 action_dim, 
                 lr_actor, 
                 lr_critic, 
                 gamma,
                 K_epochs,
                 lambda_GAE,
                 epsilon_clip,
                 action_std_init=0.3,
                 action_out_layer="sigmoid"):

        self.gamma = gamma
        self.K_epochs = K_epochs
        self.lambda_GAE = lambda_GAE
        self.epsilon_clip = epsilon_clip
        self.action_std = action_std_init

        self.buffer = Buffer()

        self.policy = ActorCritic(obs_dim, action_dim, action_std_init, action_out_layer).to(device)

        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(obs_dim, action_dim, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MSELoss = nn.MSELoss()
    
    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        self.action_std = self.action_std - action_std_decay_rate
        #self.action_std = round(self.action_std, 4) Why is rounding necessary

        if self.action_std < min_action_std:
            self.action_std = min_action_std

        self.set_action_std(self.action_std)

    def get_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(device)
            action, log_prob, value = self.policy_old.act(obs)

        self.buffer.observations.append(obs)
        self.buffer.actions.append(action)
        self.buffer.log_probs.append(log_prob)
        self.buffer.values.append(value)

        return action.detach().cpu().numpy().flatten()

    def update(self, c1=0.5, c2 = 0.01):
        reward_to_go, advantage = self._GAE(self.buffer.rewards, self.buffer.values, self.buffer.is_terminal, self.buffer.terminal_values, self.buffer.terminal_count)

        reward_to_go = torch.from_numpy(reward_to_go).to(device)
        advantage = torch.from_numpy(advantage).to(device)


        observations = torch.squeeze(torch.stack(self.buffer.observations, dim=0)).detach().to(device)

        actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)

        old_log_probs = torch.squeeze(torch.stack(self.buffer.log_probs, dim=0)).detach().to(device)

        for _ in range(self.K_epochs):

            new_log_probs, values, dist_entropy = self.policy.evaluate(observations, actions)

            values = torch.squeeze(values)

            ratios = torch.exp(new_log_probs - old_log_probs)

            unclipped = ratios * advantage
            clipped = torch.clamp(ratios, 1-self.epsilon_clip, 1+self.epsilon_clip) * advantage

            loss = -torch.min(unclipped, clipped) + c1 * self.MSELoss(values, reward_to_go) - c2 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

        self.buffer.clear()
    
    def get_value(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(device)
            value = self.policy_old.critic(obs)
        return value

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))


    def _GAE(self,
             rewards, 
             values, 
             is_terminal, 
             terminal_values, 
             terminal_count):
        
        assert len(rewards) == len(values), "Reward (len {}) and value (len {}) vectors are not the same length".format(len(rewards), len(values))

        assert len(values) == len(is_terminal), "Value (len {}) and terminal (len {}) vectors are not the same length".format(len(values), len(is_terminal))

        assert len(terminal_values) == terminal_count, "Terminal values vector (len {}) does not match terminal count (val {})".format(len(terminal_values), terminal_count)

        reward_to_go = np.zeros_like(rewards, dtype=np.float32)
        advantage = np.zeros_like(rewards, dtype=np.float32)

        for i in reversed(range(len(rewards))):
            if is_terminal[i]:
                reward_to_go[i] = rewards[i]
                delta = rewards[i] + self.gamma * terminal_values[terminal_count-1] - values[i]
                advantage[i] = delta
                terminal_count -= 1
            else:
                reward_to_go[i] = rewards[i] + self.gamma * reward_to_go[i+1]
                delta = rewards[i] + self.gamma * values[i+1] - values[i]
                advantage[i] = delta + self.gamma * self.lambda_GAE * advantage[i+1]

        return reward_to_go, advantage

