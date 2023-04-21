import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal
import numpy as np

device = torch.device("cpu")
if (torch.cuda.is_available()):
    device = torch.device('cuda')
    torch.cuda.empty_cache()

print("Device: " + str(torch.cuda.get_device_name(device)))

class Buffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.log_probs = []
        self.rewards = []
        self.q_values = []
        self.is_terminal = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.q_values[:]
        del self.is_terminal[:]
        

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init):
        super(ActorCritic, self).__init__()

        self.action_dim = action_dim
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Sigmoid()
        )
    
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,1)
        )


    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        
        # Run state through actor network
        action_mean = self.actor(state)

        # Create a covariance matrix for normal dist
        covariance = torch.diag(self.action_var).unsqueeze(dim=0)

        # Create and sample distribution for action
        dist = MultivariateNormal(action_mean, covariance)
        action = dist.sample()

        # Get log prob (used later for PPO clipping)
        action_log_prob = dist.log_prob(action)

        # Get Q value for current state
        q_val = self.critic(state)

        return action.detach(), action_log_prob.detach(), q_val.detach()

    def evaluate(self, state, action):

        action_mean = self.actor(state)

        action_var = self.action_var.expand_as(action_mean)
        covariance = torch.diag_embed(action_var).to(device)
        dist = MultivariateNormal(action_mean, covariance)

        action_log_prob = dist.log_prob(action)
        dist_entropy = dist.entropy()
        q_values = self.critic(state)

        return action_log_prob, q_values, dist_entropy

    
class PPO_TT:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std_init=0.3):

        self.action_std = action_std_init
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        
        self.buffer_t1 = Buffer()
        self.buffer_t2 = Buffer()

        self.policy = ActorCritic(state_dim, action_dim, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(),  'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim, action_std_init).to(device)
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

    
    def get_action(self, state_t1, state_t2):
        with torch.no_grad():
            state_t1 = torch.FloatTensor(state_t1).to(device)
            state_t2 = torch.FloatTensor(state_t2).to(device)
            action_t1, action_log_prob_t1, q_val_t1 = self.policy_old.act(state_t1)
            action_t2, action_log_prob_t2, q_val_t2 = self.policy_old.act(state_t2)

        self.buffer_t1.states.append(state_t1)
        self.buffer_t1.actions.append(action_t1)
        self.buffer_t1.log_probs.append(action_log_prob_t1)
        self.buffer_t1.q_values.append(q_val_t1)

        self.buffer_t2.states.append(state_t2)
        self.buffer_t2.actions.append(action_t2)
        self.buffer_t2.log_probs.append(action_log_prob_t2)
        self.buffer_t2.q_values.append(q_val_t2)

        return action_t1.detach().cpu().numpy().flatten(), action_t2.detach().cpu().numpy().flatten()


    def get_action_validation(self, state_t1):
        with torch.no_grad():
            state_t1 = torch.FloatTensor(state_t1).to(device)
            action_t1, action_log_prob_t1, q_val_t1 = self.policy_old.act(state_t1)

        return action_t1.detach().cpu().numpy().flatten()


    def update(self):
        q_values_t1, advantages_t1 = self._get_advantages_gae(self.buffer_t1.q_values, self.buffer_t1.is_terminal, self.buffer_t1.rewards)
        q_values_t2, advantages_t2 = self._get_advantages_gae(self.buffer_t2.q_values, self.buffer_t2.is_terminal, self.buffer_t2.rewards)

        q_values_t1 = torch.from_numpy(q_values_t1).to(device)
        q_values_t2 = torch.from_numpy(q_values_t2).to(device)
        advantages_t1 = torch.from_numpy(advantages_t1).to(device)
        advantages_t2 = torch.from_numpy(advantages_t2).to(device)

        old_states_t1 = torch.squeeze(torch.stack(self.buffer_t1.states, dim=0)).detach().to(device)
        old_states_t2 = torch.squeeze(torch.stack(self.buffer_t2.states, dim=0)).detach().to(device)
        
        old_states = torch.cat((old_states_t1, old_states_t2))

        old_actions_t1 = torch.squeeze(torch.stack(self.buffer_t1.actions, dim=0)).detach().to(device)
        old_actions_t2 = torch.squeeze(torch.stack(self.buffer_t2.actions, dim=0)).detach().to(device)
        
        old_actions = torch.cat((old_actions_t1, old_actions_t2))
        
        old_log_probs_t1 = torch.squeeze(torch.stack(self.buffer_t1.log_probs, dim=0)).detach().to(device)
        old_log_probs_t2 = torch.squeeze(torch.stack(self.buffer_t2.log_probs, dim=0)).detach().to(device)
        
        old_log_probs = torch.cat((old_log_probs_t1, old_log_probs_t2))
        
        q_values = torch.cat((q_values_t1, q_values_t2))
        advantages = torch.cat((advantages_t1, advantages_t2))
        

        for i in range(self.K_epochs):

            log_probs, new_q_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            new_q_values = torch.squeeze(new_q_values)

            ratios = torch.exp(log_probs - old_log_probs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2) + 0.5 * self.MSELoss(new_q_values, q_values) - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

        self.buffer_t1.clear()
        self.buffer_t2.clear()


    def get_q_value(self, state):
        with torch.no_grad():
            torch_state = torch.FloatTensor(state).to(device)
            q_value = self.policy_old.critic(torch_state)
        return q_value

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

    ## State values has one more entry than rewards
    def _get_advantages_gae(self, q_values, is_terminal, rewards, lmbda=0.95):
        new_q_values = []
        cumulative_adv = 0
        
        assert len(q_values) == len(rewards) + 1, "State values (len {}) is not one longer than rewards (len {}), GAE cannot run".format(len(q_values), len(rewards))

        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * q_values[i+1] * (not is_terminal[i]) - q_values[i]
            cumulative_adv = delta + self.gamma * lmbda * (not is_terminal[i]) * cumulative_adv
            new_q_values.insert(0, cumulative_adv + q_values[i])
        
        new_q_values = torch.squeeze(torch.stack(new_q_values)).detach().cpu().numpy()
        old_q_values = torch.squeeze(torch.stack(q_values)).detach().cpu().numpy()
        
        # A = Q - V
        adv = new_q_values - old_q_values[:-1]
        
        adv_normalized = (adv - np.mean(adv)) / (np.std(adv) + 1e-10)

        return new_q_values, adv_normalized

    def _get_advantages_mc(self, q_values, is_terminal, rewards):
        q_values = []
        discounted_reward = 0

        for reward, terminal in zip(reversed(rewards), reversed(is_terminal)):
            if is_terminal:
                discounted_reward = 0

            discounted_reward = reward + (self.gamma * discounted_reward)
            q_values.insert(0, discounted_reward)

        q_values = np.asarray(q_values)

        q_values_normalized = (q_values - np.mean(q_values)) / (np.std(q_values) + 1e-10)

        advantages = q_values_normalized - q_values[:-1]

        return q_values_normalized, advantages
