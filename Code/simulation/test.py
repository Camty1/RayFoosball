import torch
import numpy as np
from ppoAgentV2 import PPO
import foosballGym as gym
import time

def _handle_state(state, mode="full_state"):

    if mode == "full_state" or mode == "decentralized_full":
        team1 = np.concatenate((state["ball_pos"], state["ball_vel"], state["t1_pos"], state["t1_vel"], state["t2_pos"], state["t2_vel"]))
        team2 = np.concatenate((state["ball_pos"], state["ball_vel"], state["t2_pos"], state["t2_vel"], state["t1_pos"], state["t1_vel"]))
        states = {"t1": team1, "t2": team2}

    if mode == "just_position" or mode == "decentralized_position":
        team1 = np.concatenate((state["ball_pos"], state["ball_vel"], state["t1_pos"], state["t2_pos"]))
        team2 = np.concatenate((state["ball_pos"], state["ball_vel"], state["t2_pos"], state["t1_pos"]))
        states = {"t1": team1, "t2": team2}

    if mode == "just_team" or mode == "decentralized_team":
        team1 = np.concatenate((state["ball_pos"], state["ball_vel"], state["t1_pos"], state["t1_vel"]))
        team2 = np.concatenate((state["ball_pos"], state["ball_vel"], state["t2_pos"], state["t2_vel"]))
        states = {"t1": team1, "t2": team2}

    if mode == "just_team_position" or mode == "decentralized_team_position":
        team1 = np.concatenate((state["ball_pos"], state["ball_vel"], state["t1_pos"]))
        team2 = np.concatenate((state["ball_pos"], state["ball_vel"], state["t2_pos"]))
        states = {"t1": team1, "t2": team2}

    return states

runtime = 30

max_ep_len = 60*30
max_training_timesteps = int(3e6)

print_frequency = max_ep_len * 10
log_frequency = max_ep_len * 2
save_model_frequency = max_ep_len * 100

action_std = 0.00001
action_std_decay_rate = 0.0125
min_action_std = 0.05
action_std_decay_frequency = int(max_ep_len*250)

# Number of observations for different training modes 
full_obs = 36
position_obs = 20
team_obs = 20
team_position_obs = 12

# Number of actions for different training modes
centralized_actions = 8
decentralized_actions = 2

# PPO agent hyperparams (add lambda for GAE?)
update_frequency = max_ep_len * 4
actor_lr = .0003
critic_lr = .001
gamma = .99
K_epochs = 80
clip = .2

random_seed = 0

mode = "just_team"

env = gym.FoosballEnv("human")
just_goal = False

team1 = PPO(team_obs, centralized_actions, actor_lr, critic_lr, gamma, K_epochs, clip, action_std)

directory = "PPO_preTrained"
directory = directory + "/" + mode + "/"

run_num_pretrained = 0

checkpoint_path = directory + "PPO_{}_{}_{}".format(mode, random_seed, run_num_pretrained)

team1.load(checkpoint_path)

zeros = np.zeros(8)

state, _ = env.reset(None, "striker")

for _ in range(60 * runtime): 
    processed_state = _handle_state(state, mode)

    t1_action = team1.get_action(processed_state["t1"])
    t2_action = np.zeros(8)
    action = np.concatenate((t1_action, t2_action))

    state, reward, done, _, _ = env.step(action)

    if done:
        state, _ = env.reset(None, "striker")

