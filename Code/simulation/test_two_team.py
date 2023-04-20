import torch
import numpy as np
from twoTeamPPO import PPO_TT
import foosballGym as gym
import time

def _handle_state(state, mode="full_state"):

    if mode == "full_state" or mode == "decentralized_full":
        team1 = np.concatenate((state["ball_pos"], state["ball_vel"], state["t1_pos"], state["t1_vel"], state["t2_pos"], state["t2_vel"]))
        
        team2 = np.concatenate((np.negative(state["ball_pos"]), np.negative(state["ball_vel"]), state["t2_pos"], state["t2_vel"], state["t1_pos"], state["t1_vel"]))
        states = {"t1": team1, "t2": team2}

    if mode == "just_position" or mode == "decentralized_position":
        team1 = np.concatenate((state["ball_pos"], state["ball_vel"], state["t1_pos"], state["t2_pos"]))
        team2 = np.concatenate((np.negative(state["ball_pos"]), np.negative(state["ball_vel"]), state["t2_pos"], state["t1_pos"]))
        states = {"t1": team1, "t2": team2}

    if mode == "just_team" or mode == "decentralized_team":
        team1 = np.concatenate((state["ball_pos"], state["ball_vel"], state["t1_pos"], state["t1_vel"]))
        team2 = np.concatenate((np.negative(state["ball_pos"]), np.negative(state["ball_vel"]), state["t2_pos"], state["t2_vel"]))
        states = {"t1": team1, "t2": team2}

    if mode == "just_team_position" or mode == "decentralized_team_position":
        team1 = np.concatenate((state["ball_pos"], state["ball_vel"], state["t1_pos"]))
        team2 = np.concatenate((np.negative(state["ball_pos"]), np.negative(state["ball_vel"]), state["t2_pos"]))
        states = {"t1": team1, "t2": team2}

    return states

runtime = 30

# Number of observations for different training modes 
full_obs = 36
position_obs = 20
team_obs = 20
team_position_obs = 12

# Number of actions for different training modes
centralized_actions = 8
decentralized_actions = 2

# PPO agent hyperparams (add lambda for GAE?)
actor_lr = .0003
critic_lr = .001
gamma = .99
K_epochs = 80
clip = .2

random_seed = 0

mode = "full_state"

env = gym.FoosballEnv("human")

agent = PPO_TT(full_obs, centralized_actions, actor_lr, critic_lr, gamma, K_epochs, clip, 0.00001)

directory = "PPO_preTrained"
directory = directory + "/" + mode + "/"

run_num_pretrained = 0

checkpoint_path = directory + "PPO_{}_{}_{}_tt".format(mode, random_seed, run_num_pretrained)

agent.load(checkpoint_path)
#team2.load(checkpoint_path_t2)

state, _ = env.reset(None, False)

for _ in range(60 * runtime): 
    processed_state = _handle_state(state, mode)

    t1_action, t2_action = agent.get_action(processed_state["t1"], processed_state["t2"])
    
    action = np.concatenate((t1_action, t2_action))

    state, reward, done, _, _ = env.step(action)

    if done:
        state, _ = env.reset(None, False)

