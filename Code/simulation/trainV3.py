import torch
import torch.nn as nn
import numpy as np
from ppoV3 import PPO
import gymnasium as gym
import os
import time
from datetime import datetime

def train():

    # Environment hyperparameters
    max_ep_len = 200
    max_training_timesteps = int(10e6)

    print_frequency = max_ep_len * 10
    log_frequency = max_ep_len * 2
    save_model_frequency = max_ep_len * 100
    
    # PPO agent hyperparams (add lambda for GAE?)
    update_frequency = max_ep_len * 4
    actor_lr = .0003
    critic_lr = .001
    gamma = .99
    K_epochs = 80
    lambda_GAE = 0.95
    epsilon_clip = .2

    agent = PPO(3, 1, update_frequency)
    env = gym.make("Pendulum-v1")
    
    log_dir = "PPO_out"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dir = log_dir + "/pendulum/"

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    run_num = 0
    current_files = next(os.walk(log_dir))[2]
    run_num = len(current_files)

    log_file_name = log_dir + "PPO_log_" + str(run_num) + ".csv"

    print("Current run: ", run_num)
    print("Log file: " + log_file_name)

    run_num_pretrained = 0

    directory = "PPO_preTrained"

    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + "/pendulum/"

    if not os.path.exists(directory):
        os.makedirs(directory)

    checkpoint_path = directory + "PPO_{}".format(run_num_pretrained)

    print("Checkpoint paths: " + checkpoint_path)

    start_time = datetime.now().replace(microsecond=0)
    print("Training start: ", start_time)

    log_file = open(log_file_name, "w+")
    log_file.write("Episode,timestep,reward\n")

    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 1
    i_episode = 0
    log_reward = 0
    print_reward = 0

    while time_step <= max_training_timesteps + 1:
        obs, _ = env.reset()
        current_ep_reward = 0

        for t in range(update_frequency):
            
            act, val, log_prob = agent.policy.step(torch.as_tensor(obs, dtype=torch.float32))

            next_obs, rew, done, _, _ = env.step(act)
            current_ep_reward += rew
            log_reward += rew
            print_reward += rew
            i_episode += 1
            time_step += 1

            agent.buffer.store(obs, act, rew, val, log_prob)

            obs = next_obs

            timeout = t == max_ep_len
            terminal = done or timeout
            epoch_ended = t == update_frequency
            
            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print("Cut off trajectory at %d steps"%i_episode, flush=True)

                if timeout or epoch_ended:
                    _, value, _ = agent.policy.step(torch.as_tensor(obs, dtype=torch.float32))
                else:
                    value = 0

                agent.buffer.GAE(value)
                obs, _ = env.reset()
            if time_step % print_frequency == 0:
                print("{},{}".format(time_step, print_reward/print_frequency))
                print_reward = 0

            if time_step % log_frequency == 0:
                log_file.write("{},{}\n".format(time_step,log_reward/log_frequency))
                log_file.flush()

                log_reward = 0

            if time_step % save_model_frequency == 0 or time_step == max_training_timesteps-1:
                print("**************************")
                print("Saving models: " + checkpoint_path)
                agent.save(checkpoint_path)
                print("**************************")
                print("Time: ", datetime.now().replace(microsecond=0) - start_time)
                print("**************************")
        agent.update()

if __name__ == "__main__":
    train()
