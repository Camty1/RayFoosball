import torch
import numpy as np
from ppoAgentV2 import PPO
import gymnasium as gym
import os
from datetime import datetime

def train():
    max_ep_len = 1000 
    max_training_timesteps = int(3e6)

    print_frequency = max_ep_len * 10
    log_frequency = max_ep_len * 2
    save_model_frequency = max_ep_len * 100

    action_std = 0.3
    action_std_decay_rate = 0.025
    min_action_std = 0.05
    action_std_decay_frequency = int(max_ep_len*250)
    
    update_frequency = max_ep_len * 4
    actor_lr = .0003
    critic_lr = .001
    gamma = .99
    K_epochs = 80
    lambda_GAE = .95
    epsilon_clip = .2

    random_seed = 0

    agent = PPO(3, 1, actor_lr, critic_lr, gamma, K_epochs, lambda_GAE, epsilon_clip, action_out_layer="tanh")

    env = gym.make('Pendulum-v1')
    
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

    checkpoint_path = directory + "PPO_{}_{}".format(random_seed, run_num_pretrained)

    print("Checkpoint paths: " + checkpoint_path)

    start_time = datetime.now().replace(microsecond=0)
    print("Training start: ", start_time)

    log_file = open(log_file_name, "w+")
    log_file.write("Episode,timestep,reward\n")

    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0
    
    while time_step <= max_training_timesteps:
        obs, _ = env.reset()
        current_ep_reward = 0


        for t in range(1, max_ep_len+1):
            action = agent.get_action(obs)
            obs, reward, done, _, _ = env.step(action)

            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminal.append(done)

            time_step += 1
            current_ep_reward += reward
            
            if time_step % update_frequency == 0:
                agent.buffer.terminal_values.append(agent.get_value(obs))
                agent.buffer.terminal_count += 1
                agent.buffer.is_terminal[-1] = True

                agent.update()

            if time_step % action_std_decay_frequency == 0:
                agent.decay_action_std(action_std_decay_rate, min_action_std)

            if time_step % log_frequency == 0:
                log_avg_reward = log_running_reward / log_running_episodes

                log_file.write("{},{},{}\n".format(i_episode, time_step, round(log_avg_reward, 4)))
                log_file.flush()

                log_running_reward = 0
                log_running_episodes = 0

            if time_step % print_frequency == 0: 
                print_avg_reward = print_running_reward / print_running_episodes

                print("{},{},{}".format(i_episode, time_step, round(print_avg_reward, 2)))

                print_running_reward = 0
                print_running_episodes = 0

            if time_step % save_model_frequency == 0:
                print("**************************")
                print("Saving models: " + checkpoint_path)
                agent.save(checkpoint_path)
                print("**************************")
                print("Time: ", datetime.now().replace(microsecond=0) - start_time)
                print("**************************")

            if done:
                break

            print_running_reward += current_ep_reward
            print_running_episodes += 1

            log_running_reward += current_ep_reward
            log_running_episodes += 1

            i_episode += 1

        if len(agent.buffer.is_terminal) > 0:
            agent.buffer.terminal_values.append(agent.get_value(obs))
            agent.buffer.terminal_count += 1
            agent.buffer.is_terminal[-1] = True

if __name__ == "__main__":
    train()

