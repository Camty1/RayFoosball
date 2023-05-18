import torch
import torch.nn as nn
import numpy as np
from ppoV3 import PPO
from ppoV3 import PPO_2T
import foosballGym as gym
import os
import time
from datetime import datetime
import argparse 

def train(observation_mode="full", both_teams=False, reset_mode="striker", actor_lr=0.0003, critic_lr=0.001):
    observation_modes = {"full", "position", "team", "team_position"}
    reset_modes = {"normal", "random", "striker"}

    assert observation_mode in observation_modes
    assert reset_mode in reset_modes

    # Environment hyperparameters
    max_ep_len = 15*60
    print_frequency = max_ep_len * 10
    log_frequency = max_ep_len * 2
    save_model_frequency = max_ep_len * 100
    training_epochs = 2500
    
    # PPO agent hyperparams (add lambda for GAE?)
    update_frequency = max_ep_len * 4
    gamma = .99
    K_epochs = 80
    lambda_GAE = 0.95
    epsilon_clip = .2

    # Number of observations for different training modes 
    full_obs = 36
    position_obs = 20
    team_obs = 20
    team_position_obs = 12
    
    # Number of actions for different training modes
    centralized_actions = 8
    decentralized_actions = 2
    if both_teams:
        if observation_mode == "full":
            agent = PPO_2T(full_obs, centralized_actions, update_frequency, actor_lr=actor_lr, critic_lr=critic_lr)
        if observation_mode == "position":
            agent = PPO_2T(position_obs, centralized_actions, update_frequency, actor_lr=actor_lr, critic_lr=critic_lr)
        if observation_mode == "team":
            agent = PPO_2T(team_obs, centralized_actions, update_frequency, actor_lr=actor_lr, critic_lr=critic_lr)
        if observation_mode == "team_position":
            agent = PPO_2T(team_position_obs, centralized_actions, update_frequency, actor_lr=actor_lr, critic_lr=critic_lr)
    else:
        if observation_mode == "full":
            agent = PPO(full_obs, centralized_actions, update_frequency, actor_lr=actor_lr, critic_lr=critic_lr)
        if observation_mode == "position":
            agent = PPO(position_obs, centralized_actions, update_frequency, actor_lr=actor_lr, critic_lr=critic_lr)
        if observation_mode == "team":
            agent = PPO(team_obs, centralized_actions, update_frequency, actor_lr=actor_lr, critic_lr=critic_lr)
        if observation_mode == "team_position":
            agent = PPO(team_position_obs, centralized_actions, update_frequency, actor_lr=actor_lr, critic_lr=critic_lr)

    env = gym.FoosballEnv()
    
    log_dir = "PPO_out"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_dir = log_dir + "/foosball/" + observation_mode + "/centralized/" + reset_mode + "/"

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    run_num = 0
    current_files = next(os.walk(log_dir))[2]
    run_num = len(current_files)

    log_file_name = log_dir + "PPO_log_" + str(run_num) + ".csv"

    print("Current run: ", run_num)
    print("Log file: " + log_file_name)

    run_num_pretrained = 0

    save_dir = "PPO_preTrained"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_dir = save_dir + "/foosball/" + observation_mode + "/centralized/" + reset_mode + "/"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    checkpoint_path = save_dir + "PPO_{}".format(run_num_pretrained)

    print("Checkpoint paths: " + checkpoint_path)

    start_time = datetime.now().replace(microsecond=0)
    print("Training start: ", start_time)

    log_file = open(log_file_name, "w+")
    log_file.write("Episode,timestep,reward\n")

    i_episode = 0
    time_step = 0
    log_reward_t1 = 0
    print_reward_t1 = 0
    log_reward_t2 = 0
    print_reward_t2 = 0


    for e in range(training_epochs):
        obs, _ = env.reset(start_type=reset_mode)

        for t in range(update_frequency):
            
            processed_obs = handle_obs(obs, observation_mode, both_teams)
            
            act, val, log_prob = 0, 0, 0

            if both_teams:
                act_t1, val_t1, log_prob_t1 = agent.policy.step(torch.as_tensor(processed_obs[0], dtype=torch.float32))
                act_t2, val_t2, log_prob_t2 = agent.policy.step(torch.as_tensor(processed_obs[1], dtype=torch.float32))
                act = [act_t1, act_t2]
                val = [val_t1, val_t2]
                log_prob = [log_prob_t1, log_prob_t2]
            else:
                act, val, log_prob = agent.policy.step(torch.as_tensor(processed_obs, dtype=torch.float32))

            processed_act = handle_act(act, both_teams)

            next_obs, rew, done, _, _ = env.step(processed_act)

            i_episode += 1
            time_step += 1

            if both_teams:
                log_reward_t1 += rew["t1"]
                print_reward_t1 += rew["t1"]
                agent.buffer_t1.store(processed_obs[0], act[0], rew["t1"], val[0], log_prob[0])
                
                log_reward_t2 += rew["t2"]
                print_reward_t2 += rew["t2"]
                agent.buffer_t2.store(processed_obs[1], act[1], rew["t2"], val[1], log_prob[1])
                
            else:
                log_reward_t1 += rew["t1"]
                print_reward_t1 += rew["t1"]
                agent.buffer.store(processed_obs, act, rew["t1"], val, log_prob)

            obs = next_obs

            timeout = i_episode == max_ep_len
            terminal = done or timeout
            epoch_ended = t == update_frequency-1
            
            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print("Cut off trajectory at %d steps"%i_episode, flush=True)
                    i_episode = 0
                
                if both_teams:
                    if timeout or epoch_ended:
                        processed_obs = handle_obs(obs, observation_mode, both_teams)
                        _, value_t1, _ = agent.policy.step(torch.as_tensor(processed_obs[0], dtype=torch.float32))
                        _, value_t2, _ = agent.policy.step(torch.as_tensor(processed_obs[1], dtype=torch.float32))
                        
                        i_episode = 0
                    else:
                        value_t1 = 0
                        value_t2 = 0

                    agent.buffer_t1.GAE(value_t1)
                    agent.buffer_t2.GAE(value_t2)

                else:
                    if timeout or epoch_ended:
                        processed_obs = handle_obs(obs, observation_mode, both_teams)
                        _, value, _ = agent.policy.step(torch.as_tensor(processed_obs, dtype=torch.float32))
                        i_episode = 0

                    else:
                        value = 0

                    agent.buffer.GAE(value)

                obs, _ = env.reset(start_type=reset_mode)
            if both_teams:
                if time_step % print_frequency == 0:
                    print("{},{},{}".format(time_step, print_reward_t1/print_frequency, print_reward_t2/print_frequency))

                    print_reward_t1 = 0
                    print_reward_t2 = 0

                if time_step % log_frequency == 0:
                    log_file.write("{},{},{}".format(time_step, log_reward_t1/log_frequency, log_reward_t2/log_frequency))
                    log_file.flush()

                    log_reward_t1 = 0
                    log_reward_t2 = 0

            else:
                if time_step % print_frequency == 0:
                    print("{},{}".format(time_step, print_reward_t1/print_frequency))

                    print_reward_t1 = 0

                if time_step % log_frequency == 0:
                    log_file.write("{},{}".format(time_step, log_reward_t1/log_frequency))
                    log_file.flush()

                    log_reward_t1 = 0


            if time_step % save_model_frequency == 0:
                print("**************************")
                print("Saving models: " + checkpoint_path)
                agent.save(checkpoint_path)
                print("**************************")
                print("Time: ", datetime.now().replace(microsecond=0) - start_time)
                print("**************************")
        
        agent.update()

    print("Training Finished!")
    print("**************************")
    print("Saving models: " + checkpoint_path)
    agent.save(checkpoint_path)
    print("**************************")
    print("Time: ", datetime.now().replace(microsecond=0) - start_time)
    print("**************************")
    env.close()

def handle_obs(obs, observation_mode, both_teams):
    if both_teams:
        processed_obs = []
        if observation_mode == "full":
            t1_obs = np.concatenate([obs["ball_pos"], obs["ball_vel"], obs["t1_pos"], obs["t1_vel"], obs["t2_pos"], obs["t2_vel"]], dtype=np.float32).flatten()
            t2_obs = np.concatenate([-obs["ball_pos"], -obs["ball_vel"], obs["t2_pos"], obs["t2_vel"], obs["t1_pos"], obs["t1_vel"]], dtype=np.float32).flatten()

            processed_obs.append(t1_obs)
            processed_obs.append(t2_obs)

        elif observation_mode == "team":
            t1_obs = np.concatenate([obs["ball_pos"], obs["ball_vel"], obs["t1_pos"], obs["t1_vel"]], dtype=np.float32).flatten()
            t2_obs = np.concatenate([-obs["ball_pos"], -obs["ball_vel"], obs["t2_pos"], obs["t2_vel"]], dtype=np.float32).flatten()

            processed_obs.append(t1_obs)
            processed_obs.append(t2_obs)
        
        elif observation_mode == "position":
            t1_obs = np.concatenate([obs["ball_pos"], obs["ball_vel"], obs["t1_pos"], obs["t2_pos"]], dtype=np.float32).flatten()
            t2_obs = np.concatenate([-obs["ball_pos"], -obs["ball_vel"], obs["t2_pos"], obs["t1_pos"]], dtype=np.float32).flatten()

            processed_obs.append(t1_obs)
            processed_obs.append(t2_obs)

        else:
            t1_obs = np.concatenate([obs["ball_pos"], obs["ball_vel"], obs["t1_pos"]], dtype=np.float32).flatten()
            t2_obs = np.concatenate([-obs["ball_pos"], -obs["ball_vel"], obs["t2_pos"]], dtype=np.float32).flatten()

            processed_obs.append(t1_obs)
            processed_obs.append(t2_obs)
    
        return processed_obs

    else:
        if observation_mode == "full":
            return np.concatenate([obs["ball_pos"], obs["ball_vel"], obs["t1_pos"], obs["t1_vel"], obs["t2_pos"], obs["t2_vel"]], dtype=np.float32).flatten()

        elif observation_mode == "team":
            return np.concatenate([obs["ball_pos"], obs["ball_vel"], obs["t1_pos"], obs["t1_vel"]], dtype=np.float32).flatten()
        
        elif observation_mode == "position":
            return np.concatenate([obs["ball_pos"], obs["ball_vel"], obs["t1_pos"], obs["t2_pos"]], dtype=np.float32).flatten()

        else:
            return np.concatenate([obs["ball_pos"], obs["ball_vel"], obs["t1_pos"]], dtype=np.float32).flatten()
    
def handle_act(act, both_teams):

    processed_act = np.zeros((16,), dtype=np.float32)

    if both_teams:
        processed_act[:8] = act[0]
        processed_act[8:] = act[1]

    else:
        processed_act[:8] = act

    return processed_act

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Foosball Training", description="Performs training using foosball environment")
    parser.add_argument('-o', '--observation', choices=["full", "position", "team", "team_position"], default="full", help="Specify observation mode")
    parser.add_argument('-t', '--teams', action="store_true", default=False, help="Train using both teams")
    parser.add_argument('-r', '--reset', choices=["normal", "random", "striker"], default="striker", help="Choose environment reset mode")
    parser.add_argument("-a", "--actor_lr", default=0.0003, help="Actor learning rate")
    parser.add_argument("-c", "--critic_lr", default=0.001, help="Critic learning rate")
    args = parser.parse_args()
    print("Observation: " + args.observation, "| Both teams: " + str(args.teams), "| Reset: " + args.reset, "| Actor_lr: " + args.actor_lr, "| Critic_lr: " + args.critic_lr)
    train(args.observation, args.teams, args.reset, actor_lr=float(args.actor_lr), critic_lr=float(args.critic_lr))

