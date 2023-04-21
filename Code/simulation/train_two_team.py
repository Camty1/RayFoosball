import torch
import numpy as np
from twoTeamPPO import PPO_TT
import foosballGym as gym
import os
import time
from datetime import datetime

def train(mode="full_state"):

    training_modes = {"full_state", "just_position", "just_team", "just_team_position", "decentralized_full", "decentralized_position", "decentralized_team", "decentralized_team_position"}

    assert mode in training_modes, "Invalid training mode passed"
    
    # Environment hyperparameters
    max_ep_len = 60*30
    max_training_timesteps = int(5.4e6)

    print_frequency = max_ep_len * 12
    log_frequency = max_ep_len * 2
    save_model_frequency = max_ep_len * 100

    action_std = 0.3
    action_std_decay_rate = 0.0125
    min_action_std = 0.025
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
    K_epochs = 150
    clip = .2

    random_seed = 0
    
    # Handle different training modes
    if mode == "full_state":
        agent = PPO_TT(full_obs, centralized_actions, actor_lr, critic_lr, gamma, K_epochs, clip)

    if mode == "just_position":
        agent = PPO_TT(position_obs, centralized_actions, actor_lr, critic_lr, gamma, K_epochs, clip)

    if mode == "just_team":
        agent = PPO_TT(team_obs, centralized_actions, actor_lr, critic_lr, gamma, K_epochs, clip)

    if mode == "just_team_position":
        agent = PPO_TT(team_position_obs, centralized_actions, actor_lr, critic_lr, gamma, K_epochs, clip)

    if mode == "decentralized_full":
        goalie = PPO_TT(full_obs, decentralized_actions, actor_lr, critic_lr, gamma, K_epochs, clip)
        defense = PPO_TT(full_obs, decentralized_actions, actor_lr, critic_lr, gamma, K_epochs, clip)
        midfield = PPO_TT(full_obs, decentralized_actions, actor_lr, critic_lr, gamma, K_epochs, clip)
        striker = PPO_TT(full_obs, decentralized_actions, actor_lr, critic_lr, gamma, K_epochs, clip)

    # Start environment
    env = gym.FoosballEnv(just_goal=False)

    log_dir = "PPO_out"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dir = log_dir + "/" + mode + "/"

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    run_num = 0
    current_files = next(os.walk(log_dir))[2]
    run_num = len(current_files)

    log_file_name = log_dir + "PPO_log_" + mode + "_" + str(run_num) + ".csv"

    print("Current run: ", run_num)
    print("Log file: " + log_file_name)

    run_num_pretrained = 0

    directory = "PPO_preTrained"

    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + "/" + mode + "/"

    if not os.path.exists(directory):
        os.makedirs(directory)

    checkpoint_path = directory + "PPO_{}_{}_{}_tt".format(mode, random_seed, run_num_pretrained)

    print("Checkpoint paths: " + checkpoint_path)

    start_time = datetime.now().replace(microsecond=0)
    print("Training start: ", start_time)

    log_file = open(log_file_name, "w+")
    log_file.write("Episode,timestep,reward1,reward2\n")

    print_running_reward_t1 = 0
    print_running_reward_t2 = 0
    print_running_episodes = 0

    log_running_reward_t1 = 0
    log_running_reward_t2 = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    while time_step <= max_training_timesteps:
        state, _ = env.reset()
        current_ep_reward_t1 = 0
        current_ep_reward_t2 = 0
        print_reward = 0

        for t in range(1, max_ep_len+1):

            processed_state = _handle_state(state, mode)

            # Centralized
            if mode[0] != "d":
                t1_action, t2_action = agent.get_action(processed_state["t1"], processed_state["t2"])
                action = np.concatenate((t1_action, t2_action))

                state, reward, done, _, _ = env.step(action)

                agent.buffer_t1.rewards.append(reward["t1_reward"])
                agent.buffer_t2.rewards.append(reward["t2_reward"])

                agent.buffer_t1.is_terminal.append(done)
                agent.buffer_t2.is_terminal.append(done)

                time_step += 1
                current_ep_reward_t1 += reward["t1_reward"]
                current_ep_reward_t2 += reward["t2_reward"]

                if time_step % update_frequency == 0:
                    processed_state = _handle_state(state, mode)
                    q_value_t1 = agent.get_q_value(processed_state["t1"])
                    agent.buffer_t1.q_values.append(q_value_t1)
                    q_value_t2 = agent.get_q_value(processed_state["t2"])

                    agent.buffer_t2.q_values.append(q_value_t2)
                    
                    agent.update()
                
                if time_step % action_std_decay_frequency == 0:
                    agent.decay_action_std(action_std_decay_rate, min_action_std)
                
                if time_step % log_frequency == 0:
                    validation_reward = validate(env, agent, state)
                    print_reward += validation_reward
                    log_file.write("{},{},{}\n".format(i_episode, time_step, round(validation_reward, 4)))
                    log_file.flush()

                if time_step % print_frequency == 0: 
                    print_avg_reward_t1 = print_running_reward_t1 / print_running_episodes
                    print_avg_reward_t2 = print_running_reward_t2 / print_running_episodes

                    print("{},{},{}".format(i_episode, time_step, round(validation_reward * (log_frequency/print_frequency), 4)))

                    print_reward = 0


                if time_step % save_model_frequency == 0:
                    print("**************************")
                    print("Saving model: " + checkpoint_path)
                    agent.save(checkpoint_path)
                    print("**************************")
                    print("Time: ", datetime.now().replace(microsecond=0) - start_time)
                    print("**************************")

                if done:
                    break

                print_running_reward_t1 += current_ep_reward_t1
                print_running_reward_t2 += current_ep_reward_t2
                print_running_episodes += 1

                log_running_reward_t1 += current_ep_reward_t1
                log_running_reward_t2 += current_ep_reward_t2
                log_running_episodes += 1

                i_episode += 1

            # TODO
            else:
                t1_goalie_action, t2_goalie_action = goalie.get_action(processed_state["t1"], processed_state["t2"])
                t1_defense_action, t2_defense_action = defense.get_action(processed_state["t1"], processed_state["t2"])
                t1_midfield_action, t2_midfield_action = midfield.get_action(processed_state["t1"], processed_state["t2"])
                t1_striker_action, t2_striker_action = striker.get_action(processed_state["t1"], processed_state["t2"])
                action = np.concatenate((t1_goalie_action, t1_defense_action, t1_midfield_action, t1_striker_action, t2_goalie_action, t2_defense_action, t1_midfield_action, t1_striker_action))

                state, reward, done, _, _ = env.step(action)

                goalie.buffer_t1.rewards.append(reward["t1_reward"])
                goalie.buffer_t2.rewards.append(reward["t2_reward"])

                goalie.buffer_t1.is_terminal.append(done)
                goalie.buffer_t2.is_terminal.append(done)
                
                defense.buffer_t1.rewards.append(reward["t1_reward"])
                defense.buffer_t2.rewards.append(reward["t2_reward"])

                defense.buffer_t1.is_terminal.append(done)
                defense.buffer_t2.is_terminal.append(done)

                midfield.buffer_t1.rewards.append(reward["t1_reward"])
                midfield.buffer_t2.rewards.append(reward["t2_reward"])

                midfield.buffer_t1.is_terminal.append(done)
                midfield.buffer_t2.is_terminal.append(done)

                striker.buffer_t1.rewards.append(reward["t1_reward"])
                striker.buffer_t2.rewards.append(reward["t2_reward"])

                striker.buffer_t1.is_terminal.append(done)
                striker.buffer_t2.is_terminal.append(done)

                time_step += 1
                current_ep_reward_t1 += reward["t1_reward"]
                current_ep_reward_t2 += reward["t2_reward"]

                if time_step % update_frequency == 0:
                    processed_state = _handle_state(state, mode)
                    
                    q_value_t1_goalie = goalie.get_q_value(processed_state["t1"])
                    q_value_t2_goalie = goalie.get_q_value(processed_state["t2"])
                    q_value_t1_defense = defense.get_q_value(processed_state["t1"])
                    q_value_t2_defense = defense.get_q_value(processed_state["t2"])
                    q_value_t1_midfield = midfield.get_q_value(processed_state["t1"])
                    q_value_t2_midfield = midfield.get_q_value(processed_state["t2"])
                    q_value_t1_striker = striker.get_q_value(processed_state["t1"])
                    q_value_t2_striker = striker.get_q_value(processed_state["t2"])

                    goalie.buffer_t1.q_values.append(q_value_t1_goalie)
                    goalie.buffer_t2.q_values.append(q_value_t2_goalie)
                    defense.buffer_t1.q_values.append(q_value_t1_defense)
                    defense.buffer_t2.q_values.append(q_value_t2_defense)
                    midfield.buffer_t1.q_values.append(q_value_t1_midfield)
                    midfield.buffer_t2.q_values.append(q_value_t2_midfield)
                    striker.buffer_t1.q_values.append(q_value_t1_striker)
                    striker.buffer_t2.q_values.append(q_value_t2_striker)
                    
                    goalie.update()
                    defense.update()
                    midfield.update()
                    striker.update()
                
                if time_step % action_std_decay_frequency == 0:
                    goalie.decay_action_std(action_std_decay_rate, min_action_std)
                    defense.decay_action_std(action_std_decay_rate, min_action_std)
                    midfield.decay_action_std(action_std_decay_rate, min_action_std)
                    striker.decay_action_std(action_std_decay_rate, min_action_std)
                
                if time_step % log_frequency == 0:
                    validation_reward = decentralized_validate(env, goalie, defense, midfield, striker, state)
                    print_reward += validation_reward
                    log_file.write("{},{},{}\n".format(i_episode, time_step, round(validation_reward, 4)))
                    log_file.flush()

                if time_step % print_frequency == 0: 
                    print_avg_reward_t1 = print_running_reward_t1 / print_running_episodes
                    print_avg_reward_t2 = print_running_reward_t2 / print_running_episodes

                    print("{},{},{}".format(i_episode, time_step, round(validation_reward * (log_frequency/print_frequency), 4)))

                    print_reward = 0


                if time_step % save_model_frequency == 0:
                    print("**************************")
                    print("Saving model: " + checkpoint_path)
                    goalie.save(checkpoint_path+"_goalie")
                    defense.save(checkpoint_path+"_defense")
                    midfield.save(checkpoint_path+"_midfield")
                    striker.save(checkpoint_path+"_striker")
                    print("**************************")
                    print("Time: ", datetime.now().replace(microsecond=0) - start_time)
                    print("**************************")

                if done:
                    break

                print_running_reward_t1 += current_ep_reward_t1
                print_running_reward_t2 += current_ep_reward_t2
                print_running_episodes += 1

                log_running_reward_t1 += current_ep_reward_t1
                log_running_reward_t2 += current_ep_reward_t2
                log_running_episodes += 1

                i_episode += 1

    log_file.close()
    env.close()

    end_time = datetime.now().replace(microsecond=0)
    print("Training time: ", end_time-start_time)

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

def validate(env, agent, initial_state, random=False, mode="full_state"):
    state, _ = env.reset(None, False)
    cumulative_reward = 0
    for _ in range(60 * 10):
        processed_state = _handle_state(state, mode)
        
        t1_action = agent.get_action_validation(processed_state["t1"])
        
        if random == False:
            t2_action = np.zeros(8)

        else:
            t2_action = np.random.rand(8)

        action = np.concatenate((t1_action, t2_action))

        state, reward, done, _, _ = env.step(action)
        
        cumulative_reward += reward["t1_reward"]

        if done:
            state, _ = env.reset(None, False)
    
    env.set_state(initial_state)

    return cumulative_reward

def decentralized_validate(env, goalie, defense, midfield, striker, initial_state, random=False, mode="full_state"):
    state, _ = env.reset(None, False)
    cumulative_reward = 0
    for _ in range(60 * 10):
        processed_state = _handle_state(state, mode)
        
        goalie_action = goalie.get_action_validation(processed_state["t1"])
        defense_action = defense.get_action_validation(processed_state["t1"])
        midfield_action = midfield.get_action_validation(processed_state["t1"])
        striker_action = striker.get_action_validation(processed_state["t1"])
        
        if random == False:
            t2_action = np.full(.5, 8)

        else:
            t2_action = np.random.rand(8)

        action = np.concatenate((goalie_action, defense_action, midfield_action, striker_action, t2_action))

        state, reward, done, _, _ = env.step(action)
        
        cumulative_reward += reward["t1_reward"]

        if done:
            state, _ = env.reset(None, False)
    
    env.set_state(initial_state)

    return cumulative_reward

if __name__ == "__main__":
    train("decentralized_full")

