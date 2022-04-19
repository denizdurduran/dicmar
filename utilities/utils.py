import sys             # SKIPPED
from control.osim_HBP_withexo_partial import L2M2019Env, L2M2019EnvVecEnv
from control.osim_loco_reflex_song2019 import OsimReflexCtrl
import numpy as np
import argparse
import re
import random
import sys

import pickle
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

import matplotlib as mpl
mpl.use("Agg")
import utilities.plots as myplots
import utilities.ppo as ppo

import utilities.constants as ct
import utilities.networks as nt

use_cuda = torch.cuda.is_available()
print(use_cuda)
device   = torch.device("cuda" if use_cuda else "cpu")

from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from multiprocessing_env import SubprocVecEnv

seed = 64
SEED = int(seed)   # Random seed
random.seed(SEED)
np.random.seed(SEED)

difficulty = 0
sim_dt = 0.01

'''
    Write parameters and network architectures to the optimization folder, summarized in a text file
'''
def create_params_file(model_musculo, optimizer_musculo):

    with open("{}/params_{}.txt".format(ct.CHECKPOINT_PATH, ct.RUN_NUM), 'w') as f:
        params_summary_txt = '''
    NUM_ENVS           {}
    NUM_EPISODES       {}
    NUM_EPOCHS         {}
    MINI_BATCH_SIZE    {}
    CLIP_PARAM         {}
    LEARNING_RATE      {}
    DESIRED_SPEED      {}
    INITIAL_SPEED      {}
    RANGE              {}
    NUM_LAYERS         {}
    NUM_NODES          {}   

    NETWORKS_ARCHITECTURES 
    {}
    OPTIMIZER
    {}
    '''.format(ct.NUM_ENVS, ct.NUM_EPISODES, ct.NUM_EPOCHS, ct.MINI_BATCH_SIZE, ct.CLIP_PARAM, ct.LEARNING_RATE,
               ct.DESIRED_SPEED, ct.INITIAL_SPEED, ct.RANGE, ct.NUM_LAYERS, ct.NUM_NODES, model_musculo, optimizer_musculo)
        f.write(params_summary_txt)

'''
    Creates environment
'''
def make_env():
    def _thunk():
        env = L2M2019EnvVecEnv(visualize=False, seed=SEED, difficulty=difficulty, desired_speed=ct.DESIRED_SPEED)
        return env
    return _thunk

'''
    Initialize several environments with fixed initial conditions
'''
def init_envs():
    # Initial position of the model 
    global INIT_POSE
    INIT_POSE = np.array([
        ct.INITIAL_SPEED,              # forward speed
        .5,                         # rightward speed
        9.023245653983965608e-01,   # pelvis height
        2.012303881285582852e-01,   # trunk lean
        0*np.pi/180,                # [right] hip adduct
        -6.952390849304798115e-01,  # hip flex
        -3.231075259785813891e-01,  # knee extend
        1.709011708233401095e-01,   # ankle flex
        0*np.pi/180,                # [left] hip adduct
        -5.282323914341899296e-02,  # hip flex
        -8.041966456860847323e-01,  # knee extend
        -1.745329251994329478e-01]) # ankle flex

    # Create environments in parallel
    num_envs = ct.NUM_ENVS

    envs = [make_env() for i in range(num_envs)]
    envs = SubprocVecEnv(envs)
    state = envs.reset()

    env = L2M2019Env(visualize=False, seed=SEED, difficulty=difficulty, desired_speed=ct.DESIRED_SPEED)
    observation = env.reset(project=True, seed=SEED, obs_as_dict=True, init_pose=INIT_POSE)
    return envs, env, state

'''
    Get Joint Positions and Muscle Activations
'''
def get_positions():
    with open('./logs/simulation_data/cmaes_spinal_2_62_5_second_all_joints.pkl', 'rb') as f:
        joints = pickle.load(f)

    with open('./logs/simulation_data/cmaes_spinal_2_62_5_second_muscle_act.pkl', 'rb') as f:
        muscle_act = pickle.load(f)

    muscle_activities = np.vstack(muscle_act)
    joints = np.vstack(joints)

    return muscle_activities

'''
    Initialize Actor Critic Networks
'''
def init_actorcritic():

    num_inputs  = 22#envs.observation_space.shape[0]
    num_outputs = 2#envs.action_space.shape[0]

    #Hyper params:
    betas            = (0.9, 0.999)
    eps              = 1e-08
    weight_decay     = 0.001
    num_steps        = 500

    model_musculo = nt.ActorCritic(num_inputs, num_outputs).to(device)
    optimizer_musculo = optim.Adam(model_musculo.parameters(), lr=ct.LEARNING_RATE)

    return model_musculo, optimizer_musculo, num_steps

'''
    Runs an episode with given actions, returns the joints angular positions 
'''
def dummy_env(num_steps, actions, env, muscle_activities):

    current_pos = []
    state = env.reset(project=True, seed=SEED, obs_as_dict=True, init_pose=INIT_POSE)
    current_pos.append(env.get_observation_list_joints())
    done = False
    for i in range(num_steps-1):
        muscle_actions = muscle_activities[i, :]
        action = np.concatenate((muscle_actions, actions[i]*ct.RANGE))
        env.step(action)
        current_pos.append(env.get_observation_list_joints())

    return current_pos

'''
    Launch optimization according to parameters given to parser
'''
def run_optimization():

    envs, env, state = init_envs()
    model_musculo, optimizer_musculo, num_steps = init_actorcritic()
    create_params_file(model_musculo, optimizer_musculo)
    muscle_activities = get_positions()

    #Load model parameters
    if ct.CHECKPOINT is not None:
        best_mean_reward, best_reward_all, best_actions, ep  = nt.load_model(model_musculo, optimizer_musculo)
        best_actions_all = best_actions
        model_name = ct.CHECKPOINT
        best_model_name = ct.CHECKPOINT
    else:
        best_mean_reward = -10000
        best_reward_all = -10000
        ep = 0    
    
    best_reward_all_ep = ep
    best_ep = ep

    episodes_rewards = []
    best_episodes_rewards = []
    episodes_critic_losses = []
    episodes_kl = []
    episodes_actor_losses = []
    episodes_losses = []
    episodes_entropies = []
    episodes_std = []
    
    #Episodes
    for steps in range(ep, ct.NUM_EPISODES+ep):

        print("Episode #{}, Best Episode : #{}".format(steps, best_ep))

        log_probs = []
        values    = []
        states    = []
        actions   = []
        rewards   = []
        masks     = []
        stds = []
        entropy = 0
        episode_reward = 0

        #frames of episodes, each actions is determined with a NN receiving muscle activities as inputs
        #Those actions are given to the actuators which impacts the state of the model
        for i in range(num_steps):

            muscle_actions = np.tile(muscle_activities[i, :], (ct.NUM_ENVS, 1))

            state = torch.FloatTensor(state).to(device)
            dist, value, _ = model_musculo(state)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            actions.append(action)
            action = action.cpu().numpy()
            action_sent = np.hstack((muscle_actions, action*ct.RANGE))
            next_state, reward, done, _ = envs.step(action_sent)
            entropy += dist.entropy().mean()
            episode_reward += reward
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
            masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))
            states.append(state)
            state = next_state
            #if done and is_first_done:


        episodes_rewards.append(episode_reward)
        envs.reset()
        next_state = torch.FloatTensor(next_state).to(device)
        _, next_value, _ = model_musculo(next_state)
        returns = ppo.compute_gae(next_value, rewards, masks, values)

        returns   = torch.cat(returns).detach()
        log_probs = torch.cat(log_probs).detach()
        values    = torch.cat(values).detach()
        states    = torch.cat(states)
        actions   = torch.cat(actions)
        advantage = returns - values
        
        #update action policy using PPO
        actor_losses, critic_losses, entropies, losses, kl_divs, stds = ppo.ppo_update(model_musculo, optimizer_musculo,
                                                                        states, actions, log_probs, returns, advantage)
        episodes_actor_losses.append(actor_losses)
        episodes_critic_losses.append(critic_losses)
        episodes_entropies.append(entropies)
        episodes_losses.append(losses)
        episodes_kl.append(kl_divs)
        episodes_std.append(stds)

        #keep thread with best reward
        best_thread_reward = np.max(episode_reward)
        best_episodes_rewards.append(best_thread_reward)

        mean_rewards = np.mean(episode_reward)

        #Save policy with the highest reward so far
        if best_thread_reward > best_reward_all:
            best_model_name, best_actions_all, best_reward_all_ep = nt.save_best(best_thread_reward, mean_rewards, episode_reward, steps,
                                                          actions, model_musculo, optimizer_musculo, critic_losses, 'best')
            best_reward_all = best_thread_reward

        #Save policy with the highest average reward so far
        if mean_rewards > best_mean_reward:
            model_name, best_actions, best_ep = nt.save_best(best_thread_reward, mean_rewards, episode_reward, steps,
                                                           actions, model_musculo, optimizer_musculo, critic_losses, 'mean')
            best_mean_reward = mean_rewards

        #Plot Rewards, Losses (critic, actor), entropies, std of action pdf, KL divergence
        myplots.plot_rewards_losses(episodes_rewards, episodes_actor_losses, episodes_critic_losses, episodes_entropies,
                        episodes_std, episodes_losses, episodes_kl, best_episodes_rewards, best_reward_all,
                        best_reward_all_ep, ep)

    print("Optimization Completed ; Best Episode : #{} with Reward : {}. Model Saved in {}".format(best_ep, best_mean_reward, model_name))
    print("Episode with highest Reward : #{} with Reward : {}. Model Saved in {}".format(best_reward_all_ep, best_reward_all, best_model_name))
    
    #plot joints angular positions based on actions that gave highest reward ever during optimization
    # and for best mean episode
    best_ep_pos = dummy_env(num_steps, best_actions, env, muscle_activities)
    best_all_pos = dummy_env(num_steps, best_actions_all, env, muscle_activities)
    myplots.plot_pos(best_ep_pos, "best_mean", env)
    myplots.plot_pos(best_all_pos, "best_all", env)
    myplots.plot_actions(best_actions_all*ct.RANGE, best_actions*ct.RANGE)
