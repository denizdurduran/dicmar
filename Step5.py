import sys             # SKIPPED
from control.osim_HBP_withexo_partial import L2M2019Env, L2M2019EnvVecEnv
from control.osim_loco_reflex_song2019 import OsimReflexCtrl
import numpy as np
import argparse

import math
import random
import sys

import pickle
import time
import os

import gym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import LogNormal

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

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

DESIRED_SPEED       = 1.3
INITIAL_SPEED       = 1.6
INIT_POSE = np.array([
    INITIAL_SPEED,              # forward speed
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

with open('./logs/simulation_data/cmaes_spinal_2_62_1_second_joint.pkl', 'rb') as f:
    joints = pickle.load(f)

with open('./logs/simulation_data/cmaes_spinal_2_62_1_second_muscle_act.pkl', 'rb') as f:
    muscle_act = pickle.load(f)

muscle_activities = np.vstack(muscle_act)

num_envs = 1
def make_env():
    def _thunk():
        env = L2M2019Env(visualize=False, seed=SEED, difficulty=difficulty, desired_speed=DESIRED_SPEED)
        return env
    return _thunk

envs = [make_env() for i in range(num_envs)]
envs = SubprocVecEnv(envs)

env = L2M2019Env(visualize=True, seed=SEED, difficulty=difficulty, desired_speed=DESIRED_SPEED)
observation = env.reset(project=True, seed=SEED, obs_as_dict=True, init_pose=INIT_POSE)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.01)
        nn.init.constant_(m.bias, 0.1)


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.PReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.PReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.PReLU(),
            nn.Linear(hidden_size, 1),
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Tanh(),
        )
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std).data.squeeze()

        self.apply(init_weights)

    def forward(self, x):
        value = self.critic(x)
        mu    = self.actor(x)
        std   = self.log_std.exp().expand_as(mu)
        std = std.to(device)
        dist  = Normal(mu, std*0.1)
        return dist, value

def plot(frame_idx, rewards):
    plt.figure(figsize=(12,8))
    plt.subplot(111)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.savefig("results/Exp_1/ppo_exo_hips_only_{}".format(frame_idx))
    plt.close()


def test_env(num_steps, count):
    state = env.reset(project=True, seed=SEED, obs_as_dict=True, init_pose=INIT_POSE)
    with open('./logs/simulation_data/cmaes_spinal_2_62_1_second_all_joints.pkl', 'rb') as f:
        joints = pickle.load(f)
    joint_activities = np.vstack(joints)
    state_hip_r = []
    state_hip_l = []
    done = False
    total_reward = 0
    input("Please initiate the recording from GUI then press any key to continue")

    for i in range(num_steps):
        list_all = env.get_observation_list_exojoints()
        state_hip_r.append(list_all[0])
        state_hip_l.append(list_all[1])
        muscle_actions = muscle_activities[i, :]
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = model_musculo(state)
        action = dist.sample().cpu().numpy()[0]
        action = np.concatenate((muscle_actions, action))
        next_state, reward, done, _ = env.step(action, project = True, obs_as_dict=True)

        state = next_state
        total_reward += reward

    plt.figure()
    plt.plot(joint_activities[:,0])
    plt.plot(joint_activities[:,1])
    plt.plot(state_hip_r)
    plt.plot(state_hip_l)
    plt.savefig("results/Exp_1/exo_ppo_states_all_musculo_{}_{}".format(model_id, count))
    plt.close()
    envs.reset()
    return total_reward



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-mi", "--model_id", type=int, default=300, help="Model ID")
    parser.add_argument("-c", "--counter", type=int, default=1, help="number of tests")
    parser.add_argument("-ns", "--num_steps", type=int, default=100, help="number of steps")
    args = parser.parse_args()

    num_inputs  = 22#envs.observation_space.shape[0]
    num_outputs = 2#envs.action_space.shape[0]

    state = envs.reset()
    #Hyper params:
    hidden_size      = 32
    lr               = 2e-4
    betas            = (0.9, 0.999)
    eps              = 1e-08
    weight_decay     = 0.001
    mini_batch_size  = 200
    ppo_epochs       = 200
    threshold_reward = -200

    model_musculo = ActorCritic(num_inputs, num_outputs, hidden_size).to(device)
    optimizer_musculo = optim.Adam(model_musculo.parameters(), lr=lr)

    model_id = args.model_id
    counter = args.counter
    num_steps = args.num_steps

    ppo_model_arm_musculo_loaded = torch.load("results/Exp_1/ppo_exo_hips_only_{}".format(model_id), map_location=device)

    model_musculo.load_state_dict(ppo_model_arm_musculo_loaded['model_state_dict'])
    optimizer_musculo.load_state_dict(ppo_model_arm_musculo_loaded['optimizer_state_dict'])

    frame_idx = ppo_model_arm_musculo_loaded['epoch']
    #test_rewards = ppo_model_arm_musculo_loaded['loss']

    test_all_rewards = np.array([test_env(num_steps, i) for i in range(counter)])
