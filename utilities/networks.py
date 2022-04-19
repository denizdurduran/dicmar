import numpy as np

import random
import sys

import torch
import torch.nn as nn
from torch.distributions import Normal
import os
import utilities.constants as ct

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
    Set Linear Layers weights for neural networks
'''
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.01)
        nn.init.constant_(m.bias, 0.1)

'''
    Defines Actor and Critic Networks architecture 
'''
class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, std=0.0):
        super(ActorCritic, self).__init__()

        self.actor = sequential_network(num_inputs, num_outputs)

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, ct.NUM_NODES),
            nn.PReLU(),
            nn.Linear(ct.NUM_NODES, ct.NUM_NODES),
            nn.PReLU(),
            nn.Linear(ct.NUM_NODES, ct.NUM_NODES),
            nn.PReLU(),
            nn.Linear(ct.NUM_NODES, 1),
        )

        self.log_std = nn.Parameter(torch.zeros(num_outputs))

        self.apply(init_weights)

    def forward(self, x):
        value = self.critic(x)
        mu    = self.actor(x)
        std   = self.log_std.exp().expand_as(mu)
        std   = std.to(device)
        dist  = Normal(mu, std*0.1)
        return dist, value, std

'''
    Get activation function
'''
def activation_functions(name):
    #If you want to add any activation function, check if there are any parameters to be given 
    # See the Hardtanh case for an example
    if name == "relu":
        act_func = nn.ReLU()

    elif name == "prelu":
        act_func = nn.PReLU()

    elif name == "leaky":
        act_func = nn.LeakyReLU()

    elif name == "tanh":
        act_func = nn.Tanh()

    elif name == "sigmoid":
        act_func = nn.Sigmoid()

    elif name == "hardtanh":
        architecture = torch.load(ct.ARCHITECTURE_PATH)
        if ct.CHECKPOINT is None:
            min_val = input("Please enter minimum value of Hardtanh:\n")
            max_val = input("Please enter maximum value of Hardtanh:\n")
            architecture['min_value'] = min_val
            architecture['max_value'] = max_val
            torch.save(architecture, ct.ARCHITECTURE_PATH)
        else:
            min_val = architecture['min_value']
            max_val = architecture['max_value']

        act_func = nn.Hardtanh(float(min_val), float(max_val))
        
    return act_func

'''
    Create Sequential Network
'''
def sequential_network(num_inputs, num_outputs):

    network_layers = []

    for i in range(ct.NUM_LAYERS):
        if i == 0:
            network_layers.append(nn.Linear(num_inputs, ct.NUM_NODES))
            network_layers.append(activation_functions(ct.ACTIVATION))
        elif i == (ct.NUM_LAYERS-1):
            network_layers.append(nn.Linear(ct.NUM_NODES, num_outputs))
            network_layers.append(activation_functions(ct.LAST_ACTIVATION))
        else:
            network_layers.append(nn.Linear(ct.NUM_NODES, ct.NUM_NODES))
            network_layers.append(activation_functions(ct.ACTIVATION))

    network_layers
    network = nn.Sequential(*network_layers)

    return network

'''
    Saves current best actions, best reward and critic loss with actor, critic and optimizer parameters
'''
def save_model(episode, model_musculo, optimizer_musculo, best_reward, mean_reward, critic_loss, best_actions, kind):
    critic_loss = critic_loss.cpu().detach().numpy()
    critic_loss = np.mean(critic_loss)
    if kind == 'best':
        reward = best_reward
    else:
        reward = mean_reward
    print("Episode: {} - Saving model, Current Error: {}, Current {} Reward: {}".format(episode, critic_loss, kind, reward))
    ppo_model_arm_musculo = {
        'model_state_dict': model_musculo.state_dict(),
        'optimizer_state_dict': optimizer_musculo.state_dict(),
        'mean_reward': mean_reward,
        'best_reward': best_reward,
        'critic_loss': critic_loss,
        'best_actions': best_actions}
    ppo_path =  f"{ct.CHECKPOINT_PATH}/ppo_exo_hips_only_{kind}_{episode}_{reward}"
    torch.save(ppo_model_arm_musculo, ppo_path)
    return ppo_path

'''
    Load parameters into actor, critic and optimizer
'''
def load_model(actor_critic, optimizer):

    file_name, ep = get_file()
    checkpoint_file = f"{ct.CHECKPOINT_PATH}/{file_name}"
    actor_critic.load_state_dict(torch.load(checkpoint_file)["model_state_dict"])
    optimizer.load_state_dict(torch.load(checkpoint_file)["optimizer_state_dict"])
    loaded = torch.load(checkpoint_file)
    print('Loaded Model has Mean Reward : {} ; Best Reward : {} and Critic Loss : {}'.format(loaded["mean_reward"], loaded["best_reward"], loaded ["critic_loss"]))
    return loaded["mean_reward"], loaded["best_reward"], loaded["best_actions"], ep

'''
    Get file with highest reward
'''
def get_file():

    files_list = []
    rewards_list = []
    #take checkpoint file names with either best or mean reward, take the file with highest reward
    for file in os.listdir(ct.CHECKPOINT_PATH):
        if file.startswith(f"ppo_exo_hips_only_{ct.CHECKPOINT}_"):
            x = file.rsplit('_')
            rewards_list.append(float(x[-1]))
            files_list.append(file)

    file_name = max(rewards_list)
    ind = rewards_list.index(file_name)
    episode = int(files_list[ind].rsplit('_')[-2])

    return files_list[ind], episode

'''
    Saves architecture of network
'''
def save_architecture():
    architecture = {
        'activation'   : ct.ACTIVATION,
        'last_activ'   : ct.LAST_ACTIVATION,
        'num_layers'   : ct.NUM_LAYERS,
        'num_nodes'    : ct.NUM_NODES,
        #'num_inputs'   : num_inputs,
        #'num_outputs'  : num_outputs,
        'range'        : ct.RANGE 
    } 
    torch.save(architecture, ct.ARCHITECTURE_PATH)

'''
    Save model, actions and episode #
'''
def save_best(best_reward, mean_reward, rewards, ep, actions, model, optimizer, critic, kind):

    best_ep = ep
    num_thread = np.argmax(rewards)
    best_actions = actions.cpu().numpy()[num_thread::ct.NUM_ENVS]
    model_name = save_model(best_ep, model, optimizer, best_reward, mean_reward, critic, best_actions, kind)

    return model_name, best_actions, best_ep
