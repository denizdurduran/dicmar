## -*- coding: utf-8 -*

#███████╗██╗  ██╗ ██████╗ ███████╗██╗  ██╗███████╗██╗     ███████╗████████╗ ██████╗ ███╗   ██╗          
#██╔════╝╚██╗██╔╝██╔═══██╗██╔════╝██║ ██╔╝██╔════╝██║     ██╔════╝╚══██╔══╝██╔═══██╗████╗  ██║          
#█████╗   ╚███╔╝ ██║   ██║███████╗█████╔╝ █████╗  ██║     █████╗     ██║   ██║   ██║██╔██╗ ██║    █████╗
#██╔══╝   ██╔██╗ ██║   ██║╚════██║██╔═██╗ ██╔══╝  ██║     ██╔══╝     ██║   ██║   ██║██║╚██╗██║    ╚════╝
#███████╗██╔╝ ██╗╚██████╔╝███████║██║  ██╗███████╗███████╗███████╗   ██║   ╚██████╔╝██║ ╚████║          
#╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═══╝          
#                                                                                                       
#███╗   ██╗ ██████╗      █████╗  ██████╗████████╗██╗   ██╗ █████╗ ████████╗██╗ ██████╗ ███╗   ██╗       
#████╗  ██║██╔═══██╗    ██╔══██╗██╔════╝╚══██╔══╝██║   ██║██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║       
#██╔██╗ ██║██║   ██║    ███████║██║        ██║   ██║   ██║███████║   ██║   ██║██║   ██║██╔██╗ ██║       
#██║╚██╗██║██║   ██║    ██╔══██║██║        ██║   ██║   ██║██╔══██║   ██║   ██║██║   ██║██║╚██╗██║       
#██║ ╚████║╚██████╔╝    ██║  ██║╚██████╗   ██║   ╚██████╔╝██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║       
#╚═╝  ╚═══╝ ╚═════╝     ╚═╝  ╚═╝ ╚═════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝     

# with deap.


import sys             # SKIPPED
from control.osim_HBP_withexo_partial import L2M2019Env, L2M2019EnvVecEnv
from control.osim_loco_reflex_song2019 import OsimReflexCtrl
import numpy as np

import random
import sys

import pickle

seed = 64
SEED = int(seed)   # Random seed
random.seed(SEED)
np.random.seed(SEED)

difficulty = 0
sim_dt = 0.01

#Set Parameters
DESIRED_SPEED       = 1.3
INITIAL_SPEED       = 1.6

# Initial position of the model 
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

# Get Joint Angles and Muscle Activations
with open('./logs/simulation_data/cmaes_spinal_2_62_1_second_all_joints.pkl', 'rb') as f:
    joints = pickle.load(f)

with open('./logs/simulation_data/cmaes_spinal_2_62_1_second_muscle_act.pkl', 'rb') as f:
    muscle_act = pickle.load(f)

muscle_activities = np.vstack(muscle_act)

# Create environment
env = L2M2019Env(visualize=True, seed=SEED, difficulty=difficulty, desired_speed=DESIRED_SPEED)
observation = env.reset(project=True, seed=SEED, obs_as_dict=True, init_pose=INIT_POSE)

# Run an episode with no actuation on full exoskeleton. The muscles activations 
# given to the musculoskeletal model are the same as the ones provided by Step1.
for i in range(100):
    actions = muscle_activities[i, :]
    exo_actuation = np.zeros(6)
    exo_actuation[5] = 10
    actions = np.concatenate((actions, exo_actuation))
    print("Current muscle fibers")
    print(env.get_observation_muscle_fiber())
    observation, reward, done, info = env.step(actions, project = True, obs_as_dict=True)
    print("Current observation function to be the muscle fiber length only")
    print(observation)
    if i == 99:
        input("End of simulation, press enter to close!")
    if(done):
        input('End of simulation, the model fell below the threshold. Press enter to close!')
        break
