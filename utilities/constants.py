import sys   
import argparse
import os
import torch
from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

'''
    Parser Creation for Step4.py + declaration of global constants
'''
def create_parser():
    # Script parameters
    parser = argparse.ArgumentParser(prog='Reinforcement Learning (With PPO)')
    parser.add_argument("-envs",  "--envs",  help="The number of environments launched in parallel during optimization", default=16)
    parser.add_argument("-episodes", "--num_episodes", help="Number of simulations of the model", default=50000)
    parser.add_argument("-epochs", "--num_epochs", help="Number of times PPO replays one episode", default=100)
    parser.add_argument("-mbs", "--mini_batch_size", help="Number of samples for one PPO update", default=100)
    parser.add_argument("-cp", "--clip_param", help="Clip Parameter for PPO", default=0.2)
    parser.add_argument("-lr", "--learning_rate", help="Learning rate for Optimizer", default=2e-4)
    parser.add_argument("-tgt_speed","--tgt_speed",      help="Target/Desired speed", default=1.3)
    parser.add_argument("-init_speed", "--init_speed",   help="Initial speed of the simulation", default=1.6)
    parser.add_argument("-range", "--range",   help="Range of Actuation", default=100)
    parser.add_argument("-af", "--activ_func",   help="Activation functions", default="prelu")
    parser.add_argument("-laf", "--last_activ_func",   help="Last Activation function", default="hardtanh")
    parser.add_argument("-layers", "--num_layers",   help="Number of Hidden Layers", default=4)
    parser.add_argument("-nodes", "--nodes",   help="Number of Nodes per Hidden Layer", default=32)
    parser.add_argument("-c", "--checkpoint", help="Checkpoint to use for initial parameters", default= None)
    parser.add_argument("-id", "--id", help="ID of folder's checkpoint", default= 0)

    '''
        If you enter a value for -c and -id different than their default ones, there is no need to specify
        range, af, laf, layers and nodes
    '''

    args = parser.parse_args()

    global NUM_ENVS           
    global NUM_EPISODES        
    global NUM_EPOCHS          
    global MINI_BATCH_SIZE
    global CLIP_PARAM
    global LEARNING_RATE
    global DESIRED_SPEED
    global INITIAL_SPEED
    global RANGE
    global ACTIVATION
    global LAST_ACTIVATION
    global NUM_LAYERS                         
    global NUM_NODES           
    global CHECKPOINT
    global ID
    

    NUM_ENVS            = int(args.envs)
    NUM_EPISODES        = int(args.num_episodes)
    NUM_EPOCHS          = int(args.num_epochs)
    MINI_BATCH_SIZE     = int(args.mini_batch_size)
    CLIP_PARAM          = float(args.clip_param)
    LEARNING_RATE       = float(args.learning_rate)
    DESIRED_SPEED       = float(args.tgt_speed)
    INITIAL_SPEED       = float(args.init_speed) 
    CHECKPOINT          = args.checkpoint
    ID                  = int(args.id)

    create_exp_folder()
    
    if CHECKPOINT is None:
        RANGE               = float(args.range)
        ACTIVATION          = args.activ_func
        LAST_ACTIVATION     = args.last_activ_func
        NUM_LAYERS          = int(args.num_layers)
        NUM_NODES           = int(args.nodes)
        call_save_architecture()
        
    else:
        loaded = torch.load(f"{CHECKPOINT_PATH}/architecture.pt")
        RANGE               = loaded['range']
        ACTIVATION          = loaded['activation']
        LAST_ACTIVATION     = loaded['last_activ']
        NUM_LAYERS          = loaded['num_layers']
        NUM_NODES           = loaded['num_nodes']

'''
    Create (or not) folder in which models, params and plots will be saved. The figure paths are also defined here
'''
def create_exp_folder():

    global CHECKPOINT_PATH
    global FIGURE_PATH
    global FIGURE_PATH_POS
    global FIGURE_PATH_TORQUES
    global ARCHITECTURE_PATH
    global ID
    global RUN_NUM

    CHECKPOINT_PATH = './results/Exp' # Path where the checkpoints are saved
    test_path = f"{CHECKPOINT_PATH}_{ID}"

    id_plot = 1
    if ID == 0 or ID == -1:
        original_id = ID
        ID = 1
        test_path = f"{CHECKPOINT_PATH}_{ID}"
        while os.path.isdir(test_path):
            ID += 1
            test_path = f"{CHECKPOINT_PATH}_{ID}"
        if original_id == 0:
            os.mkdir(test_path)
        else:
            ID -= 1
            test_path = f"{CHECKPOINT_PATH}_{ID}"

    CHECKPOINT_PATH = test_path
    ARCHITECTURE_PATH = f"{CHECKPOINT_PATH}/architecture.pt"
    FIGURE_PATH = f"{CHECKPOINT_PATH}/reward_loss_{id_plot}.png"
    FIGURE_PATH_POS = f"{CHECKPOINT_PATH}/joint_pos"
    FIGURE_PATH_TORQUES = f"{CHECKPOINT_PATH}/actuators_torques.png"

    while os.path.isfile(FIGURE_PATH):
        id_plot += 1
        FIGURE_PATH = f"{CHECKPOINT_PATH}/reward_loss_{id_plot}.png"

    RUN_NUM = id_plot

'''
    Call save_architecture from networks.py
'''
def call_save_architecture():
    from utilities.networks import save_architecture
    save_architecture()
