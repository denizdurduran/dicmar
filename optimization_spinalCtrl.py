# -*- coding: utf-8 -*

#  ██████╗ ██████╗ ████████╗██╗███╗   ███╗██╗███████╗ █████╗ ████████╗██╗ ██████╗ ███╗   ██╗
# ██╔═══██╗██╔══██╗╚══██╔══╝██║████╗ ████║██║╚══███╔╝██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║
# ██║   ██║██████╔╝   ██║   ██║██╔████╔██║██║  ███╔╝ ███████║   ██║   ██║██║   ██║██╔██╗ ██║
# ██║   ██║██╔═══╝    ██║   ██║██║╚██╔╝██║██║ ███╔╝  ██╔══██║   ██║   ██║██║   ██║██║╚██╗██║
# ╚██████╔╝██║        ██║   ██║██║ ╚═╝ ██║██║███████╗██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║
#  ╚═════╝ ╚═╝        ╚═╝   ╚═╝╚═╝     ╚═╝╚═╝╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
# with deap.

import sys
from control.osim import L2M2019Env
from control.osim_loco_reflex_song2019 import OsimReflexCtrl
from src.cpg import PhaseManager, CpgCtrl
import torch
import numpy as np
import argparse
import random
import pickle
from mpi4py import MPI
import time
import copy
import os

# MPI Related constant
MASTER              = 0                     # MPI Master node
comm = MPI.COMM_WORLD
size = comm.Get_size() # new: gives number of ranks in comm
rank = comm.Get_rank()
# Feedback controller related variable
FBCtrl = None
env    = None
# CPG controller related variable 
FFLeftLeg          = None
FFRightLeg         = None
FFClockLeftLeg     = None
FFClockRightLeg    = None
FFPatternGenerator = None

# General environment constant 
difficulty = 0
sim_dt = 0.01


# Script parameters
parser = argparse.ArgumentParser(prog='Distributed Population Based Optimization (With MPI)')
parser.add_argument("-t",       "--type",            help="[OPT] The type of optimization can be [CMAES]", default="CMAES")
parser.add_argument("-g",       "--ngen",            help="[OPT] Number of generation", default=100)
parser.add_argument("-s",       "--seed",            help="[OPT] Random seed", default=64)
parser.add_argument("-n",       "--n-individual-fb", help="[OPT] The number of individual for an fb optimization", default=2)
parser.add_argument("-cpg",     "--enable-cpg",      help="[OPT] Enable the CPG parameter optimization", action='store_true')
parser.add_argument("-no-fdb",  "--disable-fdb",     help="[OPT] Disables the FDB parameter optimization", action='store_true')
parser.add_argument("-sig",     "--sigma-fb",        help="[OPT] Sigma parameter of the CMAES optimization", default=5.0)
parser.add_argument("-freq",    "--init_frequency",  help="[OPT] Initial frequency of CPG", default=1.0)
parser.add_argument("-fs",      "--force-sigma",     help="[OPT] Set the value of sigma even if present in the checkpoint", action='store_true')
parser.add_argument("-mu",      "--cmaes-mu",        help="[OPT] ", default=None)
parser.add_argument("-c",       "--checkpoint",      help="Checkpoint to use for initial parameters", default="")
parser.add_argument("-f",       "--file",            help="Text file to use for initial parameters, checkpoint takes precendency on this parameter", default=None)
parser.add_argument("-txt",   "--ff_txt_file",       help="Textfile to use for the initial feedforward parameters, checkpoint takes precedency on this parameter", default=None)
parser.add_argument("-mode",    "--mode",            help="2D or 3D optimization", default="2D")
parser.add_argument("-v",       "--visualize",       help="Whether to visualize the results or not, if used during optimization only the first individual is visualized", action='store_true')
parser.add_argument("-duration", "--duration",       help="Maximum duration of the simulation", default=10.0)
parser.add_argument("-MP",     "--motor_primitives", help="Number of motor primitives to use for the simulation", default=4)
parser.add_argument("-tgt_speed","--tgt_speed",      help="Target/Desired speed", default=1.4)
parser.add_argument("-init_speed", "--init_speed",   help="Initial speed of the simulation", default=1.7)
parser.add_argument("-repeat",  "--repeat",          help="Maximum repeat during testing", default=1)
parser.add_argument("-test",    "--test",            help="Testing mode", action='store_true')
parser.add_argument("-debug",   "--debug",           help="Enable debug mode", action='store_true')
parser.add_argument("-add-balance-to-cpg",   "--add-balance-to-cpg",           help="Add balance to cpg", action='store_true')
''' CPG Parameters based on mimicking reflex parameters : 
    
    A) Learn a basic CPG 
        This cpg works without optimization. 
        The step to get it working are 
        1) Extract the CPG parameters : launch the optimization_spinalCtrl.py with the --cpg-mimic-rfx --cpg-mimic-rfx-learn parameters on a working reflex gait 
            e.g. python optimization_spinalCtrl.py -t CMAES --duration=10.0 --file control/params_2D.txt --cpg-mimick-reflex --cpg-mimick-learn --visualize
            This will save a file called cpg_single_50Knot.pkl in the '/data/cpg' folder. You can change the name by providing the --cpg-mimick-reflex-file parameter.
        2) You can then test if the cpg works by launching with the --cpg-mimic-rfx and the --cpg-mimic-rfx-file 
            e.g. python optimization_spinalCtrl.py -t CMAES --duration=10.0 --file control/params_2D.txt --cpg-mimick-reflex --cpg-mimick-reflex-file data/cpg/cpg_single_50Knot.pkl --visualize
    B) Try it with motor primitive extraction

        1) Open the notebook NMF_Generator.ipynb. This allows you this will load a cpg-mimic-rfx-file and launch the script 
        2) This will generate a new mimic-rfx-file with the number of MP 
        3) Relaunch the optimization_spinalCtrl giving now this file instead of the one extracted in part A)
    
'''
parser.add_argument("-cpg-mimick-rfx"     ,"--cpg-mimick-reflex"      ,   help="[CPG Reflex mimicking] Activate the second type of cpg, the one that mimics the reflex pathways", action='store_true')
parser.add_argument("-cpg-mimick-rfx-lrn" ,"--cpg-mimick-reflex-learn",   help="[CPG Reflex mimicking] Learning mode", action='store_true')
parser.add_argument("-cpg-mimick-rfx-file","--cpg-mimick-reflex-file",    help="[CPG Reflex mimicking] Path to the file created with the --cpg-mimic-reflex-learn flag", default="./data/cpg/cpg_single_50Knot.pkl")
parser.add_argument("-k",                 "--cpg-mimick-reflex-knot",     help="[CPG Reflex mimicking] Number of Knot to be used for the learning, default is good", default=50)
# Variable used to store the cpg mimicking reflex controller

cpg_ctrl_mimick = None

#muscle_actuations = []
#muscle_actuations = np.load("muscle_actuations.npy")
#import ipdb; ipdb.set_trace()

args = parser.parse_args()

MODE                = args.mode
DEBUG = args.debug
TEST                = True if size == 1 else args.test
VISUALIZE           = False if(size > 1) else args.visualize
SIM_T               = float(args.duration)
TIMESTEP_LIMIT      = int(round(SIM_T/sim_dt))
OPTIMIZATION_TYPE   = args.type             # Optimization type
LOG_FREQ            = 1                     # Log every LOG_FREQ generation
INIT_FREQUENCY      = float(args.init_frequency)
SEED                = int(args.seed)+rank   # Random seed
NGEN                = int(args.ngen)        # Number of generation
FB_N                = int(args.n_individual_fb) # Number of individual
FB_DIM              = 46                    # Problem dimension
LEARNING_MODE       = "fb"
REPEAT              = int(args.repeat)
FF_TXT_FILE         = args.ff_txt_file
DESIRED_SPEED       = float(args.tgt_speed)       # Desired speed (= Target speed)
INITIAL_SPEED       = float(args.init_speed)      # Initial speed of the simulation 
MOTOR_PRIMITIVES    = int(args.motor_primitives)  # Number of motor primitives
ENABLE_CPG          = args.enable_cpg
DISABLE_FDB         = args.disable_fdb
''' CPG Parameters based on mimicking reflex parameters : 
'''
CPG_MIMICK_RFX          = args.cpg_mimick_reflex
CPG_MIMICK_RFX_LEARN    = args.cpg_mimick_reflex_learn
CPG_MIMICK_RFX_FILE     = args.cpg_mimick_reflex_file
KNOT                    = args.cpg_mimick_reflex_knot
'''
Write parameters to the optimization folder during optimisation so that we know what we were doing in that experiment
'''
CHECKPOINT_PATH = './logs/cmaes_spinal' # Path where the checkpoints are saved
ID = 1
if rank is MASTER and not TEST:
    if os.path.isdir(CHECKPOINT_PATH) == True:
        test_path = "{}_{}".format(CHECKPOINT_PATH,ID)
    while os.path.isdir(test_path):
        ID = ID+1
        test_path = "{}_{}".format(CHECKPOINT_PATH,ID)
    os.mkdir(test_path)
    CHECKPOINT_PATH = test_path
    with open("{}/params.txt".format(CHECKPOINT_PATH), 'w') as f:
        params_summary_txt = '''
MODE {}
TEST {}
VISUALIZE {}
SIM_T {}
TIMESTEP_LIMIT {}
OPTIMIZATION_TYPE {}
LOG_FREQ {}
INIT_FREQUENCY {}
SEED {}
NGEN {}
FB_N {}
FB_DIM {}
LEARNING_MODE {}
REPEAT {}
DESIRED_SPEED {}
INITIAL_SPEED {}
MOTOR_PRIMITIVES {}
ENABLE_CPG {}
DISABLE_FDB {}'''.format(MODE,TEST,VISUALIZE,SIM_T,TIMESTEP_LIMIT,OPTIMIZATION_TYPE,LOG_FREQ,INIT_FREQUENCY,SEED,NGEN,FB_N,FB_DIM,LEARNING_MODE,REPEAT,DESIRED_SPEED,INITIAL_SPEED,MOTOR_PRIMITIVES, ENABLE_CPG,DISABLE_FDB)
        f.write(params_summary_txt)



#######################################################################
# MPI Tags
# 0:NP        : sending individual to worker
# NP+1:2NP    : sending fitness    to master
NP = size      # NP >= N : but NP > N => improvement in speed.
               # N  : Number of individual needed to pass to next generation
               # NP : Number of processes
#######################################################################
random.seed(SEED)
np.random.seed(SEED)
from src.cpg import Clock, Leg
from RBF_NN import Learning_network
sys.path.insert(0,'./lib')
import deap_mpi

'''
    Optimization parameter initalization.
    ==============================

    Here we initialize the parameter spaces. It is defined as a set of two arrays
    the first array defines the lower bound and the second array the upper bound.

'''

PAR_SPACE = None

# FEEDBACK PARAM SPACE 
#
# The space is defined for the 3D model, and a reduced set is created for the 2D model by dropping the last elements of the array.
#
#
FB_PAR_SPACE_3D = (
    [0.0, -1.0, 0.0, 0.0, \
    -2.0, -90/15, -1.0, 0.0, \
    0.0, 0.0, 0.0, 0.0, 0.0, \
    0.0, 0.0, 0.0, 0.0, 0.0, \
    0.0, 0.0, \
    0.0, 0.0, \
    0.0, 0.0, 0.0, \
    0.0, 0.0, 0.0, 0.0, 0.0, \
    0.0, 0.0, 0.0, \
    0.0, \
    0.0, \
    0.0, 0.0, \
    0.0, 0.0, 0.0, \
    0.0, 0.0, 0.0, \
    0.0, 0.0, 0.0],
    [6.0, 3.0, 5.0, 3.0, \
    3.0, 20/15, 15/10, 3.0, \
    3.0, 3.0, 3.0, 3.0, 3.0, \
    3.0, 3.0, 3.0, 3.0, 3.0, \
    3.0, 3.0, \
    3.0, 3.0, \
    3.0, 3.0, 3.0, \
    3.0, 3.0, 3.0, 3.0, 3.0, \
    3.0, 3.0, 3.0, \
    3.0, \
    3.0, \
    3.0, 3.0, \
    2.0, 3.0, 3.0, \
    3.0, 3.0, 3.0, \
    3.0, 3.0, 3.0])

fb_params_2D = np.loadtxt('./control/params_2D.txt')
fb_params_3D_init = np.loadtxt('./control/params_3D_init.txt')
fb_N_2D = len(fb_params_2D)
fb_N_3D = len(fb_params_3D_init)


# CPG PARAM SPACE 
#
#

# Pattern formation init
n_motor_primitives = MOTOR_PRIMITIVES
n_outputs          = 9
FFPatternGenerator = Learning_network(1, n_outputs, num_rbf=n_motor_primitives)

FF_PAR_SPACE = FFPatternGenerator.generate_cpg_param_range()

if FF_TXT_FILE is None: 
    FF_TXT_FILE = "./control/params_2D_CPG_{}MP.txt".format(MOTOR_PRIMITIVES)
ff_params_2D = np.loadtxt(FF_TXT_FILE)
ff_N_2D = len(ff_params_2D)

'''
    Environment initialisation
    ==============================

    The environment is defined only for the workers (e.g. when rank is not MASTER or when running TEST on a optimized controller)
'''

is_worker = lambda: rank is not MASTER or TEST 

if is_worker():
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

    # Cpg initialization 

    # Controls the contribution of CPG versus Reflex contribution during the simulation. 
    # The transitions list is used to define the different "section" of the simulation. 
    # During each section you can choose the contribution of cpg vs reflex with the alphas list. 
    # We follow a proximo distal gradient : The get_alphas methods gets as input the contribution of hip / knee and ankle muscles. 
    # e.g.  get_alphas(0.0,0.0,0.0) means 0% cpg in hip / knee and ankle muscles. 
    #       get_alphas(1.0,0.0,0.0) means 100% cpg in hip and 0 % cpg in both ankle and knee muscles.
    #
    #       Muscle list 3D "hab", "had", "hf", "glu", "ham", "rf", "vas", "bhfs", "gas", "sol", "ta"
    #       
    #       hab and had are 3D muscles, they are still present in the 2D mode even though their contribution is not taken into account.
    # 
    # 

    # Generates alpha vector with different weighting for hip knee and ankle muscles. 
    get_alphas = lambda hip, knee, ankle: get_alphas_one_leg(hip,knee,ankle) + get_alphas_one_leg(hip,knee,ankle)
    get_alphas_one_leg = lambda hip,knee,ankle: [hip,hip,hip,hip,knee,knee,knee,knee,ankle,ankle,ankle]

    # Transitions, when the phase ends.
    transitions = [
        lambda t,x=None: t > 0.4,
        lambda t,x=None: t > 10.0,
        lambda t,x=None: t > 10.0,
        lambda t,x=None: False,
    ]
    # Alphas corresponding to each phase. alpha = 0 === 100% reflex, alpha = 1 === 100% cpg.
    alphas = [
        np.array(get_alphas(0.0,0.0,0.0)),
        np.array(get_alphas(1.0,0.5,0.0)),
        np.array(get_alphas(1.0,0.5,0.0)),
        np.array(get_alphas(1.0,0.5,0.0)),
    ]

    phaseManager = PhaseManager(transitions, alphas)
    
    def env_init():
        global env
        if rank is 2 or size is 1:
            env = L2M2019Env(visualize=VISUALIZE, seed=SEED, difficulty=difficulty, desired_speed=DESIRED_SPEED)
        else:
            env = L2M2019Env(visualize=False, seed=SEED, difficulty=difficulty, desired_speed=DESIRED_SPEED)
        env.change_model(model=MODE, difficulty=difficulty, seed=SEED)
        env.spec.timestep_limit = TIMESTEP_LIMIT
        
    '''
    Initialise the feedback controller (Song's reflex controller)
    '''
    def fb_init():
        global FBCtrl
        FBCtrl = OsimReflexCtrl(mode=MODE, dt=sim_dt)
    '''
    Initialise the cpg controller (motor primitives with gaussian processes)
        The controller is derived from the CPG controller developped by Nicolas Feppon

        The CPG controller requires 
            1. A clock for the Left leg 
            2. A clock for the Right leg 
            3. A pattern generation machine (we assume the leg is symmetric) so only one is used.
    '''
    def ff_init():
        global FFLeftLeg, FFRightLeg, FFClockLeftLeg, FFClockRightLeg
        # CPG Parameters and initialization
        theta0             = dict(left=0.6,right=0.1) # The initial phase of left and right leg.
        init_freq          = INIT_FREQUENCY                      # The initial frequency of the CPG, must be close to the gait frequency
        # Leg event init
        FFLeftLeg          = Leg()
        FFRightLeg         = Leg()
        # Clocks init
        FFClockLeftLeg     = Clock(sim_dt,init_freq,theta0=theta0['left'])
        FFClockRightLeg    = Clock(sim_dt,init_freq,theta0=theta0['right'])

    '''
    Initialise the cpg controller (mimicking the reflexes)
        The controller is derived from the CPG controller developped by Florin Dzeladini
    '''
    def ff_init_mimick():
        global CpgCtrl, cpg_ctrl_mimick, CPG_MIMICK_RFX_FILE
        print("CPG Mimicking reflex active")
        theta0             = dict(left=0.6,right=0.1) # The initial phase of left and right leg.
        init_freq          = INIT_FREQUENCY           # The initial frequency of the CPG, must be close to the gait frequency
        CPG_MODE           = "single"
        if(CPG_MIMICK_RFX and CPG_MIMICK_RFX_LEARN):
            print("\tLearning mode")
            cpg_ctrl_mimick = CpgCtrl(
                sim_dt    = sim_dt, 
                init_freq = INIT_FREQUENCY, 
                knot      = KNOT,
                theta0    = theta0,
                cpg_type  = CPG_MODE,
                dim = {
                    'left'  : 11,
                    'right' : 11
                })
        elif(CPG_MIMICK_RFX):
            print("\tTesting mode")
            try:
                with open(CPG_MIMICK_RFX_FILE, "rb") as cpg_file:
                    cpg_ctrl_mimick = pickle.load(cpg_file)
                    cpg_ctrl_mimick.disableLearning()
                    print("Cpg file loaded from {}".format(CPG_MIMICK_RFX_FILE))
            except FileNotFoundError:
                print("CPG File {} not found".format(CPG_MIMICK_RFX_FILE))
                import sys
                sys.exit(1)
            cpg_ctrl_mimick.reset(freq=None, theta0=theta0) # We don't update freq only theta0.


    '''
    Update the feedback controller and outputs cpg contribution to muscle activity, to be done at each time step.
    '''
    def fb_update(obs_dict):
        global FBCtrl
        return FBCtrl.update(obs_dict)
    '''
    Update the cpg controller and outputs cpg contribution to muscle activity, to be done at each time step.

        Process : 
            1. We first update the leg states (detects state edges, e.g. lift off and contact)
            2. Update the clock using the leg state to reset the clock if needed 
            3. Use the phase (from clock.theta) of each clock to get the cpg output
    '''
    def ff_update(obs_dict):
        # 0. Get vertical grf from observation
        left_grf = obs_dict['l_leg']['ground_reaction_forces'][2]
        right_grf = obs_dict['r_leg']['ground_reaction_forces'][2]
        # 1. Update Leg state
        FFLeftLeg.update(left_grf)
        FFRightLeg.update(right_grf)
        # 2. Update clock 
        FFClockLeftLeg.update(FFLeftLeg.contact()-FFLeftLeg.liftoff())
        FFClockRightLeg.update(FFRightLeg.contact()-FFRightLeg.liftoff())
        # 3. Get cpg output
        left_cpg_output  = FFPatternGenerator.forward(torch.tensor([FFClockLeftLeg.theta])).detach().numpy()
        right_cpg_output = FFPatternGenerator.forward(torch.tensor([FFClockRightLeg.theta])).detach().numpy()
        return np.concatenate([2*[0],left_cpg_output[0,0],2*[0],right_cpg_output[0,0]])
    '''
    Update the cpg mimicking reflex controller and outputs cpg contribution to muscle activity, to be done at each time step.

        Process : 
            1. We first update the leg states (detects state edges, e.g. lift off and contact)
            2. Update the clock using the leg state to reset the clock if needed 
            3. Use the phase (from clock.theta) of each clock to get the cpg output
    '''
    def ff_update_mimick(obs_dict,action_mimick):
        global cpg_ctrl_mimick
        # 0. Get vertical grf from observation
        left_grf = obs_dict['l_leg']['ground_reaction_forces'][2]
        right_grf = obs_dict['r_leg']['ground_reaction_forces'][2]
        if CPG_MIMICK_RFX_LEARN:
            return cpg_ctrl_mimick.update(left_grf,right_grf,action_mimick[11:],action_mimick[:11])
        else:
            return cpg_ctrl_mimick.update(left_grf,right_grf)
    '''
    Initialize the feedback and cpg controller
    '''
    def init_ctrl(): # We always initialize both controller even if not both are used
        fb_init()
        ff_init()
        ff_init_mimick()

    '''
    Sets the parameter of the feedback and or cpg controller 

        Here we receive all the parameters of our controllers. 
        
            ONLY FEEDBACK : 46 parameters
            ONLY CPG      : XX parameters 
    '''
    def set_param(x): # TODO
        global FBCtrl, FFPatternGenerator
        # If feedback controller active
        if(not DISABLE_FDB):
            FBCtrl.set_control_params(np.round(x[0:fb_N_2D],4))
        # If cpg controller active
        if(ENABLE_CPG):
            FFParams = x
            if(not DISABLE_FDB): # If the FBCtrl is active we are in a co-optimization state 
                        # we therefore need to skip the first 46 parameters that correspond 
                        # to the  feedback controller
                FFParams = x[fb_N_2D:]
            FFPatternGenerator.set_params_from_array(FFParams)
        return x
    '''
    Returns the action of the controller 

        Note that this get_action does not separate spinal and vestibular contribution. 

        If we separate the contribution from vestibular and spinal, 
        we can test the hypothesis that spinal reflexes can be replaced by feedforward contribution but not vestibular contribution.
        
    '''
    def get_action(obs_dict):
        # Deprecated we calculate the action directly in the 
        # loop because we want to log them at the end of the simulation
        pass

    def dummy_loop(init,set_param,get_action):
        time.sleep(10*np.random.rand())
        x = set_param()
        return np.abs(x).sum(),0,0,0

    def loop(init,set_param,get_action):
        global FBCtrl, TEST, DISABLE_FDB, ENABLE_CPG
        init()
        x=set_param()
        total_reward = 0
        #total_reward_footstep_0 = 0
        #total_reward_footstep_v = 0
        #total_reward_footstep_e = 0
        t = 0
        i = 0
        obs_dict = env.reset(project=True, seed=SEED, obs_as_dict=True, init_pose=INIT_POSE)
        val = env.get_observation_dict()
        obs = env.get_observation()
        elapsed_total = 0
        alpha = np.random.rand()*0.8
        _total_obs_sum = 0 # For reproducibility
        # Create variable for logging at the beginning of the loop
        cpgLog = []
        rfxLog = []
        balanceLog = []
        musLog = []
        values = []
        velocity = []
        noiseLog = []

        while True:
            #input("next")
            _total_obs_sum += sum(obs)
            i += 1
            t += sim_dt
            _t = time.time()
            speed = obs[245]
            mean = 0
            std = 0.1
            num_samples = 22
            white_noise = np.random.normal(mean, std, size=num_samples)
            # Helper function to get stimulation type
            getStimByType = lambda x,b: np.array(FBCtrl._stimdict2array(x,balance=b))
            # Extract action reflex type
            action_fdb = fb_update(obs_dict)
            # Extract 
            action_fdb_balance = getStimByType(FBCtrl.ctrl.balance_stim,True)
            action_fdb_spinal  = getStimByType(FBCtrl.ctrl.reflex_stim,False)
            action_fdb_sum     = getStimByType(FBCtrl.ctrl.stim,True)
            # Extract cpg action 
            if ENABLE_CPG:
                action_cpg = ff_update(obs_dict)
            elif CPG_MIMICK_RFX:
                action_cpg = ff_update_mimick(obs_dict,action_fdb_spinal)
            else:
                action_cpg = 0*action_fdb

            # Calculate the total action 

            action = None
            action_noise = None
            if CPG_MIMICK_RFX:
                if CPG_MIMICK_RFX_LEARN:
                    action = action_fdb
                else:
                    action_spinal,_ = phaseManager.update(t,action_fdb_spinal,action_cpg)
                    action = np.clip(action_spinal + action_fdb_balance,0.01,1)
            else:
                DISABLE_FDB == False
                ENABLE_CPG == False

                BOTH_FDB_CPG = not DISABLE_FDB and ENABLE_CPG
                ONLY_FDB = not DISABLE_FDB and not ENABLE_CPG 
                ONLY_CPG = DISABLE_FDB and ENABLE_CPG
                if BOTH_FDB_CPG: 
                    action = action_fdb + action_cpg
                    #action_noise = action_fdb_spinal + action_fdb_spinal*white_noise + action_fdb_balance + action_fdb_balance * white_noise + action_cpg
                elif ONLY_FDB:
                    action = action_fdb
                    #action_noise = action_fdb_spinal + action_fdb_spinal*white_noise + action_fdb_balance + action_fdb_balance * white_noise
                elif ONLY_CPG:
                    if args.add_balance_to_cpg:
                        # Get balance component 
                        action_fdb_balance = getStimByType(FBCtrl.ctrl.balance_stim,True)
                        action_spinal,_ = phaseManager.update(t,action_fdb_spinal,action_cpg)
                        action = np.clip(action_spinal + action_fdb_balance,0.01,1)     
                    else:
                        action = phaseManager.update(t,action_fdb,action_cpg)
                else:
                    raise ValueError('No controller are active. This is a bug contact the maintainer of the code.')
            
            #obs_dict, reward, reward_footstep_0, reward_footstep_v, reward_footstep_e, done, info = env.step(np.array(action), project = True, obs_as_dict=True)
            exo_actuation = np.zeros(6)
            exo_actuation[5] = 10
            #input(action)
            #muscle_actuations.append(action)
            action = np.concatenate((np.array(action), exo_actuation))
            #action = muscle_actuations[i-1]
#           obs_dict, reward, reward_footstep_0, reward_footstep_v, reward_footstep_e, done, info = env.step(np.array(action), project = True, obs_as_dict=True)
            #obs_dict, reward, done, info = env.step(np.array(action), project = True, obs_as_dict=True)
            obs_dict, reward, done, info = env.step(action, project = True, obs_as_dict=True)

            if TEST: # Log only when testing 
                cpgLog.append(action_cpg)           # append cpg contribution to motoneuron (before linear combination of reflex + cpg)
                rfxLog.append(action_fdb_spinal)    # append reflex contribution to motoneuron (before linear combination of reflex + cpg)
                balanceLog.append(action_fdb_balance)    # append reflex contribution to motoneuron (before linear combination of reflex + cpg)
                values.append(val)
                noiseLog.append(action_noise)
                musLog.append(action)               # append motoneuron activity (Registers the muscle activations)
                velocity.append(speed)
                mean_speed = np.mean(velocity)
            else:
                mean_speed = 0.0

            obs = env.get_observation()
            val = env.get_observation_dict()
            elapsed_total += time.time() - _t
            total_reward += reward

            if(done):
                if(CPG_MIMICK_RFX and CPG_MIMICK_RFX_LEARN):
                    with open(CPG_MIMICK_RFX_FILE, "wb") as cpg_file:
                        pickle.dump(cpg_ctrl_mimick, cpg_file)
                        print("Cpg file saved into {}".format(CPG_MIMICK_RFX_FILE))
                break
        if DEBUG: print('[WORKER {}] score={} time={}sec, realdt={}s total_obs_sum={} speed={}m/s, i={}, mean_speed={}m/s'.format(rank,total_reward, t, elapsed_total/i,_total_obs_sum, speed, i, mean_speed))

        if DEBUG: sys.stdout.flush()
        # Save the log to the checkpoint folder but only when testing
        if TEST:
            SAVE_PATH = './logs/simulation_data/'
            if(args.checkpoint == ""):
                SAVE_PATH += args.file.split('/')[-1].split('.')[0] + ".pkl"
            else:
                split = args.checkpoint.split('/')
                if(len(split) >= 2):
                    SAVE_PATH += split[-2] + split[-1]
                else:
                    SAVE_PATH += split[-1]
            
            print("Saving logs to {}_[cpg|muscle|reflex|balance|values|noise].pkl".format(SAVE_PATH.split('.pkl')[0]))
            cpgPath = SAVE_PATH.replace('.pkl', '_cpg.pkl')
            musclePath = SAVE_PATH.replace('.pkl','_muscle.pkl')
            reflexPath = SAVE_PATH.replace('.pkl', '_reflex.pkl')
            balancePath = SAVE_PATH.replace('.pkl', '_balance.pkl')
            valPath = SAVE_PATH.replace('.pkl', '_values.pkl')
            noisePath = SAVE_PATH.replace('.pkl', '_noise.pkl')
            pickle.dump(cpgLog, open(cpgPath, "wb"))
            pickle.dump(musLog, open(musclePath, "wb"))
            pickle.dump(rfxLog, open(reflexPath, "wb"))
            pickle.dump(balanceLog, open(balancePath, "wb"))
            pickle.dump(values, open(valPath, "wb"))
            pickle.dump(noiseLog, open(noisePath, 'wb'))

        print('    score={} time={}sec, realdt={}s speed={}m/s, i={}'.format(total_reward, t, elapsed_total/i, speed, i))
        #np.save("muscle_actuations", muscle_actuations)

        return total_reward, t,elapsed_total/i,env.pose[0], speed, i


    def F(x):
        return loop(init_ctrl,lambda: set_param(x),lambda obs_dict,_: get_action(obs_dict))

def get_stats():
    stats = deap_mpi.tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", lambda x: np.round(np.mean(x),2))
    stats.register("std", lambda x: np.round(np.std(x),2))
    stats.register("min", lambda x: np.round(np.min(x),2))
    stats.register("max", lambda x: np.round(np.max(x),2))

    stats_duration = deap_mpi.tools.Statistics(lambda ind: ind.fitness.values[1])
    stats_duration.register("dur", np.max)

    deap_mpi.creator.create("FitnessMax", deap_mpi.base.Fitness, weights=(1.0, 0.0001, 0.0001))
    deap_mpi.creator.create("Individual", list, fitness=deap_mpi.creator.FitnessMax)
    return stats, stats_duration


if rank is MASTER or TEST:
    #  ███▄ ▄███▓ ▄▄▄        ██████ ▄▄▄█████▓▓█████  ██▀███
    # ▓██▒▀█▀ ██▒▒████▄    ▒██    ▒ ▓  ██▒ ▓▒▓█   ▀ ▓██ ▒ ██▒
    # ▓██    ▓██░▒██  ▀█▄  ░ ▓██▄   ▒ ▓██░ ▒░▒███   ▓██ ░▄█ ▒
    # ▒██    ▒██ ░██▄▄▄▄██   ▒   ██▒░ ▓██▓ ░ ▒▓█  ▄ ▒██▀▀█▄
    # ▒██▒   ░██▒ ▓█   ▓██▒▒██████▒▒  ▒██▒ ░ ░▒████▒░██▓ ▒██▒
    # ░ ▒░   ░  ░ ▒▒   ▓▒█░▒ ▒▓▒ ▒ ░  ▒ ░░   ░░ ▒░ ░░ ▒▓ ░▒▓░
    # ░  ░      ░  ▒   ▒▒ ░░ ░▒  ░ ░    ░     ░ ░  ░  ░▒ ░ ▒░
    # ░      ░     ░   ▒   ░  ░  ░    ░         ░     ░░   ░
    #        ░         ░  ░      ░              ░  ░   ░
    #
    # Initialization of FB params
    if MODE is '2D':
        init_params = fb_params_2D
        if(ENABLE_CPG and not DISABLE_FDB):
            init_params = np.concatenate([fb_params_2D,ff_params_2D])
        if(ENABLE_CPG):
            init_params = ff_params_2D
    elif MODE is '3D':
        init_params = fb_params_3D_init
        # Uncomment to reduce ranges for 2D related parameters. This enforces the parameters to be close to the 2D one.
        # FB_PAR_SPACE_3D[0][0:fb_N_2D] = cp['best_ind'][0:fb_N_2D] - 0.1*np.abs(fb_params_2D)
        # FB_PAR_SPACE_3D[1][0:fb_N_2D] = cp['best_ind'][0:fb_N_2D] + 0.1*np.abs(fb_params_2D)
    if(args.file):
        init_params = np.loadtxt('{}'.format(args.file))

    stats, stats_duration = get_stats()
    # Initialize checkpoint storage.
    cp = {}
    # Load checkpoint from file
    if(args.checkpoint):
        with open("{}".format(args.checkpoint), "rb") as cp_file:
            cp = pickle.load(cp_file)
        #random.setstate(cp["rndstate"])
        if(args.force_sigma):
            cp["sigma"] = float(args.sigma_fb)
        print("best fitness : {}".format(cp["best_fitness"]))
        print("sigma        : {}".format(cp["sigma"]))
    # Initialize checkpoint from initial parameters
    else:
        cp = {
            'best_fitness'      : 0,
            'best_ind'          : init_params,
            'sigma'             : float(args.sigma_fb),
            'start_gen'         : 0
        }



    # [RFX PARAM] scaling 
    from control.locoCtrl_balance_reflex_separated import LocoCtrl

    get_idx_of_rfx_params = lambda l: list(np.where(np.array([k if i in l else -1 for k,i in enumerate(LocoCtrl.cp_keys)]) != -1))[0]
    # Example to get the idx of parameters associated 'GLU_3_PG','HFL_3_PG'
    idx_test = get_idx_of_rfx_params(['GLU_3_PG','HFL_3_PG']).tolist()

    if MODE is '2D':
        PAR_SPACE = (FB_PAR_SPACE_3D[0][0:fb_N_2D],FB_PAR_SPACE_3D[1][0:fb_N_2D])
        if(ENABLE_CPG and not DISABLE_FDB):
            PAR_SPACE = ( PAR_SPACE[0] + FF_PAR_SPACE[0],
                          PAR_SPACE[1] + FF_PAR_SPACE[1] )
        elif(ENABLE_CPG):
            PAR_SPACE = FF_PAR_SPACE
    else:
        # CPG Model not yet compatible 
        if ENABLE_CPG:
            raise ValueError("CPG Model not yet compatible with 3D learning, please remove --enable-cpg parameters and start again or switch to 2D")
        PAR_SPACE = FB_PAR_SPACE_3D

    deap_mpi.init(
        debug = DEBUG,
        par_space = PAR_SPACE)

    deap_mpi.printBig(OPTIMIZATION_TYPE)


if rank is MASTER and not TEST:
    cp["logbook"] = deap_mpi.tools.Logbook()
    cp["halloffame"] = deap_mpi.tools.HallOfFame(1)
    cp["logbook"].header    = ["gen", "evals", "sigma", "distance"] + stats.fields + stats_duration.fields

    sys.stdout.flush()
    toolbox = {}
    ga_tlbx, = deap_mpi.initGA(
            tournsize     = 3,
            indpb         = 0.05,
            cxpb          = 0.5,
            mutpb         = 0.8
        )
    toolbox['GA'] = ga_tlbx

    fb_cmaes_tlbx, fb_cmaes_strategy = deap_mpi.initCMAES(
        best_ind      = deap_mpi.unscale_fb(cp["best_ind"]),
        sigma         = cp["sigma"],
        N             = FB_N,
        mu            = int(args.cmaes_mu) if args.cmaes_mu else None
    )
    toolbox['CMAES_FB'] = fb_cmaes_tlbx
    best_fitness = cp["best_fitness"];
    best_ind = deap_mpi.unscale_fb(cp["best_ind"])


    generators = deap_mpi.getGenerators(toolbox);
    start_gen = cp["start_gen"]
    for g in range(start_gen+1, NGEN):
        beta = 0
        #       _ _           _             _
        #    __| (_)___ _ __ | |_ __ _  ___| |__
        #   / _` | / __| '_ \| __/ _` |/ __| '_ \
        #   \__,_|_|___/ .__/ \__\__,_|\___|_| |_|
        #              |_|
        if(g == start_gen+1):                _offsprings, _fitnesses, _sim_durations, _real_durations, _distances = deap_mpi.dispatcher(FB_N, lambda: generators['CMAES_FB']())
        elif( OPTIMIZATION_TYPE == "GA"):    _offsprings, _fitnesses, _sim_durations, _real_durations, _distances = deap_mpi.dispatcher(FB_N, lambda: generators['GA'](_offsprings))
        elif( OPTIMIZATION_TYPE == "CMAES"): _offsprings, _fitnesses, _sim_durations, _real_durations, _distances = deap_mpi.dispatcher(FB_N, lambda: generators['CMAES_FB']())
        else:                                print("Error OPTIMIZATION_TYPE={} unknown ".format(OPTIMIZATION_TYPE))

        if DEBUG: print("[MASTER] finished dispatch")
        #
        #   _   _ _ __   __| | __ _| |_ ___
        #  | | | | '_ \ / _` |/ _` | __/ _ \
        #   \__,_| .__/ \__,_|\__,_|\__\___| the next generation with the results.
        #        |_|
        if( OPTIMIZATION_TYPE == "GA" ):     _offsprings = toolbox["GA"].select(_offsprings, k=N)
        if( OPTIMIZATION_TYPE == "CMAES" ):  toolbox["CMAES_FB"].update(copy.deepcopy(_offsprings))
        if DEBUG: print("[MASTER] finished update")
        #
        #   ___| |_ __ _| |_ ___
        #  / __| __/ _` | __/ __|
        #  \__ \ || (_| | |_\__ \
        #  |___/\__\__,_|\__|___/
        #
        # this is done only for logging. The actual selection of individual is done in the update above.
        if DEBUG: print("[MASTER] logging stats")
        best_idx = np.argmax(_fitnesses)
        if(_fitnesses[best_idx] > best_fitness):
            best_duration = _sim_durations[best_idx]
            best_fitness  = np.max(_fitnesses)
            best_ind      = [i for i in _offsprings[best_idx]]

        
        _l_best_distance = _distances[best_idx]
        _l_best_fitness  = _fitnesses[best_idx]
        _l_best_fitness  = _fitnesses[best_idx]
        _l_best_duration = _sim_durations[best_idx]
        cp["logbook"].record(
            type         = LEARNING_MODE,
            gen          = f"{str(g).zfill(4)}/{NGEN}",
            evals        = len(_offsprings),
            sigma        = np.round(fb_cmaes_strategy.sigma,2),
            distance     = np.round(_l_best_distance,2),
            **stats.compile(_offsprings),
            **stats_duration.compile(_offsprings),
        )

        print(cp["logbook"].stream)
        sys.stdout.flush()
        # Saving checkpoints
        if g % LOG_FREQ == 0:
            # Fill the dictionary using the dict(key=value[, ...]) constructor
            cp = dict(
                best_ind         = deap_mpi.scale_fb(best_ind),
                best_fitness     = best_fitness,
                sigma            = fb_cmaes_strategy.sigma,
                start_gen        = g,
                halloffame       = cp["halloffame"],
                logbook          = cp["logbook"], 
                rndstate         = random.getstate()
                )


            with open("{}/{}_{}.pkl".format(CHECKPOINT_PATH, ''.join(args.checkpoint.split('_')[0:-1]),g), "wb") as cp_file:
                pickle.dump(cp, cp_file)

            #top10 = deap_mpi.tools.selBest(_offsprings, k=10)

if rank is not MASTER or TEST:
    #   ██████  ██▓    ▄▄▄    ██▒   █▓▓█████   ██████
    # ▒██    ▒ ▓██▒   ▒████▄ ▓██░   █▒▓█   ▀ ▒██    ▒
    # ░ ▓██▄   ▒██░   ▒██  ▀█▄▓██  █▒░▒███   ░ ▓██▄
    #   ▒   ██▒▒██░   ░██▄▄▄▄██▒██ █░░▒▓█  ▄   ▒   ██▒
    # ▒██████▒▒░██████▒▓█   ▓██▒▒▀█░  ░▒████▒▒██████▒▒
    # ▒ ▒▓▒ ▒ ░░ ▒░▓  ░▒▒   ▓▒█░░ ▐░  ░░ ▒░ ░▒ ▒▓▒ ▒ ░
    # ░ ░▒  ░ ░░ ░ ▒  ░ ▒   ▒▒ ░░ ░░   ░ ░  ░░ ░▒  ░ ░
    # ░  ░  ░    ░ ░    ░   ▒     ░░     ░   ░  ░  ░
    #       ░      ░  ░     ░  ░   ░     ░  ░      ░
    #                             ░
    env_init()
    if TEST: 
        for i in range(REPEAT):
            print("")
            print("")
            print("")
            print("The following parameters have been loaded:")
            print(cp['best_ind'])
            individual = cp['best_ind']
            ret = F(individual)
            print("rewards: {}".format(ret[0]))
    else:
        while True:
            if(comm.Iprobe(source=0, tag=rank)): 
                individual,best,beta = comm.recv(source=0, tag=rank)
                if DEBUG: print("[WORKER {}] received individual, doing some work".format(rank))
                sys.stdout.flush()
                to_send = F(individual)
                comm.send(to_send, dest=0, tag=NP+rank)
                if DEBUG: print("[WORKER {}] done, fit={}".format(rank,to_send[0]))
                if DEBUG: sys.stdout.flush()
            time.sleep(0.001)
