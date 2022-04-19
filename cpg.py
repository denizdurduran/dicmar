import numpy as np
import argparse
import pickle
import random
import time
import control.osim as env 

from control.locoCtrl_balance_reflex_separated import LocoCtrl
from control.osim_loco_reflex_song2019 import OsimReflexCtrl
from src.cpg import Cpg, Clock, Leg, Heading, CpgCtrl, PhaseManager

import sys
import select
# Methods to control the simulation with the keyboard. 
# This methods is called whenever the user press enter. The current line is read from stdin and can be used 
# to change the parameters of the cpg. In this case we change the frequency using the cpg_ctrl.set_frequency methods.
def heardEnter(cpg_ctrl):
    i,o,e = select.select([sys.stdin],[],[],0.0001)
    for s in i:
        if s == sys.stdin:
            try:
                input = sys.stdin.readline().strip()
                cpg_ctrl.set_frequency(float(input))
                return True
            except ValueError:
                print("Number cannot be parsed")
    return False


L2M2019Env = env.L2M2019Env
mode = '2D'
difficulty = 0
visualize=True
sim_dt = 0.01

# Parameter parser 
#
#
parser = argparse.ArgumentParser(prog='Distributed Population Based Optimization (With MPI)')
parser.add_argument("-n",   "--n-invidiual",                 help="", default=2)
parser.add_argument("-c",   "--checkpoint",                  help="Reflex checkpoint to load", default="")
parser.add_argument("-s",   "--seed",                        help="", default=64)
parser.add_argument("-k",   "--knot",                        help="Number of Knot", default=50)
parser.add_argument("-type","--type",                        help="Type of sensor source used for learning, default spinal, possible choice [spinal,balance,total]", default="spinal")
parser.add_argument("-cpg-mode","--cpg-mode",                help="Type of cpg, can be [single, half_center]", default="single")
parser.add_argument("-test", "--test",                       help="", action="store_true")
parser.add_argument("-duration","--duration",                help="Maximum duration of the simulation", default=10.0)

args                = parser.parse_args()
#

MAX_DURATION        = float(args.duration)
timstep_limit       = int(round(MAX_DURATION/sim_dt))
TEST_MODE           = args.test
KNOT                = int(args.knot)
TYPE                = args.type
CPG_MODE            = args.cpg_mode
SEED                = int(args.seed)        # Random seed 
FILE_PATH           = "./data/cpg/cpg_{}_{}_{}_{}Knot_5MP.pkl".format(mode,TYPE,CPG_MODE,KNOT) # With 5MP  (see NMFTest.ipynb jupyter notebook for details)
#FILE_PATH           = "./data/cpg/cpg_{}_{}_{}_{}Knot.pkl".format(mode,TYPE,CPG_MODE,KNOT) # Without MP
np.random.seed(SEED)

# Musculoskeletal model initial condition 
#
#
INIT_POSE_STILL = np.array([
    0.0,  #1.699999999999999956e+00,   # forward speed
    0.0,  #.5,                         # rightward speed
    0.94, #9.023245653983965608e-01,   # pelvis height
    0.0,  #2.012303881285582852e-01,   # trunk lean
    0.0,  #0*np.pi/180,                # [right] hip adduct
    0.0,  #-6.952390849304798115e-01,  # hip flex
    0.0,  #-3.231075259785813891e-01,  # knee extend
    0.0,  #1.709011708233401095e-01,   # ankle flex
    0.0,  #0*np.pi/180,                # [left] hip adduct
    0.0,  #-5.282323914341899296e-02,  # hip flex
    0.0,  #-8.041966456860847323e-01,  # knee extend
    0.0   #-1.745329251994329478e-01   # ankle flex
    ])

INIT_POSE_DYNAMIC = np.array([
    1.699999999999999956e+00,   # forward speed
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
    -1.745329251994329478e-01   # ankle flex
    ])


INIT_POSE = INIT_POSE_DYNAMIC

# Cpg initialization 
#
#

# 1. cpg / reflex contribution management 
#
#
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
    np.array(get_alphas(1.0,1.0,1.0)),
    np.array(get_alphas(1.0,1.0,1.0)),
    np.array(get_alphas(1.0,1.0,1.0)),
]
theta0        = dict(left=0.6,right=0.1)

# 2. cpg initialization 
#######################
# Testing mode (loads existing CPG)
if(TEST_MODE):
    try:
        with open(FILE_PATH, "rb") as cpg_file:
            cpg_ctrl = pickle.load(cpg_file)
            cpg_ctrl.disableLearning()
            if(CPG_MODE == "half_center"):
                cpg_ctrl.left_cpg.cpg.clock.switch = 0.45 # Parameter testing for cpg half_center.
                cpg_ctrl.right_cpg.cpg.clock.switch = 0.45 # Parameter testing for cpg half_center.
                cpg_ctrl.reset(freq=0.1) # We update only freq
                import ipdb; ipdb.set_trace()
            else:
                N_muscles = 11
                get_ff_params = lambda N,shift=0,scale=1: np.concatenate([shift+np.zeros(N),scale*np.ones(N)])
                # [CPG Param] : Here you can set the shift and scale for the different muscles output of the CPG 
                #               If shift and scale are floating points then the same shift and scale is applied to all muscles. 
                #               If you want to shift or scale only some of the signals you can use a vector for shift or scale of dimension N_Muscles.
                # cpg_ctrl.set_control_params(get_ff_params(11,-0.05,1.0))

                # [CPG Param] : Frequency change of the cpg. 
                #               To check frequency learned you can check cpg_ctrl.left_cpg.cpg.clock.freq
                # cpg_ctrl.reset(freq=1) # We update only freq

            print("Cpg file loaded from {}".format(FILE_PATH))
    except FileNotFoundError:
        print("CPG File {} not found".format(FILE_PATH))
        import sys
        sys.exit(1)
    cpg_ctrl.reset(freq=None, theta0=theta0) # We don't update freq only theta0.
# Learning mobe (initialize CPG)
else:
    cpg_ctrl = CpgCtrl(
        sim_dt    = sim_dt, 
        init_freq = 1.0, 
        knot      = KNOT,
        theta0    = theta0,
        cpg_type  = CPG_MODE,
        dim = {
            'left'  : 11,
            'right' : 11
        })
print(cpg_ctrl._cpg_mode)
cpgLog = []
rfxLog = []
musLog = []
phaseManager = PhaseManager(transitions,alphas)


# Environment initialization method
#
#
def env_init():
    global  reflex_ctrl, env
    reflex_ctrl = OsimReflexCtrl(mode=mode, dt=sim_dt)
    env = L2M2019Env(visualize=visualize, seed=SEED, difficulty=difficulty)
    env.change_model(model="2D", difficulty=difficulty, seed=SEED)
    env.spec.timestep_limit = timstep_limit

# Actual controller  
#
#
def F(x):
    total_reward = 0
    t = 0
    i = 0
    obs_dict = env.reset(project=True, seed=SEED, obs_as_dict=True, init_pose=INIT_POSE)
    obs = env.get_observation()
    val = env.get_observation_dict()
    #reflex_ctrl.set_control_params(x[0:46])
    reflex_ctrl.set_control_params(x)
    elapsed_total = 0
    y             = []
    yt            = []
    e             = []
    observations  = []
    values        = []
    actions       = []
    velocity      = []
    ########################
    env.pose      = [0,0,0]
    _g_pose       = np.array(env.pose[0:2])
    head          = Heading(_g_pose)
    while True:
        if(heardEnter(cpg_ctrl)):
            print("ENTER")
            pass
        pose = np.array(env.pose[0:2])
        target = np.array(env.vtgt.get_vtgt(env.pose[0:2]))
        xt = head.update(pose,target)
        grf_left  = obs[297]
        grf_right = obs[253]
        speed = obs[245]
        yt.append([obs[299],obs[297]])
        i += 1
        t += sim_dt
        action_rfx         = np.array(reflex_ctrl.update(obs_dict))
        
        getStimByType = lambda x,b: np.array(reflex_ctrl._stimdict2array(x,balance=b))

        action_rfx_balance = getStimByType(reflex_ctrl.ctrl.balance_stim,True)
        action_rfx_spinal  = getStimByType(reflex_ctrl.ctrl.reflex_stim,False)
        action_rfx_sum     = getStimByType(reflex_ctrl.ctrl.stim,True)

        if(TEST_MODE):
            # Calculates the motoneurons activities

            action_cpg          = cpg_ctrl.update(grf_left,grf_right)
            speed_cpg           = cpg_ctrl.update(speed)
            action_spinal,_     = phaseManager.update(t,action_rfx_spinal,action_cpg)
            action              = np.clip(action_spinal + action_rfx_balance,0.01,1)
        else:
            if(TYPE == "spinal"):       # Learns spinal contribution to motoneurons
                action_cpg      = cpg_ctrl.update(grf_left,grf_right,action_rfx_spinal[11:],action_rfx_spinal[:11])
            elif(TYPE == "balance"):    # Learns balance contibution to motoneurons
                action_cpg      = cpg_ctrl.update(grf_left,grf_right,action_rfx_balance[11:],action_rfx_balance[:11])
            else:                       # Learns everything
                action_cpg      = cpg_ctrl.update(grf_left,grf_right,action_rfx_sum[11:],action_rfx_sum[:11])
            action              = action_rfx

        # Logging various information 
        cpgLog.append(action_cpg)           # append cpg contribution to motoneuron (before linear combination of reflex + cpg)
        rfxLog.append(action_rfx_spinal)    # append reflex contribution to motoneuron (before linear combination of reflex + cpg)
        musLog.append(action)               # append motoneuron activity (Registers the muscle activations)
        observations.append(obs)
        values.append(val)
        velocity.append(speed)
        mean_speed = np.mean(velocity)
        
        _t = time.time()
        obs_dict, reward, done, info = env.step(action, project = True, obs_as_dict=True)
        elapsed_total += time.time() - _t

        obs = env.get_observation()
        val = env.get_observation_dict()

        total_reward += reward
        if(done or np.mod(i,4000) == 0):
            if(not TEST_MODE):
                with open(FILE_PATH, "wb") as cpg_file:
                    pickle.dump(cpg_ctrl, cpg_file)
                    print("Cpg file saved into {}".format(FILE_PATH))
                import matplotlib.pyplot as plt
                #plt.close()
                #plt.plot(np.array(cpgLog)[450:450+125,]) 
                #plt.plot(np.array(rfxLog)[450:450+125,])
                #plt.show()
                musclePath = args.checkpoint.replace('.pkl','_muscle.pkl')
                reflexPath = args.checkpoint.replace('.pkl', '_reflex.pkl')
                valPath = args.checkpoint.replace('.pkl', '_values.pkl')
                pickle.dump(musLog, open(musclePath, "wb"))
                pickle.dump(rfxLog, open(reflexPath, "wb"))
                pickle.dump(values, open(valPath, "wb"))
            else:
                import ipdb; ipdb.set_trace()
            break

    print('    score={} time={}sec, realdt={}s, speed={}m/s, i={}, mean_vel={}m/s'.format(total_reward, t, elapsed_total/i, speed, i, mean_speed))
    return total_reward,t,elapsed_total/i,speed,i,mean_speed

# Run the simulation 
#
#
# 1. Environment initialization 
env_init()
# 2. Parameter loading
if(args.checkpoint):
    try:
        with open("{}".format(args.checkpoint), "rb") as cp_file:
            cp = pickle.load(cp_file)
        best_fitness      = cp["best_fitness"];
        best_ind          = cp["best_ind"]
        sigma             = cp["sigma"]
        start_gen         = cp["start_gen"]
        halloffame        = cp["halloffame"]
        logbook           = cp["logbook"]
        random.setstate(cp["rndstate"])
    except:
        best_ind = np.loadtxt("{}".format(args.checkpoint))
# 3. Actual simulation 
print(best_ind)
F(best_ind)

