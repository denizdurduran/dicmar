import argparse
import numpy as np
import pickle
import random
import time
from osim.env import L2M2019Env
from osim.control.osim_loco_reflex_song2019 import OsimReflexCtrl

from src.cpg import Cpg, Clock, Leg, Heading, CpgCtrl

mode = '2D'
#difficulty = 2
difficulty = 0
visualize=True
sim_dt = 0.01
sim_t = 10
timstep_limit = int(round(sim_t/sim_dt))


parser = argparse.ArgumentParser(prog='Distributed Population Based Optimization (With MPI)')
parser.add_argument("-n",   "--n-invidiual", help="", default=2)
parser.add_argument("-t",   "--type",        help="", default="GA")
parser.add_argument("-c",   "--checkpoint",  help="", default="")
parser.add_argument("-s",   "--seed",        help="", default=64)
parser.add_argument("-g",   "--ngen",        help="", default=100)
parser.add_argument("-sig", "--sigma",       help="", default=5.0)



args = parser.parse_args()

SEED                = int(args.seed)        # Random seed 
np.random.seed(SEED)


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

def env_init():
    global  locoCtrl_python, env
    locoCtrl_python = OsimReflexCtrl(mode=mode, dt=sim_dt)
    env = L2M2019Env(visualize=visualize, seed=SEED, difficulty=difficulty)
    env.change_model(model="2D", difficulty=difficulty, seed=SEED)
    env.spec.timestep_limit = timstep_limit


def F(x):
    global locoCtrl_python, env
    if(not env or not locoCtrl_python):
        locoCtrl_python = OsimReflexCtrl(mode=mode, dt=sim_dt)
        env = L2M2019Env(visualize=visualize, seed=SEED, difficulty=difficulty)
        env.change_model(model=mode, difficulty=difficulty, seed=SEED)
        env.spec.timestep_limit = timstep_limit
    total_reward = 0
    t = 0
    i = 0
    obs_dict = env.reset(project=True, seed=SEED, obs_as_dict=True, init_pose=INIT_POSE)
    obs = env.get_observation()
    locoCtrl_python.set_control_params(x)
    elapsed_total = 0
    y             = []
    yt            = []
    e             = []
    observations  = []
    actions       = []
    ########################
    env.pose      = [0,0,0]
    _g_pose       = np.array(env.pose[0:2])
    head          = Heading(_g_pose)

    cpg_ctrl = CpgCtrl(
            sim_dt    = sim_dt, 
            init_freq = 1.0, 
            knot      = 50,
            theta0 = {
                'left'  : 0.1,
                'right' : 0.6
            },
            dim = {
                'left'  : 11,
                'right' : 11
            })
    cpg_ctrl.disableLearning()
    print(cpg_ctrl._cpg_mode)
    cpgLog = []
    while True:
        pose = np.array(env.pose[0:2])
        target = np.array(env.vtgt.get_vtgt(env.pose[0:2]))
        xt = head.update(pose,target)
        #if(head.getTurningDirection() == -1):
        #    print("Turn right with an intensity of {}%".format(head.getTurningIntensity()))
        #else:
        #    print("Turn left with an intensity of {}%".format(head.getTurningIntensity()))
        grf_left  = obs[297]
        grf_right = obs[253]
        yt.append([obs[299],obs[297]])
        i += 1
        t += sim_dt
        action_rfx      = np.array(locoCtrl_python.update(obs_dict))
        action_cpg      = cpg_ctrl.update(grf_left,grf_right)
        #action_cpg      = cpg_ctrl.update(grf_left,grf_right,np.array(action_rfx[11:]),np.array(action_rfx[:11]))
        action_speedCor = np.array(0)#np.array(x[46:46+22])*(obs_dict['pelvis']['vel'][0]-1.3)

        if(t > 10.0):
            #import matplotlib.pyplot as plt
            #plt.close()
            #plt.plot(np.array(actions)-np.array(cpgLog))
            #plt.show()
            #import ipdb; ipdb.set_trace()
            action      = 0.7*action_rfx + 0.3*action_cpg + action_speedCor
        else:
            action      = action_rfx
        #action      = 0.5*action_rfx + 0.5*action_cpg + action_speedCor

        cpgLog.append(action_cpg)
        observations.append(obs)
        actions.append(action_rfx)
        _t = time.time()

        # Muscle list 3D "hab", "had", "hf", "glu", "ham", "rf", "vas", "bhfs", "gas", "sol", "ta"
        # Muscle list 2D "hf", "glu", "ham", "rf", "vas", "bhfs", "gas", "sol", "ta"
        HAB  = 0
        HAD  = 1
        HF   = 2
        GLU  = 3
        HAM  = 4
        RF   = 5
        VAS  = 6
        BHFS = 7
        GAS  = 8
        SOL  = 9
        TA   = 10
        N_MUSCLES = 11
        T_max = 1.5

        def get_scaling(t,t0,t1):
            a  = 1.0/(t0-t1)
            b  = t1/(t1-t0)
            return (t*a+b)*(t > t0)*(t < t1)

        def add_symmetric_activity(id,x,I): 
            x *= get_scaling(t,I[0],I[1])
            action[id] += x
            action[id+N_MUSCLES] += x
            return y
        def add_left_activity(id,x,I):
            x *= get_scaling(t,I[0],I[1])
            action[id+N_MUSCLES] += x
        def add_right_activity(id,x,I):
            x *= get_scaling(t,I[0],I[1])
            action[id] += x


        def add_activity(action,x,I):
            action += x*get_scaling(t,I[0],I[1])
            return action

        # if(t<T_max):
        #     # We first move the trunk forward to create momentum
        #     action = add_activity(action, np.random.rand(22),[0,0.2])
        #     action = add_activity(action, np.random.rand(22),[0.2,0.4])
        #     action = add_activity(action, np.random.rand(22),[0.4,0.6])
        #     action = add_activity(action, np.random.rand(22),[0.6,0.8])

        obs_dict, reward, done, info = env.step(action_rfx, project = True, obs_as_dict=True)
        #if(t < 0.2):
        #    obs_dict['l_leg']['ground_reaction_forces'][2] = 0
        elapsed_total += time.time() - _t

        obs = env.get_observation()
        total_reward += reward
        if(done or np.mod(i,4000) == 0):
            import ipdb; ipdb.set_trace()
            break


    
    print('    score={} time={}sec, realdt={}s'.format(total_reward, t, elapsed_total/i))
    return total_reward,t,elapsed_total/i

env_init()

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
        #best_ind = np.loadtxt('./control/params_3D_cmaes2_rew183.txt')

print(best_ind)
F(best_ind)
