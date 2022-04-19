#!/usr/bin/env python

import torch
import random
import numpy as np
from gym import spaces

from osim.env import L2M2019Env

from catalyst.rl.core import EnvironmentSpec
from catalyst.rl.utils import extend_space

from .env_wrappers import EnvNormalizer,SPACE_TYPE
from .cpg import TinySpace, Heading #, Cpg, Clock, Leg

from osim.control.osim_loco_reflex_song2019 import OsimReflexCtrl

BIG_NUM = np.iinfo(np.int32).max
INIT_POSE = np.array([
    0.0, # 1.699999999999999956e+00,   # forward speed
    0.0, # .5,                         # rightward speed
    0.94, # 9.023245653983965608e-01,   # pelvis height
    0.0, # 2.012303881285582852e-01,   # trunk lean
    0.0, # 0*np.pi/180,                # [right] hip adduct
    0.0, # -6.952390849304798115e-01,  # hip flex
    0.0, # -3.231075259785813891e-01,  # knee extend
    0.0, # 1.709011708233401095e-01,   # ankle flex
    0.0, # 0*np.pi/180,                # [left] hip adduct
    0.0, # -5.282323914341899296e-02,  # hip flex
    0.0, # -8.041966456860847323e-01,  # knee extend
    0.0]) # -1.745329251994329478e-01]) # ankle flex


rew_shapping   = lambda rewards, tau : tau*np.exp(tau*rewards)
rew_tau_update = lambda rewards, tau: rew_shapping(rewards,tau).sum()/(rew_shapping(rewards,tau)*rewards).sum()
U = lambda R,tau: rew_shapping(R,tau).sum()
Uscaled = lambda R,tau: (R*rew_shapping(R,tau)).sum()
tau = lambda R,tau: U(R,tau)/Uscaled(R,tau)

def cumulative_discount(rewards, discount):
    future_cumulative_reward = 0
    assert np.issubdtype(rewards.dtype, np.floating), rewards.dtype
    cumulative_rewards = np.empty_like(rewards)
    for i in range(len(rewards) - 1, -1, -1):
        cumulative_rewards[i] = rewards[i] + discount * future_cumulative_reward
        future_cumulative_reward = cumulative_rewards[i]
    return cumulative_rewards

class SkeletonEnvWrapper(EnvironmentSpec):
    def __init__(
        self,
        history_len=1,
        frame_skip=1,
        reward_scale=1,
        reload_period=None,
        action_mean=None,
        action_std=None,
        visualize=False,
        mode="train",
        **params
    ):

        super().__init__(visualize=visualize, mode=mode)
        self._tau = 1
        self._maxUtilitySoFar = 1
        self._history_len = history_len
        self._frame_skip = frame_skip
        self._visualize = visualize
        self._reward_scale = reward_scale
        self._reload_period = reload_period or BIG_NUM
        self.episode = 0
        self.sim_dt = 0.01
        # Env creation
        env = L2M2019Env(**params, visualize=visualize)
        if(SPACE_TYPE >= 1):
            self._prepare_spaces(env,self.sim_dt)
            env = EnvNormalizer(env)
        else:
            env = EnvNormalizer(env)
            self._prepare_spaces(env)
        self.env = env
        # Spinal controller
        self.locoCtrl = self.initReflex(self.sim_dt)
        self.locoCtrl_param = np.loadtxt('./control/params_3D_init.txt')
        self.action_mean = np.array(action_mean) \
            if action_mean is not None else None
        self.action_std = np.array(action_std) \
            if action_std is not None else None


        self.env.pose      = [0,0,0]
        self._g_pose       = np.array(env.pose[0:2])
        self.head          = Heading(self._g_pose)

    def initReflex(self,sim_dt):
        sim_t = 10
        timestep_limit = int(round(sim_t/sim_dt))
        self.env.spec.timestep_limit  = timestep_limit
        return OsimReflexCtrl(mode="3D", dt=sim_dt)

    def reset(self):
        if self.episode % self._reload_period == 0:
            self.locoCtrl.set_control_params(self.locoCtrl_param)
            difficulty = 2
            seed = random.randrange(BIG_NUM)
            self.env.change_model(difficulty=difficulty, seed=seed)
        else: # First episode has finished 
            # CumSum the reward
            reward_discount = cumulative_discount(np.array(self._rewards[::-1]), 0.99)[::-1]
            # Calculate the utility for all the reward.


            # We calculate the scaling factor and set the self._maxUtilitySoFar variable.
            self._tau = tau(cumulative_discount(np.array(self._rewards[::-1]), 0.99)[::-1],self._tau)
            self._maxUtilitySoFar = U(reward_discount,self._tau)
        self._rewards = [0]
        print("\tTau is {}".format(self._tau))
        print("\tMax utility so for is {}".format(self._maxUtilitySoFar))

        self.episode += 1
        observation = self.env.reset(init_pose=INIT_POSE)
        self.obs_dict = self.env.get_observation_dict() # @Todo can directly use observation.
        if(SPACE_TYPE == 1):
            return self._tiny_space(self.env.get_observation(),[0.0])
        elif(SPACE_TYPE >= 2):
            return self._tiny_space(self.env.get_observation(),[0.0,self._tiny_space.left_cpg.clock.theta,self._tiny_space.right_cpg.clock.theta])
        else:
            return observation



    @property
    def history_len(self):
        return self._history_len

    @property
    def observation_space(self) -> spaces.space.Space:
        return self._observation_space

    @property
    def state_space(self) -> spaces.space.Space:
        return self._state_space

    @property
    def action_space(self) -> spaces.space.Space:
        return self._action_space

    def _prepare_spaces(self,env, sim_dt = None):
        # Tiny Space 
        if(SPACE_TYPE >= 1):
            # Action space
            if(SPACE_TYPE == 1):
                from gym.spaces import Box
                self._action_space = Box(
                    low=-1.0, 
                    high=1.0, 
                    shape=(9,), 
                    dtype=np.float32
                )
            else:
                self._action_space = env.action_space
            vtgt_tiny_space = np.array([ # @TODO 
                [-1,1],
                [0,1],
                [0,1]
                ]).transpose()
            # Observation space
            self._tiny_space = TinySpace(
                env.obs_body_space,
                vtgt_tiny_space,
                sim_dt, 
                1.0, 
                load_from_file = {
                'left': "data/left_cpg_weights",
                'right':"data/right_cpg_weights"
                },
                theta0 = {
                'left' : 0.0,
                'right' : 0.5
                }
            ) # @TODO change init_freq
            self._tiny_space.left_cpg.cpg.learning = False
            self._tiny_space.right_cpg.cpg.learning = False
            self._observation_space = self._tiny_space.get()
        # Normal Space
        else:
            self._action_space = env.action_space
            self._observation_space = env.observation_space

        self._state_space = extend_space(
            self._observation_space, self._history_len
        )
        env._state_space = self._state_space

    def _process_action(self, action):
        if self.action_mean is not None \
                and self.action_std is not None:
            action = action * (self.action_std + 1e-8) + self.action_mean
        return action

    def step(self, action):

        reward = 0
        action_brain = self._process_action(action)
        #######################
        # Gathering action state tupple
        if(SPACE_TYPE == 1):
            _params = self.locoCtrl_param
            _params[-9:] *= (1+action_brain)
            self.locoCtrl.set_control_params(_params)
        for i in range(self._frame_skip):
            action_spine = self.locoCtrl.update(self.obs_dict)
            if(SPACE_TYPE == 1):
                observation, r, done, info = self.env.step(action_spine)
            elif(SPACE_TYPE == 2):
                observation, r, done, info = self.env.step(action_spine + action_brain)
            elif(SPACE_TYPE == 3):
                observation, r, done, info = self.env.step(action_brain)
            else:
                observation, r, done, info = self.env.step(action_spine) # + action_brain
            self.obs_dict = self.env.get_observation_dict() # @Todo can directly use observation.

            if self._visualize:
                self.env.render()
            reward  += r
            if done:
                break
        #######################
        # Returning action state tuple. 
        info["raw_reward"] = rew_shapping(reward, self._tau)
        self._rewards += [reward]
        reward *= self._reward_scale
        if(SPACE_TYPE >= 1):
            pose = np.array(self.env.pose[0:2])
            target = np.array(self.env.vtgt.get_vtgt(pose))
            xt = self.head.update(pose,target)
            if(SPACE_TYPE == 1):
                tinyobs = self._tiny_space(self.env.get_observation(),[xt[0]])
            else:
                tinyobs = self._tiny_space(self.env.get_observation(),[xt[0],self._tiny_space.left_cpg.clock.theta,self._tiny_space.right_cpg.clock.theta])
            return tinyobs, reward, done, info
        else:
            return observation, reward, done, info
        
