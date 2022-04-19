DEBUG = True
import numpy as np

generateTime = lambda Tmax,dt: np.linspace(0,Tmax,int(Tmax/dt+1.0))

_normalize    = lambda x: x if np.linalg.norm(x) == 0 else np.array(x)/np.linalg.norm(x)
_rot_mat_2D   = lambda t: np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
_rotate_2D    = lambda x,theta: np.matmul(_rot_mat_2D(theta),x)
_rot_glob_loc = lambda x: 0 if np.linalg.norm(x) == 0 else np.arccos(x[0]/np.linalg.norm(x))

def roll(A, r):
    r = np.array(r)
    rows, column_indices = np.ogrid[:A.shape[0], :A.shape[1]]    
    r[r < 0] += A.shape[1]
    column_indices = column_indices - r[:,np.newaxis]
    return A[rows, column_indices]

action = lambda fb,ff,alpha : (1-alpha)*fb + alpha*ff


class PhaseManager():
    def __init__(self,transitions,alphas):
        # We add a last transitions without termination
        self.transitions = [lambda *args,**kwargs:True] + transitions + [lambda *args,**kwargs: False]
        self.alphas       =[alphas[0]]  + alphas  + [alphas[-1]]
        self.phase       = 0
    def update(self,t,*args):
        old_phase = self.phase
        if(self.transitions[self.phase](t)):
            self.phase = self.phase+1
        if(old_phase is not self.phase):
            print("Entering phase {}".format(self.phase-1))
        return action(args[0],args[1],self.alphas[self.phase]), old_phase is not self.phase


class Heading(object):
    # _g_pose = np.array(env.pose[0:2])
    # _g_vtgt = np.array(env.vtgt.get_vtgt(env.pose[0:2]))
    # _g_currentHeading = _normalize(_g_pose - _g_prevpose)
    # _g_desiredHeading = _normalize(_g_vtgt)
    # _l_desiredHeading = _normalize(_rotate_2D(_g_vtgt,_rot_glob_loc(_g_pose - _g_prevpose)))
    # print(angle)
    # print("current heading global {}".format(_g_currentHeading))
    # print("desired heading global {}".format(_g_desiredHeading))
    # print("desired heading local  {}".format(_l_desiredHeading))
    def __init__(self,g0=[0,0]):
        self.prevpose = g0
        self.current  = g0


    def update(self,pose,target):
        self.current =  _normalize(
                            _rotate_2D(target,
                                  _rot_glob_loc(pose - self.prevpose)
                                   )
                             )
        self.prevpose = pose
        return pose if np.isnan(self.current[0]) else self.current

    def getTurningDirection(self): # -1 : right, 1 : left
        return np.sign(self.current[1])

    def getTurningIntensity(self):
        return np.round(np.abs(self.current[1]*100),2)


class TinySpace():
    obs_ids_left = [
                243,   # PITCH 
                298,   # HIPCOR_LEFT
                262+1, # R_HAB length feedback
                265+1, # R_HAD length feedback
                268+1, # R_HF length feedback
                271+1, # R_GLU length feedback
                274+1, # R_HAM length feedback
                277+1, # R_RF length feedback
                280+1, # R_VAS length feedback
                283+1, # R_BFSH length feedback
                286+1, # R_GAS length feedback
                289+1, # R_SOL length feedback
                292+1, # R_TA length feedback
    ]
    obs_ids_right = [
                254,   # HIPCOR_RIGHT
                306+1, # L_HAB length feedback
                309+1, # L_HAD length feedback
                312+1, # L_HF length feedback
                315+1, # L_GLU length feedback
                318+1, # L_HAM length feedback
                321+1, # L_RF length feedback
                324+1, # L_VAS length feedback
                327+1, # L_BFSH length feedback
                330+1, # L_GAS length feedback
                333+1, # L_SOL length feedback
                336+1, # L_TA length feedback
    ]
    def __init__(self,*args, 
        sim_dt = None, 
        init_freq = 1.0, 
        freq = 1.0, 
        knot = None, 
        theta0 = None, 
        cpg_type = None,
        dim = None
        ):

        left_theta0 = theta0['left'] if theta0 is not None else None
        right_theta0 = theta0['right'] if theta0 is not None else None

        # Enable CPG mode if sim_dt is set
        self._cpg_mode = True if sim_dt is not None else False
        # Create CPG if in cpg_mode
        if(self._cpg_mode):
            print("CPG MODE IS ON, creating network")
            _dim = len(TinySpace.obs_ids_left) if(not dim) else dim["left"]
            self.left_cpg = LegNetwork(sim_dt, init_freq, freq, knot, _dim, type = cpg_type, theta0=left_theta0)
            _dim = len(TinySpace.obs_ids_right) if(not dim) else dim["right"]
            self.right_cpg = LegNetwork(sim_dt, init_freq, freq, knot, _dim, type = cpg_type, theta0=right_theta0)

        if(len(args) == 2):
            obs_body_space, extra_space = args
            self._generate(obs_body_space, extra_space)

    def reset(self,freq=None,theta0=None):
        left_theta0  = theta0['left']  if theta0 is not None else None
        right_theta0 = theta0['right'] if theta0 is not None else None
        self.left_cpg.cpg.clock.reset(freq,left_theta0)
        self.right_cpg.cpg.clock.reset(freq,right_theta0)

    def _generate(self,obs_body_space, extra_space):
        from osim.env.utils.mygym import convert_to_gym
        offset_body_ids = 242
        get_subspace  = lambda space, subspace_ids : space[:,np.array(subspace_ids)-offset_body_ids]
        observation_space_left = get_subspace(
            obs_body_space, TinySpace.obs_ids_left
            )
        observation_space_right = get_subspace(
            obs_body_space, TinySpace.obs_ids_right
            )

        observation_space = np.concatenate((observation_space_left, observation_space_right, extra_space), axis=1)
        self.observation_space =  convert_to_gym(observation_space)
        return self.observation_space

    def __call__(self,obs,extra_obs):
        self.current_obs = obs
        # No cpg so we simply concatenate obs and extra_obs.
        if( not self._cpg_mode):
            x = np.array(obs)[TinySpace.obs_ids_left + TinySpace.obs_ids_right]
            return x.tolist()+extra_obs
        # There is a cpg so we update it and then return the error between 
        # What we expected and we got for the given state. The state is provided by a discretization of a clock synchronized with the environment, see the Clock class.
        else:
            # Left cpg
            x_left = np.array(obs)[TinySpace.obs_ids_left]
            grf_left  = obs[297]
            self.left_cpg.update(grf_left,x_left)
            # Right cpg
            x_right = np.array(obs)[TinySpace.obs_ids_right]
            grf_right = obs[253]
            self.right_cpg.update(grf_right,x_right)
            # Calculation of observation vector (viewed as the prediction error)
            e = self.left_cpg.getInstanceDifference(x_left).tolist() \
               + self.right_cpg.getInstanceDifference(x_right).tolist()
            return e+extra_obs

    def enableCPG(self):
        self._cpg_mode = True

    def get(self):
        return self.observation_space



class Clock:
    def __init__(self,dt,freq, theta0=0):
        self.dt             = dt
        self.duration       = 0
        self.time           = 0
        self.switch         = 0.5
        self.reset(freq,theta0)

    def reset(self,freq=None,theta0=None):
        if(theta0):
            self.theta = theta0
            if(self.theta >= self.switch):
                self.phase = 1
            else:
                self.phase = 0       # 0=stance, 1=swing
        if(freq):
            self.freq  = freq
        

    def update(self,syncSignal):
        try:
            self.time           += self.dt
            #############################
            self.duration       += self.dt
            #############################
            # Clock update

            self.theta  += self.dt*self.freq
            if(syncSignal == 1):
                self.phase = 0
            if(syncSignal == -1):
                self.phase = 1
            if(self.phase == 0):
                self.theta   = 0 if syncSignal == 1 else self.switch if self.theta >= self.switch else self.theta
            if(self.phase == 1):
                self.theta   = self.switch if syncSignal == -1 else 1.0 if self.theta >= 1.0 else self.theta

            #############################
            # Frequency update
            min_duration = 0.8 
            if(syncSignal and self.duration  > min_duration):
                tau           = 10
                df            = tau*(1/self.duration - self.freq)
                self.freq    += self.dt*df
                self.duration = 0;
        except:
            if(self.theta >= self.switch):
                self.__setattr__('phase',1)
            else:
                self.__setattr__('phase',0)


class CpgCtrl(TinySpace):
    def __init__(self,**kwargs):
        super(CpgCtrl, self).__init__(**kwargs)

    def __call__(self,grf_left,grf_right):
        return self.update(grf_left,grf_right)

    def update(self,grf_left,grf_right, x_left = None, x_right = None):
        #return np.concatenate([self.left_cpg.update(grf_left, x_left),self.right_cpg.update(grf_right, x_right)])
        return np.concatenate([self.right_cpg.update(grf_right, x_right),self.left_cpg.update(grf_left, x_left)])

    def set_control_params(self, x):
        K = self.right_cpg.cpg.W.shape[0]
        N = self.right_cpg.cpg.W.shape[1]

        shift = x[:N]
        shift = np.round(shift*K).astype(int)
        shift = shift.clip(-K,K)

        scale = x[N:]

        self.right_cpg.cpg.shift = shift
        self.left_cpg.cpg.shift = shift
        self.right_cpg.cpg.scale = scale
        self.left_cpg.cpg.scale = scale

        self.left_cpg.cpg.applyShift()
        self.right_cpg.cpg.applyShift()

    def set_cpg_weights(self,l,r):
        self.left_cpg.cpg.W = l
        self.right_cpg.cpg.W = r

    def disableLearning(self):
        self.right_cpg.learning = False
        self.left_cpg.learning  = False

    def set_frequency(self,freq):
        self.left_cpg.clock.freq = freq
        self.right_cpg.clock.freq = freq



class LegNetwork():
    def __init__(self,sim_dt, init_freq, freq, knot, dim, theta0=0.0, type = None):
        self.learning = True
        self.clock = Clock(sim_dt,init_freq,theta0=theta0)
        self.leg   = Leg()
        self.cpg   = Cpg(self.clock, knot, dim, type = type)

    def update(self,grf,x = None):
        self.leg.update(grf)
        x = None if(not self.learning) else x  # You need to enable learning if you want this thing to be used.
        return self.cpg.update(self.leg.contact()-self.leg.liftoff(),x)

    def getInstanceDifference(self,x):
        return x-self.cpg()

    def getFF(self):
        return self.cpg()





class Cpg:
    def __init__(self, clock, K, N, tau = 10, lr = 20, type = "single"): # type can be "half_center" or "single"
        self.theta = 0
        if not type :
            type = "single"
        if(type == "single"):
            self.half_center = False
            self.W           = np.random.randn(K,N)
            self.K           = K
            self.N           = N
        elif(type == "half_center"):
            self.half_center = True
            self.stance = {
                "W":  np.random.randn(K,N),
                "K":  K,
                "N":  N
            }
            self.swing = {
                "W":  np.random.randn(K,N),
                "K":  K,
                "N":  N
            }
        else:
            import sys
            print("Error: Cpg type {} not found".format(type))
            sys.exit(1)
        self.tau   = tau
        self.clock = clock
        self.lr    = lr
        self.ase   = 20
        self.shift = np.zeros(N,dtype=int)
        self.scale = np.ones(N,dtype=float)

    def __call__(self):
        return self.getOutput()

    def _getActiveKnot(self):
        if(self.half_center):
            if(self.clock.phase == 0): # Stance 
                K = self.stance["K"]
                theta = self.clock.theta/self.clock.switch
            else: # Swing
                K = self.swing["K"]
                theta = (self.clock.theta-self.clock.switch)/(1.0-self.clock.switch)
        else:
            K = self.K
            theta = self.clock.theta

        value = int(np.round(K*theta))

        if(self.half_center):
            if(self.clock.phase == 0 and value == K): # Stance
                return 0, 1
            elif(self.clock.phase == 1 and value == K): # Stance
                return 0, 0
            else:
                return value, self.clock.phase

        else:
            STANCE_PREPARATION = True
            if(STANCE_PREPARATION):
                value = 0 if value == K else value
            else:
                value = -1 if value == K else value
            return value, self.clock.phase
        
    def getError(self):
        return np.sum(self.ase)

    def getOutput(self):
        active_knot,phase = self._getActiveKnot()
        if(self.half_center):
            if(phase == 0):
                return self.scale*self.stance["W"][active_knot]
            else:
                return self.scale*self.swing["W"][active_knot]
        else:
            return self.scale*self.W[active_knot]

    '''
    syncSignal : should be 1 when synchronizing events occurs.
    '''
    def update(self, syncSignal, teachingSignal = None):
        self.clock.update(syncSignal)
        if(type(teachingSignal) == np.ndarray):
            _id,phase     = self._getActiveKnot()
            if(self.half_center):
                if(phase == 0): # Stance 
                    w = self.stance["W"][_id]
                else: 
                    w = self.swing["W"][_id]
            else:
                w = self.W[_id]
            # error calculation 
            _ase         = (teachingSignal-self())**2
            dase         = self.tau*(_ase - self.ase)
            self.ase    += self.clock.dt*dase
            # Learn 
            if(sum(_ase) > 0.001):
                for i in range(self.lr):
                    #####################################
                    dw      = self.tau*(teachingSignal - w)
                    w      += self.clock.dt*dw
        elif(teachingSignal is not None):
            raise AttributeError('teachingSignal must be numpy array')
            
        return self()

    def applyShift(self,phase=None):
        if(self.half_center):
            if(not phase):
                self.stance["W"] = roll(self.stance["W"].transpose(),self.shift).transpose()
                self.swing["W"]  = roll(self.swing["W"].transpose(),self.shift).transpose()
            else:
                if(phase == 0): # Stance
                    self.stance["W"] = roll(self.stance["W"].transpose(),self.shift).transpose()
                else: #  Swing 
                    self.swing["W"]  = roll(self.swing["W"].transpose(),self.shift).transpose()
        else:
            self.W = roll(self.W.transpose(),self.shift).transpose()

    def setWeights(weights,weights_swing=None):
        if(self.half_center):
            if(weights_swing is None):
                raise ValueError
            self.stance["W"] = weights
            self.swing["W"] = weights_swing
        else:
            self.W = weights

class Leg:
    def __init__(self):
        self._contact        = False
        self._contact_prev   = False
        ############################
        self.thr             = 0.3

    def update(self,grf):
        self._contact_prev   = self._contact
        self._contact        = True if grf > self.thr else False

    def contact(self):
        return True if self._contact_prev is not self._contact and self._contact  is True   else False

    def liftoff(self):
        return True if self._contact_prev is not self._contact and self._contact  is False  else False



# import matplotlib.pyplot as plt
# dt = 0.01
# signal = np.array([0.3*np.sin(time),0.5*np.sin(2*time)]) 
# cpg = Cpg(1/(2*np.pi), 10, 2, dt)
# time = generateTime(100,0.01)
########### Learn #############
# [cpg.update(False,signal[i]) for (i,t) in enumerate(time)]
###########  Use  #############
# cpg.update(False)
