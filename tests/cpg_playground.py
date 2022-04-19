import numpy as np 
import matplotlib.pyplot as plt
from src.cpg import *
#######################################
sim_dt      = 0.01
Tmax        = 100
T           = 1
#######################################
time        = generateTime(Tmax,sim_dt)  
left_leg    = Leg()
left_clock  = Clock(sim_dt,1.0/T)
cpg         = Cpg(left_clock,50, 2)
yt          = np.array([0.3*np.sin((2*np.pi)/T*time),0.5*np.sin(2*(2*np.pi)/T*time)])
y           = []
error       = []
#######################################
for i in range(len(time)):
	error.append(cpg.getError())
	if i < len(time)/2:
		y.append(cpg.update(False,yt[:,i]))
	else:
		y.append(cpg.update(False))

	i=i+1
#######################################
y = np.array(y)
#plt.plot(time,y[:,0])
#plt.plot(time,yt[0,:])
plt.plot(time,error)
plt.show()