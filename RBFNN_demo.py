#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 23:36:55 2020

@author: nicolas
"""


from RBF_NN import Learning_network
from matplotlib.ticker import PercentFormatter
import numpy as np
import torch 
import matplotlib.pyplot as plt

plt.close('all')

N_RBF = 4
N_OUTPUT = 9

### DEMO to show how the NN and parameters setting works

'''
Initializes the pattern formation neural network. 

    Initialize the network, can be done also with an additional argument, basis_func, which changes 
    activation function of the hidden layer ( RBF by default )
'''
pfNN = Learning_network(1, N_OUTPUT, num_rbf=N_RBF)

'''
Input vector : Simulated phase (from 0 to 1)
'''
test_vec = np.arange(0,1,0.005)
torch_vec = torch.tensor([test_vec], dtype=torch.float).transpose(0,1)

'''
PF Parameters initialization
'''
METHOD = 2 # 1(random), 2(from_file)

if(METHOD == 1):
    # MOTOR PRIMITIVES CENTERS
    centers = np.array([i/N_RBF+np.random.random()*0.2 for i in range(N_RBF)])
    # MOTOR PRIMITIVES SCALING FACTOR
    sigmas = 5+12*np.random.random([1,N_RBF])
    # CONNECTION WEIGHTS BETWEEN MOTOR PRIMITIVES AND MUSCLES
    '''
    Initialize weights 

        Two methods are proposed here : 
            1. Sparse initialization (favors 0 in the weights matrix), this is good for showcase
            2. Uniform initialization (no sparsity), this is good for optimization init 
    '''
    # Method 1 
    weights = torch.empty(N_OUTPUT, N_RBF)
    torch.nn.init.sparse_(weights, sparsity=0.5, std=0.3)
    weights = torch.abs(weights)
    # Method 2
    # weights = torch.abs(torch.tensor(np.random.random([N_OUTPUT, N_RBF]),dtype=torch.float))
if(METHOD == 2):
    '''
        structure of the file : 
            [0-N_RBF[  : RBF center
            [N_RBF-2*N_RBF[  : RBF scaling
            [2*N_RBF-2*N_RBF+N_RBF*N_OUTPUT[ : Connection weights between RBF layer and output layer

        for example with 
            N_RBF = 4
            N_OUTPUT = 9

        structure of the file : 
            [0-4[  : RBF center
            [4-8[  : RBF scaling
            [8-44[ : Connection weights between RBF layer and output layer
    '''
    params_2D_CPG = np.loadtxt('control/params_2D_FDB_CPG_4MP.txt')
    centers = params_2D_CPG[37:37+N_RBF]
    sigmas = np.array([params_2D_CPG[37+N_RBF:37+2*N_RBF]])
    ones = np.ones((6,N_RBF), dtype=float)
    zeros = np.zeros((3,N_RBF), dtype=float)
    matrix = np.r_[ones, zeros]
    weights = torch.tensor(params_2D_CPG[37+2*N_RBF:].reshape(N_OUTPUT,N_RBF)*matrix,dtype=torch.float)

'''
RUN & PLOT THE NETWORK 
'''
# Apply the parameters
pfNN.set_params(weights=weights, centers=centers, sigmas=sigmas)

# Plot of the rbf layer 
plt.plot(pfNN.layer_rbf(torch_vec).detach().numpy()[0])
plt.title('Motor Primitives Signals')
plt.show()

xaxis = np.linspace(0, 1, len(torch_vec))
# Plot the pattern formation layer 

figpat = plt.figure()
bx = figpat.add_axes([0.1, 0.1, 0.7, 0.75])
p1, = bx.plot(xaxis, pfNN.pattern_formation(torch_vec).detach().numpy()[0,:,0], label='MP1')
p2, = bx.plot(xaxis, pfNN.pattern_formation(torch_vec).detach().numpy()[0,:,1], label='MP2')
p3, = bx.plot(xaxis, pfNN.pattern_formation(torch_vec).detach().numpy()[0,:,2], label='MP3')
p4, = bx.plot(xaxis, pfNN.pattern_formation(torch_vec).detach().numpy()[0,:,3], label='MP4')
plt.grid()
plt.xticks(np.arange(0, 1.1, step=0.1))
bx.margins(x=0)
plt.title("Motor Primitives Signals")
plt.xlabel('Gait Cycle [-]')
plt.ylabel('Motor Primitives [-]')
bx.legend(handles=[p1, p2, p3, p4, ], bbox_to_anchor = (1.05, 1), loc='upper left')
plt.show()

# # Evaluate the test vector after setting the parameters
output = pfNN.forward(torch_vec)


fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.7, 0.75])
p1, = ax.plot(test_vec, output.detach().numpy()[0,:,0], label='HFL')
p2, = ax.plot(test_vec, output.detach().numpy()[0,:,1], label='GLU')
p3, = ax.plot(test_vec, output.detach().numpy()[0,:,2], label='HAM')
p4, = ax.plot(test_vec, output.detach().numpy()[0,:,3], label='RF')
p5, = ax.plot(test_vec, output.detach().numpy()[0,:,4], label='VAS')
p6, = ax.plot(test_vec, output.detach().numpy()[0,:,5], label='BFSH')
p7, = ax.plot(test_vec, output.detach().numpy()[0,:,6], label='GAS')
p8, = ax.plot(test_vec, output.detach().numpy()[0,:,7], label='SOL')
p9, = ax.plot(test_vec, output.detach().numpy()[0,:,8], label='TA')
plt.grid()
plt.xticks(np.arange(0, 1.1, step=0.1))
ax.margins(x=0)
plt.title('CPG Contribution to the muscle activities')
plt.xlabel('Gait Cycle [-]')
plt.ylabel('CPG Contribution [-]')
ax.legend(handles=[p1, p2, p3, p4, p5, p6, p7, p8, p9], bbox_to_anchor = (1.05, 1), loc='upper left')
plt.show()
