#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 22:18:30 2020

@author: nicolas
"""

import pickle 
import numpy as np
from math import *
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch_rbf as rbf



# CLASS FOR GENERATING THE DATASET
###################################

class MyDataset(data.Dataset):
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return self.x.size(0)
    
    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx,:]
        return (x, y)
    
# CLASS FOR HANDLING THE VANISHING GRADIENT
###################################
        
class GetOut(Exception):
    pass

# CLASS FOR THE RBF NEURAL NETWORK
###################################


class Learning_network(nn.Module):
    
    def __init__(self, period, n_signals, num_rbf=6, basis_func=rbf.gaussian):
        super(Learning_network, self).__init__()
        
        self.n_signals = n_signals
        self.num_rbf = num_rbf
        self.loss_log = []
        self.basis_func = basis_func
        self.layer_rbf = rbf.RBF(1, num_rbf, self.basis_func, period)
        self.output = nn.Linear(num_rbf, n_signals, bias=None)
        self.linear_w = torch.diag(torch.ones(n_signals)).clone().detach().requires_grad_(False)
        
        #RBF layer initialization done in torch_rbf.py
        #print(self.layer_rbf.centres.data)
        #print(self.layer_rbf.sigmas.data)

    def generate_cpg_param_range(self):
        N_RBF = self.num_rbf
        N_OUTPUT = self.n_signals
        lower_bound = []
        upper_bound = []
        # CENTERS
        lower_bound += np.array([i/N_RBF for i in range(N_RBF)]).tolist()
        upper_bound += np.array([i/N_RBF for i in range(1,N_RBF+1)]).tolist()
        # SIGMAS 
        lower_bound += N_RBF*[5]
        upper_bound += N_RBF*[17]
        # WEIGHTS 
        lower_bound += N_RBF*N_OUTPUT*[0]
        upper_bound += N_RBF*N_OUTPUT*[0.6]
        return (lower_bound,upper_bound)

    def pattern_formation(self,x):
        x_1 = self.layer_rbf(x)
        x_2 = self.layer_rbf(torch.tensor(-x.detach().numpy()[::-1]))
        x_3 = self.layer_rbf(torch.tensor(1+x.detach().numpy()))
        
        return torch.max(torch.max(x_1,x_2),x_3)
    def forward(self, x):
        x = self.pattern_formation(x)
        x = self.output(x)
        return x

    
    def fit(self, dataset, batch_size, epochs, lr):
        
        lr = lr
        trainloader = data.DataLoader( dataset, batch_size=batch_size, shuffle=False)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_func = nn.MSELoss()
        
        epoch = 0
        current_loss = 0
        prev_loss = 0 
        while epoch < epochs:
            epoch += 1
            batches = 0
            
            try:
                for x, y in trainloader:
                    batches =+ 1
                    optimizer.zero_grad()
                    y_hat = self.forward(x)
                    if np.any(np.isnan(y_hat.detach().numpy())):
                            raise GetOut
                    loss = loss_func(y_hat, y)
                    current_loss = prev_loss - 0.3*(prev_loss - loss.item())      #L[batch+1] = L[batch] - gamma*(L[batch] - new_loss)
                    #self.loss_log.append(current_loss)
                    loss.backward()
                    optimizer.step()
                    prev_loss = current_loss
                    print('\n Epoch: %s, Loss: %s' % (epoch, current_loss))
                    
                self.loss_log.append(current_loss)
            except GetOut:
                self.finished = False
                break
        
        if epoch == epochs:
            self.finished = True

    def set_params_from_array(self, x):
        N_RBF = self.num_rbf
        N_OUTPUT = self.n_signals
        centers = x[0:N_RBF]
        sigmas = np.array([x[N_RBF:2*N_RBF]])
        # Matrix containing only 1 and 0, with 0 for the muscles of the ankles
        ones = np.ones((6,N_RBF), dtype=float)
        zeros = np.zeros((3,N_RBF), dtype=float)
        matrix = np.r_[ones, zeros]
        weights = torch.tensor(x[2*N_RBF:].reshape(N_OUTPUT,N_RBF)*matrix,dtype=torch.float) 
        print(weights)
        #print(type(weights))
        self.set_params(centers=centers, sigmas=sigmas, weights = weights)

    def set_params(self, weights=None, centers=None, sigmas=None):
        # Function for changing the parameters of the network
        # The input arguments should be of size [1, num_rbf]
        
        centers = torch.nn.Parameter(torch.tensor([centers], dtype=torch.float).transpose(0,1))
        sigmas = torch.nn.Parameter(torch.tensor(sigmas, dtype=torch.float)) # transpose ?
        
        if centers is not None:
            self.layer_rbf.centres = centers  # changing the centers of the RBF
        if weights is not None:
            self.output.weight.data = weights.clone().detach()
        if sigmas is not None:
            self.layer_rbf.sigmas = sigmas 
