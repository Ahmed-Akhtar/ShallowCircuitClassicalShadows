#!/usr/bin/env python
# coding: utf-8

import sys
import torch
import scipy
import itertools
import matplotlib.pyplot as plt
import random
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import torch.optim as optim
from MPS import MPS
from EF_MPS_utils import *
from vMPSSolver import vMPSSolver
torch.__version__

# In[2]:

torch.set_default_dtype(torch.float64)


# In[3]:


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(device)


# In[4]:


print(torch.cuda.get_device_name(0))

print(torch.cuda.memory_summary())


# ## Fix Parameters

n=int(sys.argv[1])
d=int(sys.argv[2])
t=int(sys.argv[3])
Dw1=int(sys.argv[4])
Dw2=int(sys.argv[5])
steps=int(sys.argv[6])
sensitivity=int(sys.argv[7])

print('n=%d d=%d t=%d Dw1=%d Dw2=%d '%(n,d,t,Dw1,Dw2))

W,tmat=product_state_mps(1,cell_size=2),ef_haar_tm(d)
for step in range(t):
    W=ef_normalize(evolve_mps(tmat,W,Dw1,(t-step-1)%2),n)
W=W.to(device)

params = None

thresholds = [10**(-i+1) for i in range(sensitivity)]
losses=[-1]*len(thresholds)

vms = vMPSSolver(phys_dim=2, aux_dim=Dw2, unit_cell=2).to(device)

with torch.no_grad():
    loss = vms.loss(W, n // 2) 
    for i in range(len(thresholds)-1) :
        if loss.item() < thresholds[i] and loss.item() >= thresholds[i+1] :
            losses[i]=loss.item()
            filename='EFMPS_N=%d_d=%d_Dw1=%d_Dw2=%d_t=%d_%d.pt' % (n,d,Dw1,Dw2,t,i)
            torch.save({"mats":vms.mats.detach()},filename)

# In[9]:

optimizer = optim.AdamW(vms.parameters(), lr=0.005)


# In[ ]:

for k in range(steps):
    loss = vms.loss(W, n // 2) 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        loss = hrc.loss(W, n // 2) 
        for i in range(len(thresholds)-1) :
            if loss.item() < thresholds[i] and loss.item() >= thresholds[i+1] :
                losses[i]=loss.item()
                filename='EFMPS_N=%d_d=%d_Dw1=%d_Dw2=%d_t=%d_%d.pt' % (n,d,Dw1,Dw2,t,i)
                torch.save({"mats":vms.mats.detach()},filename)

# In[13]:
print('losses:', losses)