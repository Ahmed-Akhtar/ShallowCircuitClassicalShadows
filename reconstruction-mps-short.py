#!/usr/bin/env python
# coding: utf-8

# In[1]:

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
from HaarReconstructionChannel import HaarReconstructionChannel
#get_ipython().run_line_magic('matplotlib', 'inline')
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


# In[5]:


#torch.cuda.empty_cache()
#import gc
#del W, hrc
#gc.collect()
print(torch.cuda.memory_summary())


# ## Fix Parameters

# In[6]:

n=int(sys.argv[1])
d=int(sys.argv[2])
Dw=int(sys.argv[3])
Dr=int(sys.argv[4])
t=int(sys.argv[5])
steps=int(sys.argv[6])
load=int(sys.argv[7])
savenum=int(sys.argv[8])
t0=int(sys.argv[9])

print('n=%d d=%d t=%d Dw=%d Dr=%d savenum=%d'%(n,d,t,Dw,Dr,savenum))

# In[7]:

if load >= 0 :
    filename='HaarReconstructionMPS_N=%d_d=%d_Dw=%d_Dr=%d_t=%d_%d.pt' % (n,d,Dw,Dr,t-t0,load) 
    result=torch.load(filename)
    eigs2,ut2,Am2=result["eigs"].to(device),result["ut"].to(device),result["Am"].to(device)
    #eigs2 += 0.1*torch.randn(Dr-1).to(device)
    params=[eigs2,ut2,Am2]
else:
    params=None

# In[8]:


hrc = HaarReconstructionChannel(d, aux_dim=Dr, params=params).to(device)
W,tmat=product_state_mps(1,cell_size=2),ef_haar_tm(d)
for step in range(t):
    W=ef_normalize(evolve_mps(tmat,W,Dw,(t-step-1)%2),n)
W=W.to(device)
losses,angles,evals=[],[],[]
with torch.no_grad():
    loss = hrc.loss(W, n // 2) 
    losses.append(loss.item())
    angs=nn.functional.pad(hrc.eigs.clone().detach(),(1,0),value=1.0)
    angles.append(angs[1:].cpu().numpy())
    evals.append(spherical_to_cartesian(Dr,n//2,angs).cpu().numpy())


# In[9]:


optimizer = optim.AdamW(hrc.parameters(), lr=0.005)


# In[ ]:


filename='HaarReconstructionMPS_N=%d_d=%d_Dw=%d_Dr=%d_t=%d_%d.pt' % (n,d,Dw,Dr,t,savenum)
echo = 100
for k in range(steps):
    loss = hrc.loss(W, n // 2) 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
        
    if (k+1)%echo == 0:
        #print('{:>4d}: {:8.6f}'.format(k+1, loss.item()))
        torch.save({"eigs":hrc.eigs.detach(), "ut":hrc.ut.detach(), "Am":hrc.Am.detach()},filename)
        losses.append(loss.item())
        angs=nn.functional.pad(hrc.eigs.clone().detach(),(1,0),value=1.0)
        angles.append(angs[1:].cpu().numpy())
        evals.append(spherical_to_cartesian(Dr,n//2,angs).cpu().numpy())

print('step: loss')
print('\n'.join('{}: {}'.format(*k) for k in enumerate(losses)))
print(angles)
#fig=plt.figure(figsize=(12,3), dpi= 100, facecolor='w', edgecolor='k')        
#plt.subplot(1,3,1)        
#plt.plot(losses,'.-')
#plt.xlabel(r'step (per %d)'%echo)
#plt.title(r'loss per step')
#plt.yscale("linear")

#plt.subplot(1,3,2)
#plt.plot(torch.tensor(angles),'-')
#plt.xlabel(r'step (per %d)'%echo)
#plt.title(r'$ \varphi_i $')
#plt.yscale("linear")

#plt.subplot(1,3,3)
#plt.plot(torch.tensor(evals),'-')
#plt.xlabel(r'step (per %d)'%echo)
#plt.title(r'eig($ A_{+} $)')
#plt.yscale("linear")


# In[13]:


An=torch.diag_embed(spherical_to_cartesian(Dr,n//2,nn.functional.pad(hrc.eigs.detach(),(1,0),value=1.0)))
An[torch.triu_indices(Dr,Dr,offset=1).unbind()]=hrc.ut.detach()
rmats=torch.stack([(An+hrc.Am.detach())/(2.0*d**2),(An-hrc.Am.detach())/(2.0*d**2)],dim=2).view(1,Dr,Dr,2)
mps1=MPS(torch.tensordot(rmats,hrc.dijk,dims=([3],[0])))
mps2=MPS(torch.tensordot(W.mats[0],W.mats[-1],dims=([1],[0])).transpose(1,2).reshape(1,W.mats.shape[1],W.mats.shape[2],4))
y2 = mps1.merge(mps2, hrc.f2)

#print('region:     r      y        W       m')
#for alphas in itertools.product(range(4),repeat=n//2):
#    print('{}: {: 8.2f} {: 8.4f} {: 8.4f} {: d}'.format(alphas, 
#                                         mps1.component(alphas).item(), 
#                                         y2.component(alphas).item(), 
#                                         mps2.component(alphas).item(),sum(alphas)))

print('region:     r      y        W       m')
configs = (3*torch.randint(2,(20,n//2))).tolist()
configs.append([3]*(n//2))
for alphas in configs:
    print('{}: {: 8.2f} {: 8.4f} {: 8.4f} {: d}'.format(alphas, 
                                         mps1.component(alphas).item(), 
                                         y2.component(alphas).item(), 
                                         mps2.component(alphas).item(),sum(alphas)))


# In[32]:


#filename='HaarReconstructionMPS_N=%d_d=%d_Dw=%d_Dr=%d_t=%d_2.pt' % (n,d,Dw,Dr,t)
#torch.save({"eigs":hrc.eigs.detach(), "ut":hrc.ut.detach(), "Am":hrc.Am.detach()},filename)

