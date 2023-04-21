#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 2021

@author: aaa

Fidelity circuit calculation job for OSG cluster. 
        
Arguments: 
n - system size
t - circuit depth
d - local hilbert space dimension
Dw - ef bond dimension
Dr - reconstruction bond dimension
fn - needed to specify input file
M - number of samples 
lbl - output file label

Inputs: 
base, mps libraries
reconstruction coefficients (mps)
necessary python packages

Output:
M snapshot fidelities (each fidelity is a single number)
    
"""

from context import *
import torch
import pickle
import scipy
import itertools
import random
from MPS import MPS
import random
from EF_MPS_utils import *
from stabilizer_mps_utils import *
import sys
torch.set_default_dtype(torch.float64)

#load relevant arguments
n=int(sys.argv[1])
t=int(sys.argv[2])
d=int(sys.argv[3])
Dw=int(sys.argv[4])
Dr=int(sys.argv[5])
fn=int(sys.argv[6])
M=int(sys.argv[7])
lbl=int(sys.argv[8])

#check if cuda is available
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

#generate random snapshots
def make_snapshot(rho0,L):
    comp_basis=paulis(pauli({i:'Z'}, n) for i in range(n))
    if math.inf>L>0:
        u=identity_map(n)
        for step in range(L):
            u=u.compose(random_brickwall_layer(n,step%2))
    elif L==0:
        u=random_pauli_map(n)
    else :
        u=random_clifford_map(n)
    rhof=rho0.to_map().compose(u).to_state()
    b=rhof.measure(comp_basis)[0]
    sigma=paulis(((-1)**(b[i]))*pauli({i:'Z'}, n) for i in range(n))
    return stabilizer_state(sigma).to_map().compose(u.inverse()).to_state()

#load reconstruction vector, return mps and boundary matrix
def load_rmps(n,d,Dw,Dr,t,acc,path='',device=torch.device('cpu')):
    if math.inf>t>0:
        filename='HaarReconstructionMPS_N=%d_d=%d_Dw=%d_Dr=%d_t=%d_%d.pt' % (n,d,Dw,Dr,t,acc)
        result=torch.load(path+filename,map_location=device)
        eigs,ut,Am=result["eigs"].to(device),result["ut"].to(device),result["Am"].to(device)
        rmps=r_vector(n,d,eigs,ut,Am)
        #check if there's a boundary tensor
        if "bdiag" in result :
            bdiag=result["bdiag"].to(device)
            rbdry=torch.diag(torch.tanh(bdiag))
        else :
            rbdry=None
    elif t==0 :
        rmps=MPS(torch.tensor([1.,-3./2.,-3./2.,9./4.]).reshape(1,1,1,4))
        rbdry=None
    else:
        #random Clifford case
        rbdry=torch.diag(torch.tensor([-1.,1.]))
        rmats=torch.zeros((1,2,2,4))
        rmats[0,0,0,0]=1
        rmats[0,1,1,1]=(1+d**(-n))**(2/n)
        rmps=MPS(rmats)
    return rmps,rbdry

#load reconstruction coefficient parameters and construct r mps
rmps,rbdry=load_rmps(n,d,Dw,Dr,t,fn)
rho0=ghz_state(n)

#calculate fidelities
fid=numpy.zeros(M)
for snap in range(M) :
    sigma=make_snapshot(rho0,t)
    fid[snap]=snapshot_fidelity(rho0,sigma,rmps,rbdry).real
    
#save result
numpy.savetxt('fid-%d-test.csv' % (lbl), fid, delimiter=",")
