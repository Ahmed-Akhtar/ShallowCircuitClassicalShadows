#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 11:34:08 2021

@author: ahmedakhtar

EF_MPS_utils - a collection of methods utilizing the MPS class to represent
    entanglement feature and reconstruction coefficients
"""

from MPS import MPS 
import math
import torch
from torch import tensordot as tdot
import torch.nn.functional as F

pm = torch.tensor([[[1,0],[0,1]], [[0,1],[1,0]],
    [[0,-1j],[1j,0]], [[1,0],[0,-1]]]) # this is Pauli matrices: I, X, Y, Z

######## compute kronecker product ########
def kron(A, B):
    return torch.einsum("ab,cd->acbd", A, B).view(A.size(0)*B.size(0), \
                                                  A.size(1)*B.size(1))
        
######## SVD update tensors after applying tmat ########
def svd_approx_mps(tmat,left,right,aux_dim):
    #contract mps tensors with transfer matrix (d x d x d x d) with left right
    #onsite tensors of dimension D1 x D2 x d and D2 x D1 x d
    D1,D2,d=left.shape
    u,s,v=torch.svd(tdot(tmat,tdot(left,right,dims=([1],[0])), \
        dims=([2,3],[1,3])).transpose(1,2).reshape(d*D1,d*D1),some=True)
    #keep only aux_dim largest singular values
    vh=v.transpose(0,1).conj()
    uu,ss,vvh=u[:,:aux_dim],torch.sqrt(s[:aux_dim]),vh[:aux_dim,:]
    #form new left,right tensors 
    a,b=torch.mm(uu,torch.diag(ss)),torch.mm(torch.diag(ss),vvh)
    return a.view(d,D1,-1).permute(1,2,0), b.view(-1,d,D1).transpose(1,2)

######## EF MPS for product state with D=aux_dim ########
def product_state_mps(aux_dim,cell_size=1):
    mat=torch.zeros(aux_dim,aux_dim,2)
    mat[0,0,:]=1.0
    return MPS(torch.stack([mat]*cell_size))

######## D=2 EF MPS Ansatz state ########
def ef_mps(n,d,alpha,theta,cell_size=1):
    M0 = math.cosh(alpha) * torch.tensor([[1.,0.],[0.,1.]]) 
    M1 = math.sinh(alpha) * math.sin(theta) * torch.tensor([[0.,1.],[1.,0.]])
    M3 = math.sinh(alpha) * math.cos(theta) * torch.tensor([[1.,0.],[0.,-1.]])
    mats = torch.stack([M0+M1+M3, M0+M1-M3], -1).unsqueeze(0)
    W = MPS(torch.stack([mats[0]] * cell_size))
    W.mats = W.mats / W.component([0]*n)**(1/n)
    return W

######## re-normalize ef mps state ########
def ef_normalize(mps,n):
    return MPS(torch.clone(mps.mats) / mps.component([0]*n)**(1/n))

######## common two-local ef transfer matrices ########
def ef_haar_tm(d): #2 x 2 x 2 x 2
    i,x,z=pm[0].real,pm[1].real,pm[3].real
    ii,zz,xi,ix=kron(i,i),kron(z,z),kron(x,i),kron(i,x)
    return (ii - 0.5 * torch.mm(ii-zz,ii-(d/(d*d+1))*(xi+ix))).reshape(2,2,2,2)

def ef_ham_tm(d,g,beta):
    i,x,z=pm[0].real,pm[1].real,pm[3].real
    ii,zz,xi,ix,xx=kron(i,i),kron(z,z),kron(x,i),kron(i,x),kron(x,x)
    delta=math.atanh(1/d)
    return (ii-g*0.5*torch.mm(ii-zz, \
        torch.matrix_exp(-delta*(xi+ix)-beta*xx))).reshape(2,2,2,2) 

def proj_zz_even():
    i,z=pm[0].real,pm[3].real
    ii,zz=kron(i,i),kron(z,z)
    return (0.5*(ii+zz)).reshape(2,2,2,2)

def pgate(a,b):
    gates=[pm[0].real,pm[1].real,pm[3].real @ pm[1].real,pm[3].real]
    return kron(gates[a],gates[b])
    
######## update mps parameters ########
def update_mps_parameters(n,d,alpha0,theta0):
    return False

######## apply even+odd layer to mps ########
def evolve_mps(tmat,mps,aux_dim,layer):
    a,b=mps.mats[0],mps.mats[1]
    if not layer :
        aa,bb=svd_approx_mps(tmat,a,b,aux_dim)
    else :
        bb,aa=svd_approx_mps(tmat,b,a,aux_dim)
    #pad tensors to make them square
    if aa.shape[0]==aa.shape[1] :
        return MPS(torch.stack([aa,bb]))
    else :
        D=max(aa.shape[0],aa.shape[1])
        aaa=F.pad(input=aa,pad=(0,0,0,max(0,D-aa.shape[1]),0, \
            max(0,D-aa.shape[0])),mode='constant',value=0)
        bbb=F.pad(input=bb,pad=(0,0,0,max(0,D-bb.shape[1]),0, \
            max(0,D-bb.shape[0])),mode='constant',value=0)
        return MPS(torch.stack([aaa,bbb]))

######## pad mps ########
def pad_mps(mps,D):
    nmats=torch.zeros(mps.mats.shape[0],D,D,mps.mats.shape[3])
    nmats[:,:mps.mats.shape[1],:mps.mats.shape[2],:]=mps.mats
    return MPS(nmats)



######## l_{n,p}-sphere coordinate transforms ########
def spherical_to_cartesian(n,p,coords):
    if n==1:
        return coords
    else:#n>1
        x=coords[0]*torch.ones(n,dtype=torch.float64).to(coords.device)
        Np=lambda th: (abs(math.sin(th))**p + abs(math.cos(th))**p)**(1.0/p)
        sinp=lambda th: math.sin(th)/Np(th)
        cosp=lambda th: math.cos(th)/Np(th)
        for i in range(n):
            for j in range(i):
                x[i]*=sinp(coords[j+1])
            if i<n-1:
                x[i]*=cosp(coords[i+1])
        return x
        
######## return support vector as MPS for observable ########
def support_vector(obs_supp,n,factor=1):
    tensors=factor*torch.ones(n//2,1,1,4)
    for i in range(n//2):
        if 2*i in obs_supp:
            tensors[i,0,0,:2]=0
        if 2*i+1 in obs_supp:
            tensors[i,0,0,0:3:2]=0
    return MPS(tensors)

######## return r vector as MPS given matrices ########
def r_vector(n,d,eigs,ut,Am):
    Dr=Am.shape[0]
    An=torch.diag_embed(spherical_to_cartesian(Dr,n//2,F.pad(eigs,(1,0),value=1.0)))#dtype=torch.float64    
    An[torch.triu_indices(Dr,Dr,offset=1).unbind()]=ut
    rmats=torch.stack([(An+Am)/(2.0*d**2),(An-Am)/(2.0*d**2)],dim=2).view(1,Dr,Dr,2)
    ijk=[[[float(i==j==k) for i in range(2)] for j in range(2)] for k in range(2)]
    return MPS(torch.tensordot(rmats,torch.tensor(ijk).reshape(2,4),dims=([3],[0])))
            
    