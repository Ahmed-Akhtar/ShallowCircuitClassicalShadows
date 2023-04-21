#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 00:28:08 2022

@author: ahmedakhtar

stabilizer_MPS_utils - a collection of methods for studying tensor representations
    of stabilizer states
"""

import numpy
import itertools
from functools import reduce
import torch

######## return support of pauli string ########
def pauli_support(g):
    return numpy.ndarray.flatten(numpy.argwhere(g[::2]+g[1::2]))

######## return range of sites on which pauli string acts ########
def pauli_range(g):
    supp,n=pauli_support(g),len(g)//2
    if len(supp)%n==0:
        return supp
    #rotate so that the first site is nonzero
    col=numpy.roll(g[::2]+g[1::2],-supp[0])
    #find the contiguous strings of zeros
    ids=numpy.ndarray.flatten(numpy.argwhere(col==0))
    components,curr=[],[ids[0]]
    for i in range(1,len(ids)):
        if ids[i]==(curr[-1]+1):
            curr.append(ids[i])
        else :
            components.append(curr)
            curr=[ids[i]]
    components.append(curr)
    #the longest zero string's complement is the range
    lens=[len(com) for com in components]
    biggest=lens.index(max(lens))
    return numpy.array(list(range(supp[0]+components[biggest][-1]+1, \
                                  supp[0]+components[biggest][0]+n))) % n    

####### mps tensor for 1+P ########
def projtensor(g,p,i):
    rng=pauli_range(g)
    shape=[1 if i==rng[0] else 2,1 if i==rng[-1] else 2]
    gi=g[::2][i]+3*g[1::2][i]-2*g[::2][i]*g[1::2][i]
    tensor=torch.zeros(shape+[4])
    tensor[0,0,0]=1.0
    tensor[-1,-1,gi]=1j**p if i==rng[0] else 1.0
    return tensor

######## same as merge but for arbitrary shape tensors ########
def pfuse(t1,t2,vx):
    Da1,Da2,Db1,Db2=t1.shape[0],t1.shape[1],t2.shape[0],t2.shape[1]
    t3=kron(t1.reshape(-1,4),t2.reshape(-1,4)).type(torch.complex64).mm(vx.reshape(-1,4))
    return t3.reshape(Da1,Da2,Db1,Db2,4).transpose(1,2).reshape(Da1*Db1,Da2*Db2,4)

######## simple kronecker product ########
def kron(A, B):
    return torch.einsum("ab,cd->acbd", A, B).view(A.size(0)*B.size(0), \
                                                  A.size(1)*B.size(1))

######## combine two sites into one ########
def coarse_grain(t1,t2):
    return torch.tensordot(t1,t2,dims=([1],[0])).transpose(1,2).reshape(t1.shape[0],t2.shape[1],
                                                                t1.shape[2]*t2.shape[2])

######## return pauli algebra fusion vertex ########
def fusion_vertex():
    return torch.tensor([[[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
         [0.+0.j, 1.+0.j, 0.+0.j, 0.+-0.j],
         [0.+0.j, 0.+-0.j, 1.+0.j, 0.+0.j],
         [0.+0.j, 0.+0.j, 0.+-0.j, 1.+0.j]],
        [[0.+0.j, 1.+0.j, 0.+0.j, 0.+-0.j],
         [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
         [0.+0.j, -0.+0.j, 0.+0.j, 0.+1.j],
         [0.+-0.j, -0.+0.j, 0.-1.j, 0.+0.j]],
        [[0.+0.j, 0.+-0.j, 1.+0.j, 0.+0.j],
         [0.+-0.j, 0.+0.j, -0.+0.j, 0.-1.j],
         [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
         [0.+0.j, 0.+1.j, -0.+0.j, 0.+0.j]],
        [[0.+0.j, 0.+0.j, 0.+-0.j, 1.+0.j],
         [0.+0.j, 0.+0.j, 0.+1.j, -0.+0.j],
         [0.+-0.j, 0.-1.j, 0.+0.j, -0.+0.j],
         [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]]],dtype=torch.complex64)

######## return stabilizer mps given bond dimension ########
def stabilizer_tensor(gs,ps,i):
    n=len(gs[0])//2
    #define range for each generator
    ranges=[pauli_range(gen) for gen in gs]
    #determine which generators overlap at a site
    overlaps=[k for k in range(n) if i in ranges[k]]
    if len(overlaps)>0 :
        #construct tensors
        tensors=[projtensor(gs[j],ps[j],i) for j in overlaps]
        fv=fusion_vertex()
        result=reduce(lambda x,y:pfuse(x,y,fv), tensors)
        return result
    else :
        return torch.tensor([[[1,0,0,0]]],dtype=torch.complex64)

######## compute snapshot fidelity given reference and snapshot states ########
def snapshot_fidelity(rho1,rho2,rmps,rbdry):
    #define r-coefficient mps contracted with 
    ijk=torch.tensor([[[float(i==j==k) for i in range(4)] for j in range(4)] for k in range(4)],dtype=torch.complex64)
    v=torch.tensor([[[float(i==0)*float(j==0),float(i==0),float(j==0),1.0] for i in range(4)] for j in range(4)]).reshape(16,4)
    mps3=torch.tensordot(rmps.mats[0],v.transpose(0,1),dims=1).type(torch.complex64)
    #get local tensors for stabilizer states
    t1l,t1r=stabilizer_tensor(rho1.gs,rho1.ps,0),stabilizer_tensor(rho1.gs,rho1.ps,1)
    t2l,t2r=stabilizer_tensor(rho2.gs,rho2.ps,0),stabilizer_tensor(rho2.gs,rho2.ps,1)
    #merge w/ delta index and combine unit cell to match r formatting
    tt=coarse_grain(pfuse(torch.conj(t1l),t2l,ijk),pfuse(torch.conj(t1r),t2r,ijk))
    D3L,D3R=tt.shape[0]*rmps.aux_dim,tt.shape[1]*rmps.aux_dim
    #contract w/ r
    tmat=torch.tensordot(mps3,tt,dims=([2],[2])).transpose(1,2).reshape(D3L,D3R)
    for i in range(1,rho1.N//2):
        #get local tensors for stabilizer states
        t1l,t1r=stabilizer_tensor(rho1.gs,rho1.ps,2*i),stabilizer_tensor(rho1.gs,rho1.ps,2*i+1)
        t2l,t2r=stabilizer_tensor(rho2.gs,rho2.ps,2*i),stabilizer_tensor(rho2.gs,rho2.ps,2*i+1)
        #merge w/ delta index and combine unit cell to match r formatting
        tt=coarse_grain(pfuse(torch.conj(t1l),t2l,ijk),pfuse(torch.conj(t1r),t2r,ijk))
        D3L,D3R=tt.shape[0]*rmps.aux_dim,tt.shape[1]*rmps.aux_dim
        #contract w/ r
        mat=torch.tensordot(mps3,tt,dims=([2],[2])).transpose(1,2).reshape(D3L,D3R)
        tmat=tmat @ mat
    if rbdry == None :
        return tmat.diagonal(0).sum()
    else :
        tmat=tmat @ kron(rbdry,torch.eye(D3R//rmps.aux_dim)).type(torch.complex64)
        return tmat.diagonal(0).sum()

######## return r-coefficients for layer of k-local unitaries ########
def krickwall_r_matrix(n,k,i):  
    rmats=torch.zeros((1 if i%k==0 else 2, 1 if i%k==(k-1) else 2, 2))
    rmats[0,0,0]=-1.0 if i%k==0 else 1.0
    rmats[-1,-1,1]=(1.+2**(-k))**(1/k)
    return rmats

######## compute snapshot fidelity for k-local circuit ########
def krickwall_snapshot_fidelity(rho1,rho2,k):
    #define relevant tensors
    ijk=torch.tensor([[[float(i==j==k) for i in range(4)] for j in range(4)] for k in range(4)],dtype=torch.complex64)
    v=torch.tensor([[1,1],[0,1],[0,1],[0,1]],dtype=torch.float64)
    rmat=krickwall_r_matrix(rho1.N,k,0)
    #define r-coefficient mps contracted with 
    mps3=torch.tensordot(rmat,v.transpose(0,1),dims=1).type(torch.complex64)
    #get local tensors for stabilizer states
    t1,t2=stabilizer_tensor(rho1.gs,rho1.ps,0),stabilizer_tensor(rho2.gs,rho2.ps,0)
    #merge w/ delta index and combine unit cell to match r formatting
    tt=pfuse(torch.conj(t1),t2,ijk)
    D3L,D3R=tt.shape[0]*rmat.shape[0],tt.shape[1]*rmat.shape[1]
    #contract w/ r
    tmat=torch.tensordot(mps3,tt,dims=([2],[2])).transpose(1,2).reshape(D3L,D3R)
    for i in range(1,rho1.N):
        rmat=krickwall_r_matrix(rho1.N,k,i)
        #define r-coefficient mps contracted with 
        mps3=torch.tensordot(rmat,v.transpose(0,1),dims=1).type(torch.complex64)
        #get local tensors for stabilizer states
        t1,t2=stabilizer_tensor(rho1.gs,rho1.ps,i),stabilizer_tensor(rho2.gs,rho2.ps,i)
        #merge w/ delta index and combine unit cell to match r formatting
        tt=pfuse(torch.conj(t1),t2,ijk)
        D3L,D3R=tt.shape[0]*rmat.shape[0],tt.shape[1]*rmat.shape[1]
        #contract w/ r
        tmat=tmat @ torch.tensordot(mps3,tt,dims=([2],[2])).transpose(1,2).reshape(D3L,D3R)

    return tmat.diagonal(0).sum()
    