#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 09:37:39 2021

@author: yzy

Matrix Product State (MPS) (using PBC)

Host MPS matrices, providing methods for MPS inner product and merging
        
Arguments: mats (TorchTensor) - MPS tensors of shape 
    (unit-cell size, auxiliary dim, auxiliary dim, physical dim)
    
"""

#import math
#import itertools
import torch
#import torch.nn as nn
#import torch.nn.functional as F
#import torch.distributions as dist
#import torch.optim as optim

class MPS(object):
    
    def __init__(self, mats):
        if isinstance(mats, list): # convert list to torch tensor
            self.mats = torch.tensor(mats)
        else:
            self.mats = mats
        self.cell_size = self.mats.shape[0]
        assert self.mats.shape[1] == self.mats.shape[2], 'Auxiliary dimensions must match, got {} and {}.'.format(self.mats.shape[1],self.mats.shape[2])
        self.aux_dim = self.mats.shape[1]
        self.phy_dim = self.mats.shape[3]
            
    def __repr__(self):
        return 'MPS({} tensor of {})'.format(self.cell_size,'x'.join(str(d) for d in self.mats.shape[-3:]))
    
    def to(self, *args, **kwargs):
        self.mats = self.mats.to(*args, **kwargs)
        return self
    
    def dot(self, other, cell_num=1, bdry=None):
        mats1 = self.mats.view(self.cell_size, self.aux_dim, 1, self.aux_dim, 1, self.phy_dim)
        mats2 = other.mats.view(other.cell_size, 1, other.aux_dim, 1, other.aux_dim, other.phy_dim)
        tmats = torch.sum(mats1 * mats2, -1)
        cell_size = tmats.shape[0]
        aux_dim = self.aux_dim * other.aux_dim
        tmats = tmats.view(cell_size, aux_dim, aux_dim)
        if cell_size == 1:
            tmat = tmats[0]
        else:
            tmat = torch.eye(aux_dim).to(tmats)
            for k in range(cell_size):
                tmat = tmat.mm(tmats[k])
        if bdry == None : # allow boundary matrix
            return tmat.matrix_power(cell_num).trace()
        else :
            return tmat.matrix_power(cell_num).mm(bdry).trace()
    
    def dot2(self, other, cell_num=1, bdry=None): #requires cell sizes to divide eachother 
        ttype = torch.complex64 if (self.mats.dtype==torch.complex64 or other.mats.dtype==torch.complex64) \
            else self.mats.dtype
        if self.mats.shape[0] <= other.mats.shape[0] :
            tA=self.mats.type(ttype)
            tB=other.mats.type(ttype)
        else :
            tA=other.mats.type(ttype)
            tB=self.mats.type(ttype)
        DA,csA=tA.shape[1],tA.shape[0]
        DB,csB=tB.shape[1],tB.shape[0]
        tmat=torch.eye(DA*DB).to(other.mats) #put on same device as other
        for site in range(csB):
            mat=torch.tensordot(tA[site%csA],tB[site],dims=([2],[2])).transpose(1,2).reshape(DA*DB,DA*DB)
            tmat=tmat.type(ttype) @ mat
        if bdry == None : # allow boundary matrix
            return MPS.matrixpower(tmat,cell_num).diagonal(0).sum()
        else :
            return MPS.matrixpower(tmat,cell_num).mm(bdry).diagonal(0).sum()
        
    def merge(self, other, vertex):
        mats1 = self.mats.view(self.cell_size, self.aux_dim, 1, self.aux_dim, 1, self.phy_dim, 1)
        mats2 = other.mats.view(other.cell_size, 1, other.aux_dim, 1, other.aux_dim, 1, other.phy_dim)
        pmats = torch.tensordot(mats1 * mats2, vertex, dims=2)
        cell_size = pmats.shape[0]
        aux_dim = self.aux_dim * other.aux_dim
        pmats = pmats.view(cell_size, aux_dim, aux_dim, -1)
        return MPS(pmats)
    
    def component(self, alphas, bdry=None):
        mat = torch.eye(self.aux_dim).to(self.mats)
        for i, alpha in enumerate(alphas):
            mat = mat.mm(self.mats[i%self.cell_size, :, :, alpha])
        if bdry == None : # allow boundary matrix
            return mat.diagonal(0).sum()
        else :
            return mat.mm(bdry).diagonal(0).sum()
        
    @staticmethod
    def matrixpower(a,n):
        return MPS.mp(a,1,a,n)
    
    @staticmethod
    def mp(a,k,ak,n):
        if n==k: 
            return ak
        elif 2*k<=n:
            return mp(a,2*k,ak@ak,n)
        else:
            return ak@mp(a,1,a,n-k)
