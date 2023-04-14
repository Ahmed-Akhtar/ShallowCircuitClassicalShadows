#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 10:40:19 2021

@author: aaa,yzy,hyh
    
HaarReconstructionChannel - a model to learn the reconstruction channel from 
    the snapshot entanglement feature as an MPS. Utilizes extra symmetry for
    circuits evolved by brickwall Haar random unitaries.
        
Arguments: d (int) - qudit dimension, aux_dim (int) - the auxilary dimension
 of the MPS representation of the reconstruction coefficient
 
Explanation: In this version, we explicitly reinforce the symmetry of the Haar
circuit, which says that within each unit cell only r_00 and r_11 are nonzero.
This makes the effective cell size of the r vector one.

Furthermore, we re-parameterize the variables so that instead of learning directly
the r-mats, we learn $\lambda$ the (normalized) eigenvalues and upper triangle 
of $A_{+}$ and the matrix $A_{-}$, where 

$A_{\pm} = d^2(r_{00}\pmr_{11})$

See note in shared folder for explanation.

"""

from MPS import MPS
import math
import torch
import torch.nn as nn
from EF_MPS_utils import spherical_to_cartesian, pgate, kron

class HaarReconstructionChannel(nn.Module):
    
    def __init__(self, d:int=2, aux_dim:int=2, params=None):
        super(HaarReconstructionChannel, self).__init__()
        assert d>1, 'qudit dimension must be greater than 1, got {}.'.format(d)
        self.d = d
        self.aux_dim = aux_dim
        
        if params==None:
            eigs,Am,ut=math.pi*torch.zeros(aux_dim-1),torch.randn(aux_dim,aux_dim), \
                torch.randn(aux_dim*(aux_dim-1)//2)
            #self.eigs=nn.Parameter(eigs,requires_grad=False)
            self.eigs=nn.Parameter(eigs)
            self.ut=nn.Parameter(ut)
            self.Am=nn.Parameter(Am)
        else :
            #self.eigs=nn.Parameter(params[0], requires_grad=False)
            self.eigs=nn.Parameter(params[0])
            self.ut=nn.Parameter(params[1])
            self.Am=nn.Parameter(params[2])
        
        f = torch.tensor([[[self.d,0.],[0.,0.]],[[self.d,-1.],[-1.,self.d]]])
        f[1,0,:] = f[1,0,:]*(self.d**2)/(self.d**2-1)
        f[1,1,:] = f[1,1,:]*(self.d)/(self.d**2-1)
        f2=torch.einsum('abc,def->adbecf',torch.transpose(f,1,2),torch.transpose(f,1,2)).view(4,4,4)
        self.f2 = nn.Parameter(f2, requires_grad=False) 
        
        mat = (pgate(1,1)+0.5*(pgate(0,3)+pgate(3,0)))/(2*self.d**2)    
        self.f2a = nn.Parameter(torch.tensordot(mat,f2,dims=1), requires_grad=False) #vertex for A mps
        
        dijk= torch.tensor([[[float(i==j==k) for i in range(2)] for j in range(2)] \
                            for k in range(2)]).reshape(2,4)
        self.dijk=nn.Parameter(dijk, requires_grad=False)
        
        self.dmats = nn.Parameter(torch.tensor([[[[0.,0.,0.,1.]]]]), 
                                  requires_grad=False) 
        self.d2 = MPS(self.dmats) 
    
    def loss(self, W:MPS, cell_num:int=1):
        An=torch.diag_embed(spherical_to_cartesian(self.aux_dim,cell_num,nn.functional.pad(self.eigs,(1,0),value=1.0)))
        An[torch.triu_indices(self.aux_dim,self.aux_dim,offset=1).unbind()]=self.ut
        mps1=MPS(torch.tensordot(torch.stack([An,self.Am],dim=2).view(1,self.aux_dim,self.aux_dim,2),self.dijk,dims=([3],[0])))
        mps2=MPS(torch.tensordot(W.mats[0],W.mats[-1],dims=([1],[0])).transpose(1,2).reshape(1,W.mats.shape[1],W.mats.shape[2],4))
        y2 = mps1.merge(mps2, self.f2a)
        return y2.dot(y2, cell_num)-2*y2.dot(self.d2, cell_num) + self.d2.dot(self.d2, cell_num) 
