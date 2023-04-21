#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 10:40:19 2021

@author: aaa,yzy,hyh
    
vMPSSolver - a model to learn the best D bond-dimensional MPS to approximate
    a given MPS. 
        
Arguments: state (MPS) - initial state, aux_dim (int) - the auxilary dimension
 of the MPS representation of the approximate state, cell_num - number of cells
 
Explanation: Minimize the squared difference of the two MPS

"""

from MPS import MPS
import math
import torch
import torch.nn as nn

class vMPSSolver(nn.Module):
    
    def __init__(self, phys_dim:int=2, aux_dim:int=2, unit_cell:int=2, init_mats=None):
        super(vMPSSolver, self).__init__()
        self.phys_dim = phys_dim
        self.aux_dim = aux_dim
        self.unit_cell = unit_cell
        if init_mats==None:
            mats=torch.randn(unit_cell,aux_dim,aux_dim,phys_dim)
        else :
            mats=init_mats
        self.mats=nn.Parameter(mats)
           
    def loss(self, state:MPS, cell_num:int=1):
        approx=MPS(self.mats)
        return approx.dot(approx, cell_num=cell_num)+state.dot(state, cell_num=cell_num)-2*state.dot(approx,cell_num=cell_num)
        
