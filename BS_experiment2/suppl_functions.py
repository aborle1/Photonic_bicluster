import numpy as np
import networkx as nx
from scipy.linalg import sqrtm
import time

import perceval as pcvl
from perceval.algorithm import Sampler
from bs_bicluster_class import BS_bicluster
from thewalrus import perm

def idx_from_string(string_state):
    list_state = list(pcvl.BasicState(string_state))
    idx = []

    for i in range(0,len(list_state)):
        if list_state[i] != 0:
            idx.append(i)
    return idx

def permanent_of_submatrix(A,row_idx,col_idx):
    temp_submatrix = A[np.ix_(row_idx,col_idx)]

    return perm(temp_submatrix)

def drop_columns(A,row_idx,col_idx):
    #useful when more than one photon is measured in one mode
    
     
    len_diff = len(col_idx) - len(row_idx)
    norm_and_idx = []
    
    for col in col_idx:
        norm_and_idx.append([col,np.linalg.norm(A[:,col])])
    
    norm_and_idx = np.array(norm_and_idx)
    norm_and_idx = norm_and_idx[norm_and_idx[:, 1].argsort()]
    
    new_col_idx = list(np.sort(norm_and_idx[len_diff:,0]).astype(int))
    return new_col_idx

def next_neighbor(col_idx,max_mode):
    #move one photon over from an allocated mode to an unallocated mode
    
    #Step 0: create a list of 'potential moves'
    potential_modes = [i for i in range(0,max_mode)]
    potential_modes = list(filter(lambda a: a not in col_idx, potential_modes)) #filter out modes already in col_idx
    
    #Step 1: pick up 1 mode for col_idx
    mode_from = np.random.choice(col_idx, size=1, replace=False)[0]
    
    #Step2: pick up 1 mode from potential_modes
    mode_to = np.random.choice(potential_modes, size=1, replace=False)[0]
    
    #Step 3: replace mode_from with mode_to
    col_idx = np.where(col_idx == mode_from, mode_to, col_idx)
    #sort it for good measured
    col_idx = np.sort(col_idx)
    
    return col_idx, mode_from,mode_to
    