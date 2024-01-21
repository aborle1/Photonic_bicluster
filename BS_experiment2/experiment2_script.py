import numpy as np
import networkx as nx
from scipy.linalg import sqrtm
import time
import json
import perceval as pcvl
from perceval.algorithm import Sampler
from bs_bicluster_class import BS_bicluster
from thewalrus import perm

from suppl_functions import *
import copy

fp = np.load('./problems/bs_exp2_part1.npz')

dataset = fp['arr_0']
shuffle_row_idx = fp['arr_1']
shuffle_col_idx = fp['arr_2']
bicluster_row_idx = fp['arr_3']
bicluster_col_idx = fp['arr_4']

#Simulated Annealing 
#How many steps to anneal
p= 20

for ctr in range(0,20): # do it a multiple times for statistical purposes

    #Setting start and end temperatures
    #create an arange array out of the steps
    t = np.arange(p)
    #Set up initial and final temperatures
    T_i = 100 #T_i is 1/beta_i
    T_f = 0.01 #T_f is 1/beta_f
    #Considering an exponential decay schedule
    T_schedule = T_i * ((T_f/T_i) ** (t/p))
    shots = 10 ** 5

    cur_energy = None
    cur_row_idx = None
    cur_col_idx = None
    for tval in T_schedule:
        #Get the col_idx
        if cur_energy == None: #that means doing this for the first time
            #Initialize the entire col_idx to some starting config
            col_idx = np.sort(np.random.choice([i for i in range(0,12)], size=6, replace=False))
            print("Current selection:", col_idx)
        else:
            #select
            print("Previous accepted selection:",cur_col_idx_full)
            col_idx , mode_from, mode_to = next_neighbor(cur_col_idx_full,12)
            print("Current selection:", col_idx)
            print("From:", mode_from)
            print("To:", mode_to)

        bs_cluster = BS_bicluster(dataset)#,'quandela_token.txt')
        bs_cluster.prepare_backend()#('sim:clifford')
        _ = bs_cluster.boson_sampling(col_idx,shots)

        #Get the majority sample (and derive row_idx from it)
        row_state_string = bs_cluster.get_majority_sample(max_photon_per_mode=1)
        if row_state_string == None:
            cur_max_photon_per_mode = 1
            while row_state_string == None and cur_max_photon_per_mode < 5:
                cur_max_photon_per_mode += 1
                #increase the acceptable photons
                row_state_string = bs_cluster.get_majority_sample(max_photon_per_mode=cur_max_photon_per_mode)
            if row_state_string == None: #if row_state_string is still None
                print("None string")
                row_idx = None
                new_col_idx = None
                energy = -1000.00 #give it very bad energy
            else:
                row_idx = idx_from_string(row_state_string)
                new_col_idx = drop_columns(dataset,row_idx,col_idx)
                energy = permanent_of_submatrix(dataset,row_idx,new_col_idx)
        else:
            row_idx = idx_from_string(row_state_string)
            new_col_idx = copy.deepcopy(col_idx)
            energy = permanent_of_submatrix(dataset,row_idx, col_idx)

        print("-------")
        print("p=",p)
        print("tval = ",tval)
        print("col_idx :",new_col_idx)
        print("row_idx :",row_idx)
        print("energy :",energy)
        
        if cur_energy == None:
            cur_energy = energy
            cur_col_idx_full = copy.deepcopy(col_idx)
            cur_col_idx_refined = copy.deepcopy(new_col_idx) #if col_idx has been trimmed down, otherwise same as cur_col_idx_full
            cur_row_idx = copy.deepcopy(row_idx)
        else:
            ediff = energy - cur_energy
            rand_num = np.random.rand()
            if ediff > 0.0 or rand_num < np.exp( ediff/tval):
                #Accept move
                cur_energy = energy
                cur_col_idx_full = copy.deepcopy(col_idx)
                cur_col_idx_refined = copy.deepcopy(new_col_idx) #if col_idx has been trimmed down, otherwise same as cur_col_idx_full
                cur_row_idx = copy.deepcopy(row_idx)
                print("Move Accepted")
            else:
                print("Move Rejected")
                

    print("Final bicluster:")
    print("rows:",cur_row_idx)
    print("columns:",cur_col_idx_refined)
    print("energy:",cur_energy)
    
    op_filepath = "./results/res_p" + str(p)+"_"+ str(ctr) + ".npz"
    np.savez(op_filepath,cur_row_idx,cur_col_idx_refined,cur_energy)
    