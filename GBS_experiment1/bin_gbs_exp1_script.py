import numpy as np
import strawberryfields as sf
from strawberryfields.apps import sample
from strawberryfields import decompositions
import networkx as nx
from gbs_bicluster_class import GBS_cluster

#Firstly, read the problem file
problem_filepath = './problems/gbs_exp1_part1.npz'

fp = np.load(problem_filepath)
dataset = fp['arr_0']

# mean photons and shots
n_mean = 1 * 24 #24 is the number of modes, sf.apps.sample.sample divides n_mean by the total no of modes
shots = 10000

#Create an object of GBS_cluster type and load the dataset
gbs_cluster = GBS_cluster(dataset)
#Make data binary
gbs_cluster.make_data_binary()

print("---Now sampling for:",problem_filepath,"---")
print("total (mean) photons:",n_mean)
print("shots:",shots)
print("------")


gbs_cluster.gb_sampling(n_mean,shots)

results_filepath = './results/' + 'resbin_gbs_exp1_photon'+str(n_mean)+'_shots'+str(shots)+'.npy'
gbs_cluster.save_samples(results_filepath)

