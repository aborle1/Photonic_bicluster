import numpy as np
import strawberryfields as sf
from strawberryfields.apps import sample
from strawberryfields import decompositions
import networkx as nx

class GBS_cluster:
    def __init__(self,dataset:np.ndarray,token_file=None):
        self.dataset = dataset
        #Get rows and columns of the original dataset
        self.original_rows = dataset.shape[0]
        self.original_cols = dataset.shape[1]

        #Get adjacency matrix out of this
        #Create an all-zero matrix
        self.dataset_adj_matrix = np.zeros((self.dataset.shape[0]+self.dataset.shape[1],self.dataset.shape[0]+self.dataset.shape[1]))

        #convert data into adjacency matrix
        for i in range(self.dataset.shape[0]):
            for j in range(self.dataset.shape[1]):
                self.dataset_adj_matrix[i,j+self.dataset.shape[0]] = self.dataset[i,j]

        #Create symmetry
        self.dataset_adj_matrix = self.dataset_adj_matrix + self.dataset_adj_matrix.T
         
    def gb_sampling(self, n_mean,shots,threshold=True):
        self.raw_samples = sample.sample(self.dataset_adj_matrix, n_mean, shots, threshold=threshold)

    def save_samples(self,filepath):
        #Only to be done AFTER boson sampling
        #Unlike boson sampling, this one saves to a npy file (easier)
        np.save(filepath, self.raw_samples)