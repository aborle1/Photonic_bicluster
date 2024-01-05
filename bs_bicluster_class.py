import numpy as np
import networkx as nx
from scipy.linalg import sqrtm

import perceval as pcvl
from perceval.algorithm import Sampler

class BS_bicluster:
    def __init__(self,dataset:np.ndarray,token_file=None):
        self.dataset = dataset
        #Get rows and columns of the original dataset
        self.original_rows = dataset.shape[0]
        self.original_cols = dataset.shape[1]
        
        
        #see if dataset (a matrix) is rectangle or square
        if self.dataset.shape[0] == self.dataset.shape[1]:
            issquare = True
        else:
            issquare = False
            self.make_data_square() #Basically, pad rows or columns (whichever is less) of 1s
        
        #Prepare the unitary matrix that nests the dataset
        self.U, self.c = self.to_unitary(dataset) #U is the unitary matrix where the first original_rows and original_cols is the original dataset (scaled by 1/c) 
        
        if token_file != None: #Meaning that backend will be remote
            self.token = open(token_file,'r').read()
            self.remote_backend = True
        else:
            self.remote_backend = False
            
        
        
    def prepare_backend(self,backend_name='CliffordClifford2017'):
        if self.remote_backend ==  True:
            self.QPU = pcvl.RemoteProcessor(backend_name,self.token)
        elif self.remote_backend == False:
            self.QPU = pcvl.Processor(backend_name)
        
    def prep_input(self,col_idx:list):
        
        #Use col_idx to prepare inputstate_raw (i.e inputstate as a list)
        inputstate_raw = [0 for i in range(self.U.shape[1])]
        
        for item in col_idx:
            inputstate_raw[item] = 1 #1 photon for the index position inside col_idx
        
        inputstate = pcvl.BasicState(inputstate_raw)
        return inputstate
        

    
    def boson_sampling(self,col_idx:list,shots): #prepare photonic circuit
        #Building block for our circuit
        self.mzi = (pcvl.BS() // (0, pcvl.PS(phi=pcvl.Parameter("φ_a")))
       // pcvl.BS() // (1, pcvl.PS(phi=pcvl.Parameter("φ_b"))))
        
        #Create circuit from U
        boson_circuit = pcvl.Circuit.decomposition(self.U, self.mzi,
                                               phase_shifter_fn=pcvl.PS,
                                               shape="triangle",allow_error=True)
        
        #Preparing input_state
        inputstate = self.prep_input(col_idx)
        print("Input state is: ",inputstate)
        #Now do the boson sampling for the input state
        #set circuit and inputs
        self.QPU.set_circuit(boson_circuit)
        self.QPU.with_input(inputstate)
        #create a sampler object
        sampler = Sampler(self.QPU)
        #finally, let us do the boson sampling

        self.raw_samples = sampler.samples(shots)
        return self.raw_samples
        
    
    
    
    def make_data_square(self):
        pass #To be expanded upon later
    
    
    def to_unitary(self,A): #Code from the people at quandela. Paper: arXiv:2301.09594
        ''' Input: graph A either as:
                                    an adjacency matrix of size mxm
                                    a networkX graph with m nodes
            Output: unitary with size 2mx2m, largest singular value
        '''
    
        if type(A) == type(nx.Graph()):
            A = nx.convert_matrix.to_numpy_matrix(A)
        P1, D, V = np.linalg.svd(A)
    
        c = np.max(D)
        # if it is not complex, then np.sqrt will output nan in complex values
        An = np.matrix(A/c, dtype=complex)
        P = An
        m = len(An)
        Q = sqrtm(np.identity(m)-np.dot(An, An.conj().T))
        R = sqrtm(np.identity(m)-np.dot(An.conj().T, An))
        S = -An.conj().T
        Ubmat = np.bmat([[P, Q], [R, S]])
        return (np.copy(Ubmat), c)

    def get_probability(self,outputstate:list):
        #Only to be done AFTER boson sampling
        outputstate_str = str(pcvl.BasicState(outputstate)) #outputstate
        total_count = len(self.raw_samples['results'])

        output_count = 0
        
        for item in self.raw_samples['results']:
            if outputstate_str == str(item):
                output_count += 1

        return output_count/total_count