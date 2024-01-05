#NOTE : CURRENTLY CANNOT HANDLE rectangular datasets or biclusters
import numpy as np
import networkx as nx
from scipy.linalg import sqrtm
import json

import perceval as pcvl
from perceval.algorithm import Sampler
from  perceval.utils.postselect import PostSelect

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
            #Not fully implemented yet
            issquare = False
            self.make_data_square() #Basically, pad rows or columns (whichever is less) of 1s
        
        #Prepare the unitary matrix that nests the dataset
        self.Utemp, self.c = self.to_unitary(dataset) #Utemp is the unitary matrix where the first original_rows and original_cols is the original dataset (scaled by 1/c) 
        
        #Sometimes Utemp is not good enough, so need to call closest_unitary to calculate U
        #such that ||Utemp - U|| is minimized
        self.U = self.closest_unitary(self.Utemp)
        
        self.unitary_error = np.linalg.norm(self.Utemp-self.U) #absolute error in the Unitary (wrt to the original unitary)
        
        if token_file != None: #Meaning that backend will be remote
            self.token = open(token_file,'r').read()
            self.remote_backend = True
        else:
            self.remote_backend = False
            
        
    def closest_unitary(self,A):
        """ 
        NOTE : Original code from https://michaelgoerz.net/notes/finding-the-closest-unitary-for-a-given-matrix/
        Calculate the unitary matrix U that is closest with respect to the
        operator norm distance to the general matrix A.
        
        Return U as a numpy matrix.
        """
        V, __, Wh = np.linalg.svd(A)
        U = V @ Wh 
        return U

    
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
        #allow_error is set to true because often it cannot find a circuit with error below precision threshold
        #max_try set to 100 as a heuristic
        self.boson_circuit = pcvl.Circuit.decomposition(self.U, self.mzi,
                                               phase_shifter_fn=pcvl.PS,
                                               shape="triangle",allow_error=True,max_try=100)
                                               
        self.reconU = self.boson_circuit.compute_unitary() #reconU is U reconstructed from circuit
        
        self.circuit_error = np.linalg.norm(self.U - self.reconU) #absolute error in the Unitary (wrt to the circuit)
        #Preparing input_state
        inputstate = self.prep_input(col_idx)
        print("Input state is: ",inputstate)
        #Now do the boson sampling for the input state
        #set circuit and inputs
        self.QPU.set_circuit(self.boson_circuit)
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
    
    def condition_builder(self,desired_modes, undesired_modes, max_photon_per_mode = 1):
        #Builds conditions for perceval's PostSelect
        #desired_modes and undesired_modes are mode numbers mentioned in the list
        tempstr = ''
        first = True

        #desired
        for item in desired_modes:
            if first == True:
                tempstr += '[' + str(item) + '] < ' + str(max_photon_per_mode + 1)
                first = False
            else:
                tempstr += ' & '
                tempstr += '[' + str(item) + '] < ' + str(max_photon_per_mode + 1)
        
        #undesired
        for item in undesired_modes:
            if first == True:
                tempstr += '[' + str(item) + '] == 0 '
                first = False
            else:
                tempstr += ' & '
                tempstr += '[' + str(item) + '] == 0'

        return tempstr

    def get_probability(self,outputstate:list,row_postselect=False,max_photon_per_mode=1):
        #Only to be done AFTER boson sampling
        #row_postselect will only calculate the denominator for the starting m rows of U
        #max_photon_per_mode (ONLY TO BE USED WITH row_postselect currently) is going to post-select for that value 
        outputstate_str = str(pcvl.BasicState(outputstate)) #outputstate
        
        if row_postselect == False:
            total_count = len(self.raw_samples['results'])
        else:
            total_count = 0 #we'll add to this total_count later on
            
            #get postselector ready
            desired_modes = [i for i in range(int(self.U.shape[0]/2))]
            undesired_modes = [i for i in range(int(self.U.shape[0]/2),self.U.shape[0])]
            print("Desired modes:",desired_modes)
            print("Undesired modes:",undesired_modes)
            
            condition_string = self.condition_builder(desired_modes,undesired_modes,max_photon_per_mode)
            ps = PostSelect(condition_string) #ps will come in handy in the next section

        output_count = 0
        
        for item in self.raw_samples['results']:
            if outputstate_str == str(item):
                output_count += 1
            if row_postselect == True:
                if ps(pcvl.BasicState(str(item))) == True:
                    total_count += 1
            

        return output_count/total_count, output_count,total_count # prob, numerator, denominator
        
    def save_samples(self,filepath):
        #Only to be done AFTER boson sampling
        raw_samples2 = self.raw_samples.copy()
        temp_list1 = raw_samples2['results']
        temp_list2 = []

        #going through temp_list1, converting each item to a string and storing it in temp_list2
        for item in temp_list1:
            temp_list2.append(str(item))

        #change the value for results
        raw_samples2['results'] = temp_list2

        #save raw_samples2 to filepath
        with open(filepath, 'w') as f:
            json.dump(raw_samples2, f)
        
