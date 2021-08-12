#stuff for RNN

import numpy as np 
from numpy.random import randn

class RNN:
    def __init__(self, input_size, output_size, hidden_size=64):
        self.Whh = randn(hidden_size, hidden_size) / 1000
        self.Wxh = randn(hidden_size, input_size) / 1000
        self.Why = randn(output_size, hidden_size) / 1000

        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))
    

    def forward(self, inputs):
        '''
        Perform a forward pass of the RNN for given inputs. Return the final output, y_T and hidden
        state, h_T. 
        '''
        #h = ...
        # for loop here to loop over t
        # y = ...
        
        #return y, h


    #dy is dL/dy in the input below
    def backprop(self, dy, learn_rate=2e-2):
        '''
        '''
        #dL / dW_hy =...
        #dL / db_y = ...

        #Initialize dL/dWhh, dL/dWxh, dL/dbh to zeros of appropriate shape

        #find dL/dh for the last h, i.e. h_T

        #backpropogate through time...
    
    def train(self, num_loops):
        #train the network by running the backpropagation algorithm n_loops times...


    

    







