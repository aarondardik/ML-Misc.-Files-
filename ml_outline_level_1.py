##Basic Outline for High Level Approach to Backpropagation
import numpy as np
import pickle
import theano
from numpy import random
import pandas as pd 


class Network(object):
    def __init__(self, sizes, cost=QuadraticCost):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weight_initialization()
        self.cost = cost
    def weight_initialization(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]] 
        self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(self.sizes [:-1], self.sizes[1:])]
    
    '''The three important functions here are feedforward, stochastic descent and backpropagation'''
    def feedforward(self, a):
        #Return network output if a is the input
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a
    
    def stochastic_descent(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        #function to run the stochastic gradient descent method as per
        #Wikistat 5-6
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0}: Complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases] 
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch: 
            delta_nabla_b, delta_nabla_w = self.backpropagation(x, y) 
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)] 
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)] 
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]
    
    def backpropagation(self, x, y):
        #Similarly uses Wikistat 5-6. This method is used to compute the gradient at each level. 
        nabla_b = [np.zeros(b.shape) for b in self.biases] 
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward 
        activation = x 
        activations = [x] # list to store all the activations, layer by layer 
        zs = [] # list to store all the z vectors, layer by layer 
        for b, w in zip(self.biases, self.weights): 
            z = np.dot(w, activation)+b
            zs.append(z) 
            activation = sigmoid(z) 
            activations.append(activation)
        #backward pass 
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1]) 
        nabla_b[-1] = delta 
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in xrange(2, self.num_layers): 
            z = zs[-l] 
            sp = sigmoid_prime(z) 
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp 
            nabla_b[-l] = delta 
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose()) 
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
    
    def accuracy(self, input_vector):
        #Return the accuracy of the network as represented
        #by the portion of inputs that are correcty classified
        return None
    
    def total_cost(self, input_vector):
        #Return the total cost associated with
        #the network, note the choice of cost function is important
        return None
    
    
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))
    
    def sigmoid_prime(z): 
        return sigmoid(z)*(1-sigmoid(z))


#Sample cost method using the average norm
class QuadraticCost(object):
    @staticmethod 
    def fn(a, y): 
        return 0.5*np.linalg.norm(a-y)**2
    @staticmethod 
    def delta(z, a, y): 
        return (a-y) * sigmoid_prime(z)


##If a Convolutional Layer is Needed
class ConvolutionalLayer(object):
    def __init__(self):
        self.params = None


if __name__ == "__main__":
    
    num_layers = input("How many layers will be in your neural network?")
    sizes = []
    for (i in range(num_layers)):
        sizes.append(input("How may neurons will layer {0} have?".format(i)))
    
    neuralNet = Network(sizes)
        
                       
    ##THE BELOW PART OF MAIN IS UNFINISHED
    # load the data from the file
    data = load_data("C:/Users/aarondardik/Desktop/marks.txt", None)

    # X = feature values, all the columns except the last column
    X = data.iloc[:, :-1]

    # y = target values, last column of the data frame
    y = data.iloc[:, -1]

    
