#
import numpy as np
import pandas as pd 
import matplotlib as mpl 
import matplotlib.cm as cm 
import matplotlib.pyplot as plt 


class Linear_Regression: 
    def __init__(self, solutions, features):
        self.solutions = solutions
        self.features = features 
        self.theta = np.zeros(features.shape[1])

    def learn_theta(self, learning_rate, num_loops):
        num_rows = self.features.shape[0]
        num_cols = self.features.shape[1]
        theta = np.zeros(num_cols)

        #here i create a "stored_values" matrix which contains the values of theta
        #as i move through the loop...
        stored_values = np.zeros([num_loops, theta.size])

        for loop_num in range(num_loops):
            for j in range(theta.size):
                total = 0
                for i in range(num_rows):
                    total += (self.solutions[i] - np.dot(self.features[i], theta))*self.features[i, j]
                theta[j] += learning_rate * total
                stored_values[loop_num, j] = theta[j]
            #print("iteration number:{}".format(loop_num))
            #print(theta)
        self.theta = theta 
        return theta, stored_values
    
    def plot_projection_graph(self, learning_rate, num_loops, projection_axis):
        theta_ , values_ = self.learn_theta(learning_rate, num_loops)
        n_steps = np.linspace(1, num_loops, num=num_loops) 
        fig, ax = plt.subplots()
        plt.plot(n_steps, values_[:, projection_axis], label='Regression')
        plt.show()  
    
    #ONLY CALL THIS AFTER WE HAVE ALREADY "LEARNED THETA"
    def predict_value(self, x_point):
        #if we've already accounted for x_0 being equal to 1
        if (x_point[0] == 1 and x_point.size == self.theta.size):
            return np.dot(self.theta, x_point)
        
        elif (x_point[0] != 1 and x_point.size == self.theta.size - 1):
            a = np.array([1])
            x_point = np.concatenate((a, x_point), axis=0)
            return np.dot(self.theta, x_point)

        else:
            print("x is of invalid shape for prediction")


class Weighted_Linear_Regression:
    #if our features matrix has a constant term, (first column = 1), then point[0]
    #must be = 1 for this to make sense
    def __init__(self, solutions, features, point):
        self.solutions = solutions
        self.features = features
        self.theta = np.zeros(features.shape[1])
        self.weights = np.zeros([features.shape[0]])
        self.point = point 
   
    def set_weights(self, tau):
        if (self.point.shape[0] == self.features.shape[1]):
            weights = np.zeros([self.features.shape[0]])
            for i in range(weights.size):
                weights[i] = np.exp(-np.dot(np.transpose(self.features[i]-self.point), (self.features[i]-self.point))/2*tau**2)
        else:
            print("x has dimensions{}, while the solution matrix has dimensions{}".format(self.point.shape[0], self.features.shape))
        self.weights = weights
        return weights 
    
    def learn_theta_at_point(self, learning_rate, num_loops, tau):
        #fill in the function - specifically update gradient to account for the weights...
        num_rows = self.features.shape[0]
        num_cols = self.features.shape[1]
        theta = np.zeros(num_cols)

        stored_values = np.zeros([num_loops, theta.size])
        weight_at_point = self.set_weights(tau)

        for loop_num in range(num_loops):
            for j in range(theta.size):
                total = 0
                for i in range(num_rows):
                    total += (self.solutions[i] - np.dot(self.features[i], theta))*self.features[i, j]
                theta[j] += learning_rate * weight_at_point[j] * total
                stored_values[loop_num, j] = theta[j]
        self.theta = theta 
        return theta, stored_values
    
    def plot_projection_graph(self, learning_rate, num_loops, tau, projection_axis):
        theta_ , values_ = self.learn_theta_at_point(learning_rate, num_loops, tau)
        n_steps = np.linspace(1, num_loops, num=num_loops) 
        fig, ax = plt.subplots()
        plt.plot(n_steps, values_[:, projection_axis], label='Regression')
        plt.show()  

     #ONLY CALL THIS AFTER WE HAVE ALREADY "LEARNED THETA"
    
    def predict_value(self):
        #if we've already accounted for x_0 being equal to 1
        if (self.point[0] == 1 and self.point.size == self.theta.size):
            return np.dot(self.theta, self.point)

        elif (self.point[0] != 1 and self.point.size == self.theta.size - 1):
            a = np.array([1])
            x_point = np.concatenate((a, self.point), axis=0)
            return np.dot(self.theta, self.point)

        else:
            print("x is of invalid shape for prediction, or the presense / lack of a constant term is off")


y = np.array([1, 3, 2, 7, 8, 6, 10, 12])
x = np.array([[1, 3, 2], [1, 4, 1], [1, 0, -1], [1, 5, 6], [1, -2, 3], [1, 8, 8], [1, 7, 6], [1, 0, 2]])
p = np.zeros([3])
rate = 1 / 5000
n_loo = 10000

#reg_obj = Linear_Regression(y, x)
#reg_obj.learn_theta(rate, n_loo)
#yp = reg_obj.predict_value(np.array([0, 1]))
#reg_obj.plot_projection_graph(rate, n_loo, 2)


p1 = np.array([1, -1.3, 0.7])
t = 1/3 
weighted_obj = Weighted_Linear_Regression(y, x, p1)
weighted_obj.learn_theta_at_point(rate, n_loo, t)
ab = weighted_obj.predict_value()
print(ab)
#weighted_obj.plot_projection_graph(rate, n_loo, t, 1)

