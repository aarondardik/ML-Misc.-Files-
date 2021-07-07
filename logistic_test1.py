# imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def load_data(path, header):
    marks_df = pd.read_csv(path, header=header)
    return marks_df



#my functions..
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def train_losistic(solutions, features, num_steps, rate):
    num_rows = features.shape[0]
    num_cols = features.shape[1]
    theta = np.zeros(num_cols)

    #prev_theta = np.zeros(theta.size)
    for loop_num in range(num_steps):
        for j in range(theta.size):
            total = 0
            for i in range(num_rows):
                total += (solutions[i] - sigmoid(np.dot(features[i], theta)))*features[i, j]
            theta[j] += rate * total
            #prev_theta = theta
    return theta 
#///



if __name__ == "__main__":
    # load the data from the file
    data = load_data("C:/Users/aarondardik/Desktop/marks.txt", None)

    # X = feature values, all the columns except the last column
    X = data.iloc[:, :-1]

    # y = target values, last column of the data frame
    y = data.iloc[:, -1]

    # filter out the applicants that got admitted
    admitted = data.loc[y == 1]

    # filter out the applicants that din't get admission
    not_admitted = data.loc[y == 0]

    # plots
    plt.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1], s=10, label='Admitted')
    plt.scatter(not_admitted.iloc[:, 0], not_admitted.iloc[:, 1], s=10, label='Not Admitted')
    plt.legend()
    plt.show()