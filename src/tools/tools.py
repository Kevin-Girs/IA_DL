import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))
    
def sigmoidPrime(x):
    return x*(1-x)