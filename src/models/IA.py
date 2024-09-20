import numpy as np
from tools.tools import *

class IA:
    
    def __init__(self, layers) -> None:
        if len(layers) > 1:
            self.__layers = layers
            self.__initialized = False
            self.__initialized = self.__init_network()
            self.__is_in_training = True
            if self.__initialized:
                print("Network initialized.")
                self.__back_propagation_is_ready = False
                self.__cost = 0
        else:
            raise Exception("There must be at least two rows of neurons.")

    def __bool__(self,) -> bool:
        return self.__initialized
    
    def __init_network(self,) -> bool:
        return self.__init_neurons() and self.__init_weights()
    
    def __init_neurons(self,) -> bool:
        if self.__initialized is False:
            print("Initialization of neurons.")
            self.__neurons = [np.array([0] * nb_neurones, dtype=float) for nb_neurones in self.__layers]
            return True
        return False
        
    def __init_weights(self,) -> bool:
        if self.__initialized is False:
            print("Initialization of weights.")
            self.__weights = [self.__glorot_initialization(len(self.__neurons[i]), len(self.__neurons[i+1])) for i in range(len(self.__neurons) - 1)]
            return True
        return False
    
    def get_output(self,) -> []:
        return self.__neurons[-1]
    
    def is_in_training(self,) -> bool:
        return self.__is_in_training
    
    def set_state_of_training(self,val) -> None:
        self.__is_in_training = val
        
    def get_cost(self,) -> float:
        return self.__cost
    
    def propagation(self, inputs) -> bool:
        if (self.__initialized and (len(self.__neurons[0]) == len(inputs))) and (self.__back_propagation_is_ready is not True or self.__is_in_training):
            self.__neurons[0] = np.array(inputs, dtype=float)
            for i in range(1, len(self.__neurons)):
                self.__neurons[i] = sigmoid(np.dot(self.__neurons[i-1], self.__weights[i-1]))       
            self.__back_propagation_is_ready = False if self.__is_in_training is not True else True
            return True    
        else:
            raise Exception("...")  
            
    def retro_propagation(self, target) -> str:
        if (self.__initialized and len(self.__neurons[-1]) == len(target)) and (self.__back_propagation_is_ready or self.__is_in_training):
            __output_error = target - self.__neurons[-1]
            self.__cost = str(np.mean(np.abs(__output_error)))
            __output_gradient = __output_error * sigmoidPrime(self.__neurons[-1])
            self.__weights[-1] += np.outer(self.__neurons[-2], __output_gradient)
            for i in reversed(range(len(self.__weights) - 1)):
                __hidden_error = np.dot(self.__weights[i + 1], __output_gradient)
                __hidden_gradient = __hidden_error * sigmoidPrime(self.__neurons[i + 1])
                self.__weights[i] += np.outer(self.__neurons[i], __hidden_gradient)
                __output_gradient = __hidden_gradient
            self.__back_propagation_is_ready = False
            return True
        else:
            raise Exception("...")

    def __glorot_initialization(self, input_size, output_size) -> np.ndarray:
        __limit = np.sqrt(6 / (input_size + output_size))
        __weights = 2 * __limit * np.random.random((input_size, output_size)) - __limit
        return __weights
        
    # def normalize_data(data) -> float:
    #     min_val = np.min(data)
    #     max_val = np.max(data)
    #     normalized_data = -1 + 2 * (data - min_val) / (max_val - min_val)
    #     return normalized_data

    # def relu(x):
    #     return np.maximum(0, x)

    # def he_initialization(input_size, output_size):
    #     limit = np.sqrt(2 / input_size)  # Ajustement pour ReLU
    #     weights = np.random.normal(0, limit, (input_size, output_size))
    #     return weights