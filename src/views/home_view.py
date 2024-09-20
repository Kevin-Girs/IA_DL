import matplotlib.pyplot as plt

class HomeView:
    
    def __init__(self) -> None:
        self.__xGraphCoatFunction = []
        self.__yGraphCoatFunction = []
        
    def add_in_graph(self,idx,val):
        self.__xGraphCoatFunction.append(idx)
        self.__yGraphCoatFunction.append(val)
    
    def view_of_graph(self,):
        plt.plot(self.__xGraphCoatFunction, self.__yGraphCoatFunction)
        plt.show()
    
    