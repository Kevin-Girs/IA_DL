from models.IA import IA
from views.home_view import HomeView

class IAController:
    
    def __init__(self,layers,inputs_FT,outputs_FT,nb_iter_FT,inputs_test) -> None:
        self.__layers = layers
        self.__ia = IA(self.__layers)
        self.__inputs_FT = inputs_FT
        self.__outputs_FT = outputs_FT
        self.__nb_iter_FT = nb_iter_FT
        self.__inputs_test = inputs_test
        self.__graph_view = HomeView()
        self.__train()
        
        
    def __train(self,):
        for _ in range(self.__nb_iter_FT):
            for i in range(len(self.__inputs_FT)):
                if self.__ia.propagation(self.__inputs_FT[i]):
                    if self.__ia.retro_propagation(self.__outputs_FT[i]):
                        print(_)
                        if _ % 1000 == 0:
                            self.__graph_view.add_in_graph(_, "{0:.3f}".format(float(self.__ia.get_cost())))
                        
                        
        self.__ia.set_state_of_training(False)
        
        print("TEST :")
        
        if self.__ia.is_in_training:
            for _ in range(len(self.__inputs_test)):
                if self.__ia.propagation(self.__inputs_test[_]):
                    predicted_output = self.__ia.get_output()
                    print("Input :", self.__inputs_test[_], "Prediction :", predicted_output)
        else:
            print("Training seems to be never over !")
        
        self.__graph_view.view_of_graph()
