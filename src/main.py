from controllers.ia_controller import IAController

if __name__ == "__main__":
    
    __layers = [2, 7, 1]
    
    __inputs = [
       [0,0],
       [0,1],
       [1,0],
       [1,1] 
    ]

    __outputs = [
        [1],
        [0],
        [0],
        [1]
    ]
    
    __inputs_test = [
        [0,1],
        [1,0],
        [0,0],
        [1,1]
    ]
    
    __nb_iter = 10000

    __IAController = IAController(layers=__layers,inputs_FT=__inputs,outputs_FT=__outputs,nb_iter_FT=__nb_iter,inputs_test=__inputs_test)
    

    