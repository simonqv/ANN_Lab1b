import generate_data as gen
import matplotlib.pyplot as plt
import numpy as np

input_arr, target, classA, classB, init_w = gen.get_data()

def generalised_d_batch(init_v, init_w):# n_hidden is the number of nodes in the hidden layer
    h_in = np.dot(init_v, input_arr)
    h_out = 2/ (1 + np.exp(-h_in)) -1
    h_out = np.vstack((h_out, np.ones((1, len(h_out[0])))))
    print(h_out)
    return 0

def create_init_weights(n_hidden):
    #  init weight v from input to hidden layer is 3 (Xx, Xy, bias) times number of nodes in layer
    init_v = np.random.normal(loc=0, scale=0.5, size=(2, n_hidden))
    init_v = (np.vstack((init_v, np.ones((1,n_hidden))))).T
    
    # second weights has dim n_hidden + bias (1) times 1 since only one output node
    init_w = np.random.normal(loc=0, scale=0.5, size=(n_hidden, 1))
    init_w = (np.vstack((init_w, np.ones((1,1))))).T

    return init_v, init_w



    

init_v, init_w = create_init_weights(2)
generalised_d_batch(init_v, init_w)

plt.scatter(classA[0], classA[1])
plt.scatter(classB[0], classB[1])
plt.show()