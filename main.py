import generate_data as gen
import matplotlib.pyplot as plt
import numpy as np

ETA = 0.001
EPOCHS = 10

def generalised_d_batch(n_hidden, init_v, init_w, input_arr, targets):
    # n_hidden is the number of nodes in the hidden layer
    v = init_v
    w = init_w
    theta = 1
    psi = 1

    for i in range(EPOCHS):
        # forward pass
        h_in = np.dot(v, input_arr)
        h_out = 2 / (1 + np.exp(-h_in)) - 1
        h_out = np.vstack((h_out, np.ones((1, len(h_out[0])))))

        o_in = np.dot(w, h_out)
        out = 2 / (1 + np.exp(-o_in)) - 1

        # backward pass
        delta_o = (out - targets) * ((1 + out) * (1 - out)) * 0.5
        delta_h = (w.T * delta_o) * ((1 + h_out) * (1 - h_out)) * 0.5
        delta_h = delta_h[:n_hidden, :]

        # weight update - with momentum
        alfa = 0.9        
        theta = alfa*theta - (1-alfa)*np.dot(delta_h, input_arr.T)
        psi = alfa*psi - (1-alfa)*np.dot(delta_o, h_out.T)
        delta_v = -ETA * theta
        delta_w = -ETA * psi
        
        v += delta_v
        w += delta_w

    print(v.shape, w.shape)
    print(w)
    return w

def create_init_weights(n_hidden):
    #  init weight v from input to hidden layer is 3 (Xx, Xy, bias) times number of nodes in layer
    init_v = np.random.normal(loc=0, scale=0.5, size=(2, n_hidden))
    init_v = (np.vstack((init_v, np.ones((1,n_hidden))))).T
    
    # second weights has dim n_hidden + bias (1) times 1 since only one output node
    init_w = np.random.normal(loc=0, scale=0.5, size=(n_hidden, 1))
    init_w = (np.vstack((init_w, np.ones((1,1))))).T

    return init_v, init_w




def main():

    input_arr, targets, classA, classB, init_w = gen.get_data()
    
    # scatter plot
    plt.scatter(classA[0], classA[1])
    plt.scatter(classB[0], classB[1])
    plt.show()
    
    n_hidden = 5
    init_v, init_w = create_init_weights(n_hidden)
    # For task 1 make a loop here for different n_hidden
    w = generalised_d_batch(n_hidden, init_v, init_w, input_arr, targets)
    print(w)



main()