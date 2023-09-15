import generate_data as gen
import matplotlib.pyplot as plt
import numpy as np
import plot_drawer as plotter

ETA = 0.1
EPOCHS = 30
N_HIDDEN = [2, 4, 8, 10, 20, 30]


def create_init_weights(n_hidden):
    #  init weight v from input to hidden layer is 3 (Xx, Xy, bias) times number of nodes in layer
    init_v = np.random.normal(loc=0, scale=0.5, size=(2, n_hidden))
    init_v = (np.vstack((init_v, np.ones((1, n_hidden))))).T

    # second weights has dim n_hidden + bias (1) times 1 since only one output node
    init_w = np.random.normal(loc=0, scale=0.5, size=(n_hidden, 1))
    init_w = (np.vstack((init_w, np.ones((1, 1))))).T

    return init_v, init_w


def generalised_d_batch(n_hidden, init_v, init_w, input_arr, targets):
    # n_hidden is the number of nodes in the hidden layer
    v = init_v
    w = init_w
    theta = 0
    psi = 0
    mse = []
    miss_ratio = []

    for i in range(EPOCHS):
        # forward pass
        h_in = np.dot(v, input_arr)
        h_out = 2 / (1 + np.exp(-h_in)) - 1
        h_out = np.vstack((h_out, np.ones((1, len(h_out[0])))))

        o_in = np.dot(w, h_out)
        out = (2 / (1 + np.exp(-o_in))) - 1

        # backward pass
        delta_o = (out - targets) * ((1 + out) * (1 - out)) * 0.5
        delta_h = (w.T * delta_o) * ((1 + h_out) * (1 - h_out)) * 0.5
        delta_h = delta_h[:n_hidden, :]

        # weight update with momentum
        alfa = 0.9

        # theta is for v and psi is for w.
        theta = alfa * theta - (1 - alfa) * np.dot(delta_h, input_arr.T)
        psi = alfa * psi - (1 - alfa) * np.dot(delta_o, h_out.T)
        delta_v = ETA * theta
        delta_w = ETA * psi

        v += delta_v
        w += delta_w

        # Add to MSE list
        mse.append(np.sum((targets - out) ** 2) / len(out[0]))

        # Count number of misses per epoch
        miss_ratio.append(count_misses_per_epoch(out, targets))

    print(v.shape, w.shape)
    return v, w, mse, out, miss_ratio


def count_misses_per_epoch(out, targets):
    correct = 0
    miss = 0
    for i, target in enumerate(targets):
        if out[0, i] >= 0 and targets[i] == 1:
            correct += 1
        elif out[0, i] < 0 and targets[i] == -1:
            correct += 1
        else:
            miss += 1
    return miss/len(out[0])


def count_misses_per_hidden_n(out_list, targets):
    for ind, out in enumerate(out_list):
        correct = 0
        miss = 0
        for i in range(len(targets)):
            if out[0, i] >= 0 and targets[i] == 1:
                correct += 1
            elif out[0, i] < 0 and targets[i] == -1:
                correct += 1
            else:
                miss += 1
        print(f"N = {N_HIDDEN[ind]} correct = ", correct)
        print(f"N = {N_HIDDEN[ind]} miss    = ", miss)


def main():
    input_arr, targets, classA, classB, init_w = gen.get_data()

    mse_list = []
    v_list = []
    w_list = []
    out_list = []
    miss_ratio_list = []

    # Task 1
    for n_hidden in N_HIDDEN:
        init_v, init_w = create_init_weights(n_hidden)
        v, w, mse, out, miss_ratio = generalised_d_batch(n_hidden, init_v, init_w, input_arr, targets)
        v_list.append(v)
        w_list.append(w)
        mse_list.append(mse)
        out_list.append(out)
        miss_ratio_list.append(miss_ratio)

    count_misses_per_hidden_n(out_list, targets)

    # make scatter plot and boundaries
    """
    plt.figure(1)
    plotter.draw_scatter(classA, classB)
    plotter.draw_boundaries(v_list, input_arr)
    plt.show()
    """

    # make MSE plot
    plt.figure(2)
    plotter.draw_mse_or_miss_rate(mse_list, N_HIDDEN, EPOCHS, "Mean Square Error")
    plt.show()

    # make miss rate plot
    plt.figure(3)
    plotter.draw_mse_or_miss_rate(miss_ratio_list, N_HIDDEN, EPOCHS, "Misclassification Rate")
    plt.show()


main()
