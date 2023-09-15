import generate_data as gen
import matplotlib.pyplot as plt
import numpy as np
import plot_drawer as plotter

ETA = 0.1
EPOCHS = 30
N_HIDDEN = [5]


def create_init_weights(n_hidden):
    #  init weight v from input to hidden layer is 3 (Xx, Xy, bias) times number of nodes in layer
    init_v = np.random.normal(loc=0, scale=0.5, size=(2, n_hidden))
    init_v = (np.vstack((init_v, np.ones((1, n_hidden))))).T

    # second weights has dim n_hidden + bias (1) times 1 since only one output node
    init_w = np.random.normal(loc=0, scale=0.5, size=(n_hidden, 1))
    init_w = (np.vstack((init_w, np.ones((1, 1))))).T

    return init_v, init_w

def generalised_d_sequential(n_hidden, init_v, init_w, input_arr, targets):
    number_of_inputs = len(input_arr[0])
    # n_hidden is the number of nodes in the hidden layer
    v = init_v
    w = init_w
    theta = 0
    psi = 0
    mse = []
    miss_ratio = []

    for epoch in range(EPOCHS):
        out_list = []  # Used for accumulating all out values for each epoch 
        sum = 0         # Used for calculating the mse error
        for i in range(number_of_inputs):
            input = np.array([[input_arr[0][i]], [input_arr[1][i]], [input_arr[2][i]]])
            target = np.array([[targets[i]]])

            # Forward pass
            h_in = np.dot(v, input)
            h_out = 2 / (1 + np.exp(-h_in)) - 1
            h_out = np.vstack((h_out, np.ones((1, len(h_out[0])))))

            o_in = np.dot(w, h_out)
            out = (2 / (1 + np.exp(-o_in))) - 1
    
            # Backward pass
            delta_o = (out - target) * ((1 + out) * (1 - out)) * 0.5
            delta_h = (w.T * delta_o) * ((1 + h_out) * (1 - h_out)) * 0.5
            delta_h = delta_h[:n_hidden, :]

            # Weight update with momentum
            alfa = 0.9

            # Theta is for v and psi is for w
            theta = alfa * theta - (1 - alfa) * np.dot(delta_h, input.T)
            psi = alfa * psi - (1 - alfa) * np.dot(delta_o, h_out.T)
            delta_v = ETA * theta
            delta_w = ETA * psi

            v += delta_v
            w += delta_w

            # Accumulate mse sum
            sum += (target - out)[0][0]**2

            # Collect 'out' values 
            out_list.append(out)
        
        # Add to MSE list
        mse.append(sum/number_of_inputs)
       
        # Count number of misses per epoch
        out_list = np.array(out_list).reshape(1, 200)
        miss_ratio.append(count_misses_per_epoch(out_list, targets))
    return v, w, mse, out, miss_ratio

def forward_pass(v, w, input_arr):
    h_in = np.dot(v, input_arr)
    h_out = 2 / (1 + np.exp(-h_in)) - 1
    h_out = np.vstack((h_out, np.ones((1, len(h_out[0])))))

    o_in = np.dot(w, h_out)
    out = (2 / (1 + np.exp(-o_in))) - 1
    return h_out, out


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
        h_out, out = forward_pass(v, w, input_arr)

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

    return v, w, mse, out, miss_ratio


def count_misses_per_epoch(out, targets):
    """
    Counts how many misses and hits per epoch.
    :param out: predicted output
    :param targets: targeted output
    :return: rate of misclassified predictions
    """
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



def task1_1(input_arr, targets):
    mse_list = []
    v_list = []
    w_list = []
    out_list = []
    miss_ratio_list = []
    for n_hidden in N_HIDDEN:
        init_v, init_w = create_init_weights(n_hidden)
        v, w, mse, out, miss_ratio = generalised_d_batch(n_hidden, init_v, init_w, input_arr, targets)
        v_list.append(v)
        w_list.append(w)
        mse_list.append(mse)
        out_list.append(out)
        miss_ratio_list.append(miss_ratio)

    count_misses_per_hidden_n(out_list, targets)

    # make MSE plot
    plt.figure(2)
    plotter.draw_mse_or_miss_rate(mse_list, N_HIDDEN, EPOCHS, "Mean Square Error")
    plt.show()

    # make miss rate plot
    plt.figure(3)
    plotter.draw_mse_or_miss_rate(miss_ratio_list, N_HIDDEN, EPOCHS, "Misclassification Rate")
    plt.show()
    
def task1_2_seq(input_arr, targets):
    # Task 2 - sequential delta question
    mse_list = []
    v_list = []
    w_list = []
    out_list = []
    miss_ratio_list = []

    for n_hidden in N_HIDDEN:
        init_v, init_w = create_init_weights(n_hidden)
        v, w, mse, out, miss_ratio = generalised_d_sequential(n_hidden, init_v, init_w, input_arr, targets)
        mse_list.append(mse)
        miss_ratio_list.append(miss_ratio)

    # make MSE plot
    plt.figure(1)
    plotter.draw_mse_or_miss_rate(mse_list, N_HIDDEN, EPOCHS, "Mean Square Error")
    plt.show()

    # make miss rate plot
    plt.figure(4)
    plotter.draw_mse_or_miss_rate(miss_ratio_list, N_HIDDEN, EPOCHS, "Misclassification Rate")
    plt.show()

def task1_2(input_arr, targets, a_frac, b_frac):
    train_set_A = []
    val_set_A = []
    train_set_B = []
    val_set_B = []
    counter_a = 0
    counter_b = 0

    for i, target in enumerate(targets):
        if target == -1:
            if counter_b / len(targets) < b_frac:
                train_set_B.append(input_arr[i])
                counter_b += 1
            else:
                val_set_B.append(input_arr[i])
        elif target == 1:
            if counter_a / len(targets) < a_frac:
                train_set_A.append(input_arr[i])
                counter_a += 1
            else:
                val_set_A.append(input_arr[i])
        else:
            print(target, i)

    return train_set_A, train_set_B, val_set_A, val_set_B


def main():
    input_arr, targets, classA, classB, _ = gen.get_data()

    # make scatter plot and boundaries

    plt.figure(1)
    plotter.draw_scatter(classA, classB)
    plt.show()

    # task1_1(input_arr, targets)

    task1_2(input_arr, targets)
    
    task1_2_seq(input_arr, targets)


main()
