import generate_data as gen
import matplotlib.pyplot as plt
import numpy as np
import plot_drawer as plotter

ETA = 0.1
EPOCHS = 30
N_HIDDEN = [5, 10, 20, 30, 45]


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
        sum = 0  # Used for calculating the mse error
        for i in range(number_of_inputs):
            input = np.array([[input_arr[0][i]], [input_arr[1][i]], [input_arr[2][i]]])
            target = np.array([[targets[i]]])

            # Forward pass
            h_out, out = forward_pass(v, w, input)

            # Backward pass
            v, w, theta, psi = backward_pass_and_update(n_hidden, input, target,w,v,out,h_out,theta,psi)

            # Accumulate mse sum
            sum += (target - out)[0][0] ** 2

            # Collect 'out' values 
            out_list.append(out)

        # Add to MSE list
        mse.append(sum / number_of_inputs)

        # Count number of misses per epoch
        out_list = np.array(out_list).reshape(1, len(targets))
        miss_ratio.append(count_misses_per_epoch(out_list, targets))
    return v, w, mse, out, miss_ratio


def forward_pass(v, w, input_arr):
    h_in = np.dot(v, input_arr)
    h_out = 2 / (1 + np.exp(-h_in)) - 1
    h_out = np.vstack((h_out, np.ones((1, len(h_out[0])))))

    o_in = np.dot(w, h_out)
    out = (2 / (1 + np.exp(-o_in))) - 1
    return h_out, out


def backward_pass_and_update(n_hidden, input_arr, targets, w, v, out, h_out, theta, psi):
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
    return v, w, theta, psi


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
        v, w, theta, psi = backward_pass_and_update(n_hidden, input_arr, targets, w, v, out, h_out, theta, psi)

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
    return miss / len(out[0])


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


def task1_2_seq(train_input, train_targets, val_inputs, val_targets, validation=False):

    # Task 2 - sequential delta question
    mse_list_train = []
    mse_list_val = []
    v_list = []
    w_list = []
    out_list = []
    miss_ratio_list = []
    miss_ratio_list_val = []
    sum = 0

    for n_hidden in N_HIDDEN:
        init_v, init_w = create_init_weights(n_hidden)
        v, w, mse, out, miss_ratio = generalised_d_sequential(n_hidden, init_v, init_w, train_input, train_targets)
        mse_list_train.append(mse)
        miss_ratio_list.append(miss_ratio)

        if(validation):
            for i in range(len(train_input)):
                _, val_out = forward_pass(v, w, val_inputs)
                sum += (val_targets - val_out) ** 2
            sum = sum / len(val_out)
            mse_list_val.append(sum)
            misses_val = count_misses_per_epoch(val_out, val_targets)
            miss_ratio_list_val.append(misses_val)

    

    # make MSE plot
    plt.figure(1)
    plotter.draw_mse_or_miss_rate(mse_list_train, N_HIDDEN, EPOCHS, "Mean Square Error")
    plt.show()

    # make miss rate plot
    plt.figure(4)
    plotter.draw_mse_or_miss_rate(miss_ratio_list, N_HIDDEN, EPOCHS, "Misclassification Rate")
    plt.show()

    return mse_list_val, miss_ratio_list_val


def train_val_split(input_arr, targets, a_frac, b_frac):
    counter_a = 0
    counter_b = 0

    size_train = int((len(targets) / 2) * (a_frac + b_frac))
    train_set = np.empty((3, 1))
    val_set = np.empty((3, 1))
    train_targets = np.empty((1,))
    val_targets = np.empty((1,))

    for i, target in enumerate(targets):

        if target == -1:
            if counter_b / (len(targets) / 2) < b_frac:
                train_set = np.hstack((train_set, np.reshape(input_arr[:, i], (3, 1))))
                train_targets = np.concatenate((train_targets, [target]))
                counter_b += 1
            else:
                val_set = np.hstack((val_set, np.reshape(input_arr[:, i], (3, 1))))
                val_targets = np.concatenate((val_targets, [target]))

        elif target == 1:
            if counter_a / (len(targets) / 2) < a_frac:
                train_set = np.hstack((train_set, np.reshape(input_arr[:, i], (3, 1))))
                train_targets = np.concatenate((train_targets, [target]))
                counter_a += 1
            else:
                val_set = np.hstack((val_set, np.reshape(input_arr[:, i], (3, 1))))
                val_targets = np.concatenate((val_targets, [target]))

    return train_set[:, 1:], val_set[:, 1:], train_targets[1:], val_targets[1:]


def special_train_val_split(input_arr, targets, l_frac, r_frac):
    counter_a_left = 0
    counter_a_right = 0

    size_train = int((len(targets) / 2) * (l_frac + r_frac + 1))
    train_set = np.empty((3, 1))
    val_set = np.empty((3, 1))
    train_targets = np.empty((1,))
    val_targets = np.empty((1,))

    for i, target in enumerate(targets):
        if target == -1:
            val_set = np.hstack((train_set, np.reshape(input_arr[:, i], (3, 1))))
            val_targets = np.concatenate((train_targets, [target]))

        else:
            if input_arr[0, i] < 0 and counter_a_left / (len(targets) / 4) < l_frac:
                train_set = np.hstack((train_set, np.reshape(input_arr[:, i], (3, 1))))
                train_targets = np.concatenate((train_targets, [target]))
                counter_a_left += 1

            elif input_arr[0, i] >= 0 and counter_a_right / (len(targets) / 4) < r_frac:
                train_set = np.hstack((train_set, np.reshape(input_arr[:, i], (3, 1))))
                train_targets = np.concatenate((train_targets, [target]))
                counter_a_right += 1

            else:
                val_set = np.hstack((val_set, np.reshape(input_arr[:, i], (3, 1))))
                val_targets = np.concatenate((val_targets, [target]))
    print(counter_a_left, counter_a_right)

    return train_set[:, 1:], val_set[:, 1:], train_targets[1:], val_targets[1:]


def task1_2(input_arr, targets, a_frac, b_frac, special=False):
    if not special:
        train_set, val_set, train_targets, val_targets = train_val_split(input_arr, targets, a_frac, b_frac)
        print(train_set.shape, val_set.shape, train_targets.shape, val_targets.shape)
    else:
        train_set, val_set, train_targets, val_targets = special_train_val_split(input_arr, targets, a_frac, b_frac)

    mse_list = []
    mse_list_val = []
    v_list = []
    w_list = []
    out_list_train = []
    out_list_val = []
    miss_ratio_list_train = []
    miss_ratio_list_val = []

    for n_hidden in N_HIDDEN:
        init_v, init_w = create_init_weights(n_hidden)

        # Training section
        v, w, mse, out, miss_ratio = generalised_d_batch(n_hidden, init_v, init_w, input_arr, targets)

        # Stuff for plotting
        v_list.append(v)
        w_list.append(w)
        mse_list.append(mse)
        out_list_train.append(out)
        miss_ratio_list_train.append(miss_ratio)

        # Validation section
        _, out_v = forward_pass(v, w, val_set)

        # Count misses
        miss_ratio_list_val.append(count_misses_per_epoch(out_v, val_targets))

        # Count mse
        mse_list_val.append(np.sum((val_targets - out_v) ** 2) / len(out_v[0]))



    return mse_list_val, miss_ratio_list_val, train_set, train_targets, val_set, val_targets


def task1_2_graph(mse_list_val, miss_ratio_list_val ):
    for i in range(len(mse_list_val)):
        # MSE bar plot
        plt.figure(2)
        plotter.draw_mse_or_miss_rate_bar(mse_list_val, N_HIDDEN, "MSE")
        # miss rate bar plot
        plt.figure(3)
        plotter.draw_mse_or_miss_rate_bar(miss_ratio_list_val, N_HIDDEN, "Miss Rate")

    # make MSE bar plot
    plt.figure(2)
    plt.show()

    # make miss rate bar plot
    plt.figure(3)
    plt.show()


def main():
    input_arr, targets, classA, classB, _ = gen.get_data()

    # make scatter plot and boundaries
    '''
    plt.figure(1)
    plotter.draw_scatter(classA, classB)
    plt.show()

    '''
    # task1_1(input_arr, targets)

    # Task 1.2 a) 25% of each
    mse_list_a, miss_ratio_a, train_set_a, train_targets_a, val_set_a, val_targets_a = task1_2(input_arr, targets, 0.25, 0.25)
    mse_seq_a, miss_seq_a = task1_2_seq(train_set_a, train_targets_a, val_set_a, val_targets_a, True)

    # Task 1.2 b) 50% of a
    mse_list_b, miss_ratio_b, train_set_b, train_targets_b, val_set_b, val_targets_b = task1_2(input_arr, targets, 0.5, 0.0)
    mse_seq_b, miss_seq_b = task1_2_seq(train_set_b, train_targets_b, val_set_b, val_targets_b, True)

    # Task 1.2 c) special case, here a_frac is left, b_frac is right
    mse_list_c, miss_ratio_c, train_set_c, train_targets_c, val_set_c, val_targets_c = task1_2(input_arr, targets, 0.2, 0.8, True)
    mse_seq_c, miss_seq_c = task1_2_seq(train_set_c, train_targets_c, val_set_c, val_targets_c, True)

    # Plot for 1.2a-c
    # MSE
    plotter.draw_mse_and_miss_rate_bar([mse_list_a, mse_list_b, mse_list_c], N_HIDDEN, "MSE batch", 2)
    # Miss Ratio
    plotter.draw_mse_and_miss_rate_bar([miss_ratio_a, miss_ratio_b, miss_ratio_c], N_HIDDEN, "Miss Ratios batch", 3)
    # task1_2_seq(input_arr, targets)


main()
