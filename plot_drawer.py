import numpy as np
from matplotlib import pyplot as plt


def draw_scatter(classA, classB):
    # scatter plot
    plt.scatter(classA[0], classA[1])
    plt.scatter(classB[0], classB[1])


def draw_plot(weights, x_decision, label):
    # Plot the decision boundary line using the final weights
    if len(weights) == 3:
        y_decision = (-weights[0] * x_decision - weights[2]) / weights[1]
    else:
        y_decision = (-weights[0] * x_decision) / weights[1]
    plt.plot(x_decision, y_decision, label=label)


def draw_boundaries(v_list, input_array):
    """
        x          y            bias
    [[ 0.50093174  0.4048389   1.0095681 ]
     [ 0.85888211 -0.52920806  0.94146333]
     [-0.79641668 -0.12584958  1.00834653]
     [-0.05110146 -0.2469575   0.97682161]
     [-0.58920201  0.00404003  1.00317191]]
    """
    for v in v_list:
        for i, line in enumerate(v):
            x_axis = np.linspace(min(input_array[0]), max(input_array[0]), 100)
            draw_plot(line, x_axis, label=f"Line: {i + 1}")


def draw_mse_and_miss_rate_bar(plot_list_batch, plot_list_seq, n_hidden, plot_name, fig_num):
    a_bars = plot_list_batch[0]
    b_bars = plot_list_batch[1]
    c_bars = plot_list_batch[2]

    a_bars_seq = plot_list_seq[0]
    b_bars_seq = plot_list_seq[1]
    c_bars_seq = plot_list_seq[2]
    

    """
    mse_a_bars = mses[0]
    miss_a_bars = misses[0]

    mse_b_bars = mses[1]
    miss_b_bars = misses[1]

    mse_c_bars = mses[2]
    miss_c_bars = misses[2]
    """

    # Position of bars on x-axis
    ind = np.arange(len(n_hidden))

    width = 1/7
    #
    # Plotting
    plt.figure(fig_num)
    print(a_bars)
    plt.bar(ind - 3*width, a_bars, width, label='a) 25% of each')
    plt.bar(ind - 2*width, b_bars,width, label='b) 50% of class A')
    plt.bar(ind - width, c_bars, width, label='c) Special')
    print("SE", a_bars_seq)
    plt.bar(ind, a_bars_seq, width, label='a) 25% of each')
    plt.bar(ind + width, b_bars_seq, width, label='b) 50% of class A')
    plt.bar(ind + 2*width, c_bars_seq, width, label='c) Special')
    
    plt.xlabel('Number of hidden layers')
    plt.ylabel(plot_name)
    plt.title(f'{plot_name} for different number of hidden layers')
    plt.xticks(ind + width / 6, n_hidden)

    # Finding the best position for legends and putting it
    plt.legend(loc='best')
    plt.show()




def draw_mse_or_miss_rate(mse_list, n_hidden, epochs, plot_title):
    x_axis = np.linspace(0, epochs, epochs)
    for i, n in enumerate(n_hidden):
        plt.plot(x_axis, mse_list[i], label=f"{plot_title} for {n} hidden nodes ")

    plt.ylabel(plot_title)
    plt.xlabel("Epoch")
    plt.title(f"{plot_title} for different number of hidden nodes")
    plt.legend()
