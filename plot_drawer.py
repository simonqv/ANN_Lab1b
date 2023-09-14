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
            x_axis =np.linspace(min(input_array[0]), max(input_array[0]), 100)
            draw_plot(line, x_axis, label=f"Line: {i+1}")


def draw_mse(mse_list, n_hidden, epochs):
    for i, n in enumerate(n_hidden):
        x_axis = np.linspace(0, epochs, epochs)
        plt.plot(x_axis, mse_list[i], label=f"MSE for {n} hidden nodes ")

    plt.ylabel("Mean Square Error")
    plt.xlabel("Epoch")
    plt.title("Mean Square Error for different number of hidden nodes")
    plt.legend()