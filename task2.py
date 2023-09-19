import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor as mlpR

VAL_START = 301
VAL_END = 1275 # Fixed, should not be changed (1300 - 25 for back in time steps)
TEST_START = 1300
EPOCHS = 200 # default in sklearn is 200
N = 1500

def generate_data():
    x = np.array([1.5])
    beta = 0.2
    gamma = 0.1
    n = 10
    tau = 25

    for i in range(N-1):
        if i-tau >= 0: 
            x = np.concatenate((x, [(x[i] + (beta * x[i-tau]/(1 + np.power(x[i-tau],n))) - gamma*x[i])])) 
        else:
            x = np.concatenate((x, [(x[i] - gamma*x[i])]))

    
    train_set = np.zeros((VAL_END - VAL_START, 5))
    train_targets = np.zeros(VAL_END - VAL_START)
    test_set = np.zeros((N - TEST_START, 5))
    test_targets = np.zeros(N - TEST_START)

    # create train matrix with each row having 5 time steps and a corresponding target t+5
    for t in range(VAL_END - VAL_START):
        train_t, target_t = extract_5_prev_and_target(VAL_START + t, x)
        train_set[t] = train_t
        train_targets[t] = target_t
        
    # same for test matrix and targets for error checking
    for t in range(N - TEST_START - 5):
        test_t, test_target = extract_5_prev_and_target(TEST_START + t, x)
        test_set[t] = test_t
        test_targets[t] = test_target

    return train_set, train_targets, test_set, test_targets


def train_model(train_set, targets, model):
    regr_model = model.fit(train_set, targets)
    return regr_model

def extract_5_prev_and_target(t, time_series):
    new_slice = time_series[t-20 : t+5 : 5]
    target = time_series[t+5]
    return new_slice, target



def main():
    model = mlpR([3, 2], 'logistic', 'sgd', learning_rate='constant', 
                    batch_size=200, learning_rate_init=0.1, max_iter=EPOCHS, 
                    shuffle=False, momentum=0.9, early_stopping=True, validation_fraction=0.1)

    train_set, train_targets, test_set, test_targets = generate_data()   

    plt.plot(np.arange(N)[VAL_START:VAL_END], train_targets)
    plt.plot(np.arange(N)[TEST_START:TEST_START+200], test_targets)
    
    regressor = train_model(train_set, train_targets, model)
    y_out = regressor.predict(test_set)
    print(y_out)
    plt.plot(np.arange(N)[TEST_START:TEST_START+200], y_out)

    plt.show()

    #for i in range(len(test_set)):

    #   regerssor.predict(test_set)
    
main()