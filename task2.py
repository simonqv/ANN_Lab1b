import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor as mlpR

VAL_START = 301
VAL_END = 1275 # Fixed, should not be changed (1300 - 25 for back in time steps)
TEST_START = 1300
EPOCHS = 200  # default in sklearn is 200
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

    return train_set, train_targets, test_set, test_targets, x

    
def main():
    
    print("\nFirst part of assignment 2\n")

    train_set, train_targets, test_set, test_targets, x = generate_data() 

    plt.figure(1)
    make_train_val_start_plot(x)
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.title("Mackey-Glass time series (no noice)")

    hidden_1 = [3, 4, 5]
    hidden_2 = [2, 4, 6]
    scores = {}
    predictions = {}
    c = 0
    for i1, h1 in enumerate(hidden_1):
        for i2, h2 in enumerate(hidden_2):
            model = mlpR([h1, h2], 'logistic', 'sgd', learning_rate='constant', 
                            batch_size=20, learning_rate_init=0.1, max_iter=EPOCHS, 
                            shuffle=False, momentum=0.9, early_stopping=True, 
                            validation_fraction=0.1, alpha=0.0001)

    
            regressor = train_model(train_set, train_targets, model)
            val_scores = regressor.validation_scores_
            y_out = regressor.predict(test_set)
            scores[(h1, h2)] = (regressor.best_validation_score_)
            predictions[(h1, h2)] = y_out
            plt.figure(13)
            plt.plot(val_scores, label=f"({h1} x {h2}) hidden layers")
            plt.legend()
            plt.ylabel("Score, R²")
            plt.xlabel("Epoch")
            plt.title("Validation Score per Model")

            print(f"R² scores ({h1} x {h2}) hidden layers", regressor.best_validation_score_)
            # plt.figure(2 + c)
            # make_plots(train_targets, test_targets, y_out)
            c +=1
    
    best_score = max(scores, key=lambda k: scores[k])
    worst_score = min(scores, key=lambda k: scores[k])
    print(f"Best: {best_score}. Worst: {worst_score}")

    plt.figure(14)
    make_plots(predictions[best_score], f"Predictions \"best\" score: {best_score}", 'b', test_targets)
    make_plots(predictions[worst_score], f"Predictions \"worst\" score: {worst_score}", 'r')
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.title("Time Series Predictions for \"Best\" and \"Worst\" Models")
    
    plt.legend()
    plt.show()

    # 3.2.4 Add Noise
    print("\nSecond part of assignment 2\n")

    h1_best = best_score[0]
    h2_best = [3, 6, 9]

    new_train_set1, new_train_set2 = add_noise(train_set, train_set.shape)
    new_training = [new_train_set1, new_train_set2]
    std_noise = [0.05, 0.15]
    i = 0
    for training_with_noise in new_training:
        for h in h2_best:
            model = mlpR([h1_best, h], 'logistic', 'sgd', learning_rate='constant', 
                                batch_size=20, learning_rate_init=0.1, max_iter=EPOCHS, 
                                shuffle=False, momentum=0.9, early_stopping=True, 
                                validation_fraction=0.1, alpha=0.0001)

            regressor = train_model(training_with_noise, train_targets, model)
            val_scores = regressor.validation_scores_
            y_out = regressor.predict(test_set)
            scores[(h1_best, h)] = (regressor.best_validation_score_)
            predictions[(h1_best, h)] = y_out
            
            plt.figure(15)
            plt.plot(val_scores, label=f"({h1_best} x {h}) hidden layers")
            

            print(f"R² scores ({h1_best} x {h}) hidden layers", regressor.best_validation_score_)
            # plt.figure(2 + c)
            # make_plots(train_targets, test_targets, y_out)
            c += 1
        
        plt.figure(15)
        plt.legend()
        plt.ylabel("Score, R²")
        plt.xlabel("Epoch")
        plt.title(f"Validation Scores, Noise with Standard Deviation {std_noise[i]}")

        plt.figure(16)
        make_plots(predictions[(h1_best, h2_best[0])], f"Predictions {h1_best, h2_best[0]}",test_targets=test_targets)
        make_plots(predictions[(h1_best, h2_best[1])], f"Predictions {h1_best, h2_best[1]}")
        make_plots(predictions[(h1_best, h2_best[2])], f"Predictions {h1_best, h2_best[2]}")
        plt.xlabel("t")
        plt.ylabel("x(t)")
        plt.title(f"Time Series Predictions for Differently Sized Hidden Layers,\nNoise with Standard Deviation {std_noise[i]}")

        i += 1
        plt.show()


#############################################################
#       HELPERS
#############################################################

def train_model(train_set, targets, model):
    regr_model = model.fit(train_set, targets)
    return regr_model


def extract_5_prev_and_target(t, time_series):
    new_slice = time_series[t-20 : t+5 : 5]
    target = time_series[t+5]
    return new_slice, target


def make_train_val_start_plot(x):
    plt.plot(np.linspace(VAL_START, N, len(x[301+5:])) , x[301+5:])


def add_noise(train_set, shape):
    mean = 0
    std_deviation = [0.05, 0.15]
    na1 = np.random.normal(mean, std_deviation[0], shape)
    na2 = np.random.normal(mean, std_deviation[1], shape)
    
    return train_set + na1,  train_set + na2


def make_plots(y_out, lb, color=None, test_targets=None, train_targets=None):
    # plt.plot(np.arange(N)[VAL_START:VAL_END], train_targets, label="Training")
    if test_targets is not None:
        plt.plot(np.arange(N)[TEST_START:TEST_START+200 - 5], test_targets[:-5], label="Target Output", c='grey', ls="--")
    plt.plot(np.arange(N)[TEST_START:TEST_START+200 - 5], y_out[:-5], label=lb, c=color)
    plt.legend()


main()