import numpy as np

# constant N
N = 100


def generate():
    # Class A
    mean_A = [1.0, 0.3]
    sigma_A = 0.2
    x_A_left = np.random.permutation(np.random.normal(size=(int(0.5 * N))) * sigma_A - mean_A[0])
    x_A_right = np.random.permutation(np.random.normal(size=(int(0.5 * N))) * sigma_A + mean_A[0])
    y_A = np.random.permutation(np.random.normal(size=N) * sigma_A + mean_A[1])

    # Class B
    mean_B = [0.0, -0.1]
    sigma_B = 0.3
    x_B = np.random.permutation(np.random.normal(size=N) * sigma_B + mean_B[0])
    y_B = np.random.permutation(np.random.normal(size=N) * sigma_B + mean_B[1])

    # init weight
    init_weight_x_r = np.random.normal(0, 0.2)
    init_weight_y_r = np.random.normal(0, 0.2)
    theta = 1
    init_w = np.array([init_weight_x_r, init_weight_y_r, -theta])

    return x_A_left, x_A_right, y_A, x_B, y_B, init_w


def sub_sampling(x_A_left, x_A_right, y_A, x_B, y_B, frac_left_A, frac_right_A, frac_B, special=False):
    N_A_left = int(frac_left_A * N / 2)
    N_A_right = int(frac_right_A * N / 2)

    if (not special):
        frac_A = N_A_left + N_A_right

        x_A = np.concatenate((x_A_left, x_A_right))
        p = np.random.permutation(len(x_A))
        x_A = x_A[p]
        y_A = y_A[p]

        x_A = x_A[:frac_A]
        y_A = y_A[:frac_A]

    else:
        y_A_left = y_A[:N_A_left]
        middle = int(len(y_A) / 2)
        y_A_right = y_A[middle:middle + N_A_right]  ##TODO: check divisions to make even

        # choose number of values specified by frac (they are already in random order so no shuffle here, do big shuffle together with B)
        x_A_left = x_A_left[:len(y_A_left)]
        x_A_right = x_A_right[:len(y_A_right)]
        x_A = np.concatenate((x_A_left, x_A_right))
        y_A = np.concatenate((y_A_left, y_A_right))
        N_A_left = len(x_A_left)
        N_A_right = len(x_A_right)

    # get slice of B
    frac_B = int(frac_B)
    x_B = x_B[:frac_B * N]
    y_B = y_B[:frac_B * N]

    targets = np.concatenate((np.ones(N_A_left + N_A_right), (np.ones(frac_B * N) * -1)))

    # shuffle the entire partial dataset
    p2 = np.random.permutation(len(targets))
    x = np.concatenate((x_A, x_B))[p2]
    y = np.concatenate((y_A, y_B))[p2]
    targets = targets[p2]

    classA = [x_A, y_A]
    classB = [x_B, y_B]
    input_arr = np.array([x, y, np.ones(len(x))])

    return input_arr, targets, classA, classB


def get_data():
    x_A_left, x_A_right, y_A, x_B, y_B, init_w = generate()
    input_arr, target, classA, classB = sub_sampling(x_A_left, x_A_right, y_A, x_B, y_B, 1.0, 1.0, 1.0, False)
    return input_arr, target, classA, classB, init_w
