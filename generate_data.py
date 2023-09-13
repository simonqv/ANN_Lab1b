import numpy as np
import matplotlib.pyplot as plt

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
  init_w = [init_weight_x_r, init_weight_y_r, -theta]

  return x_A_left, x_A_right, y_A, x_B, y_B, init_w
  

def sub_sampling(x_A_left, x_A_right, y_A, x_B, y_B, frac_left_A, frac_right_A, frac_B, special=False):

  N_A_left = int(frac_left_A * N / 2)
  N_A_right = int(frac_right_A * N / 2)

  y_A_left = y_A[:N_A_left]
  y_A_right = y_A[N_A_left:]

  if(not special):
    frac_A = frac_left_A + frac_right_A

    temp = zip(np.concatenate(x_A_left, x_A_right), y_A)
    np.random.shuffle(temp)
    x_A, y_A = zip(*temp)

    x_A = x_A[:N*frac_A]
    y_A = y_A[:N*frac_A]

  else:
    # TODO: frac left * xleft och frac right * xright

  
  



  # extract subset 
  slice_x_A = xA_left[:s_A_left_val] + xA_right[:s_A_right_val]
  slice_y_A = yA_left[:s_A_left_val] + yA_right[:s_A_right_val]
  classA = [slice_x_A, slice_y_A]
  print("left ", xA_left[:s_A_left_val], "\n")
  print("right ", xA_right[:s_A_right_val], "\n")
  print(slice_x_A)


  slice_x_B = xB[:s_B] 
  slice_y_B = yB[:s_B] 
  classB = [slice_x_B, slice_y_B]
    

    targets_perceptron = np.ones(s_A_left_val + s_A_right_val).tolist() + np.zeros(s_B).tolist()
    targets_delta = np.ones(s_A_left_val + s_A_right_val).tolist() + (np.ones(s_B) * -1).tolist()

    temp = list(zip(slice_x_A + slice_x_B, slice_y_A + slice_y_B, targets_perceptron, targets_delta))
    random.shuffle(temp)
    x_coord, y_coord, target_p, target_d = zip(*temp)
    x_coord, y_coord, target_p, target_d = list(x_coord), list(y_coord), list(target_p), list(target_d)

    return x_coord, y_coord, target_p, target_d, classA, classB


'''

