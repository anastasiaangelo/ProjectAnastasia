import noisy_simulations_production as NSP
from supporting_functions import get_hyperparameters
import sys

num_rot_arr = [2, 3]
num_res_arr = [2, 3, 4, 5, 6, 7, 8, 9]
shots_arr = [10, 50, 100, 500, 1000, 5000]
alpha_arr = [0.0001, 0.001, 0.01, 0.1, 1]
p_arr = [1, 2]

if __name__ == "__main__":
    # get count arguments from command line, it is an integer
    match = int(sys.argv[1])
    num_rot, num_res, shots, alpha, p = get_hyperparameters(match, num_rot_arr, num_res_arr, shots_arr, alpha_arr, p_arr)
    NSP.noisy_simulation(num_rot=num_rot, num_res=num_res, shots=shots, alpha=alpha, p=p)



