import noisy_simulations_production as NSP
import sys

num_rot_arr = [2, 3]
num_res_arr = [2, 3, 4, 5, 6, 7, 8, 9]
shots_arr = [10, 50, 100, 500, 1000, 5000]
alpha_arr = [0.0001, 0.001, 0.01, 0.1, 1]
p_arr = [1, 2]

if __name__ == "__main__":
    # get count arguments from command line, it is an integer
    match = int(sys.argv[1])
    count = 1
    for num_rot in num_rot_arr:
        for num_res in num_res_arr:
            for shots in shots_arr:
                for alpha in alpha_arr:
                    if count == match:
                        NSP.noisy_simulation(num_rot, num_res, shots, alpha, p_arr)
                        break
                    count += 1


