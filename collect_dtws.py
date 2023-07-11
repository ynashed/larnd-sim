import torch
from glob import glob
import os
import sys
sys.path.insert(0, "../")
from optimize.ranges import ranges
from tqdm import tqdm
import pickle

import numpy as np
import argparse
    
def dtw(s, t):
    n, m = len(s), len(t)
    dtw_matrix = np.zeros((n+1, m+1))
    for i in range(n+1):
        for j in range(m+1):
            dtw_matrix[i, j] = np.inf
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(s[i-1] - t[j-1])
            # take last min from a square box
            last_min = np.min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
            dtw_matrix[i, j] = cost + last_min
    return dtw_matrix


def main(param=None, noise=False, n_noise=None):
    individual_f_list = os.listdir("nominal_vals_batch/")

    target_file_dir = "nominal_vals_batch/"
    all_dist = {}

    if noise:
        print("noise")
        all_dist['noise'] = {}
        for noise_iter in range(n_noise):
            all_dist['noise'][noise_iter] = []
            dir_name = f"noisy_sample{noise_iter}_batch/"
            for file in individual_f_list:
                full_path = os.path.join(dir_name, file)
                full_path_target = os.path.join(target_file_dir, file)
                
                output_shift = torch.load(full_path).cpu().numpy()[-1]
                target = torch.load(full_path_target).cpu().numpy()[-1]
                
                dtw_val = dtw(output_shift, target)[-1][-1]
                all_dist['noise'][noise_iter].append(dtw_val)
        
        with open("collect_adc_dtw_noise.pkl", "wb") as file:
            pickle.dump(all_dist, file)
            
    else:
        print(param)
        all_dist[param] = {}
        all_param_vals = np.linspace(ranges[param]['down'], ranges[param]['up'], 20)
        for val in tqdm(all_param_vals):
            all_dist[param][val] = []
            dir_name = f"shift_{param}_{val}_batch"
            for file in individual_f_list:
                full_path = os.path.join(dir_name, file)
                full_path_target = os.path.join(target_file_dir, file)
                
                output_shift = torch.load(full_path).cpu().numpy()[-1]
                target = torch.load(full_path_target).cpu().numpy()[-1]
                
                dtw_val = dtw(output_shift, target)[-1][-1]
                all_dist[param][val].append(dtw_val)
    
        with open(f"collect_adc_dtw_{param}.pkl", "wb") as file:
            pickle.dump(all_dist, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--param", dest="param", default=None,
                        help="List of parameters to plot.")
    parser.add_argument("--n-noise", dest="n_noise", default=None,
                        help="Number of noise iterations")
    args = parser.parse_args()

    noise=False
    n_noise=None
    if args.n_noise is not None:
        noise=True
        n_noise = int(args.n_noise)

    main(param=args.param, noise=noise, n_noise=n_noise)