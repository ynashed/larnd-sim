from optimize.ranges import ranges
import numpy as np

def make_submit_str(param=None, value=None, noise=False, target=False, noise_iter=None):
    base_str =  '''#!/bin/bash
#SBATCH --partition=ml
#SBATCH --job-name=larndsim
#SBATCH --output=output-%j.txt --error=output-%j.txt
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00

SIF_FILE=/sdf/group/neutrino/images/larndsim_latest.sif

singularity exec -B /sdf --nv ${SIF_FILE} \\
    python3 run_batch_sim.py \\
'''
   
    if target:
        save_name = "nominal_vals"
        base_str += f"       --save-name {save_name} \n"
    elif noise and noise_iter is not None:
        save_name = f"noisy_sample{noise_iter}"
        base_str += f"       --save-name {save_name} \\\n       --noise \n"
    else:
        save_name = f"shift_{param}_{value}"
        base_str += f"       --params {param} \\\n       --param-vals {value} \\\n       --save-name {save_name} \n"

    return base_str

submit_str = make_submit_str(target=True)
with open("configs/target_submit.sh", "w") as file:
    file.write(submit_str)

for iter in range(10):
    submit_str = make_submit_str(noise=True, noise_iter=iter)
    with open(f"configs/noisy_submit_iter{iter}.sh", "w") as file:
        file.write(submit_str)

param_list = ['Ab', 'kb', 'lifetime', 'long_diff', 'tran_diff', 'eField']

eval_vals = {}
for param in param_list:
    eval_vals = np.linspace(ranges[param]['down'], ranges[param]['up'], 20)

    for val in eval_vals:
        submit_str = make_submit_str(param=param, value=val)
        with open(f"configs/shift_{param}_{val}_submit.sh", "w") as file:
            file.write(submit_str)
