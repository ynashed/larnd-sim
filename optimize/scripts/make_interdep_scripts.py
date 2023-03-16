import yaml
import os
import sys
sys.path.insert(0, '../..')
from optimize.ranges import ranges

relevant_params = ['Ab', 'kb', 'eField', 'lifetime', 'tran_diff', 'long_diff']

template_string = \
'''#!/bin/bash
#SBATCH --partition=ml
#SBATCH --job-name=larndsim-fit
#SBATCH --output=output-%j.txt --error=output-%j.txt
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=1
#SBATCH --time=40:00:00

INPUT_FILE=/sdf/group/neutrino/cyifan/muon-sim/fake_data_S1/edepsim-output.h5
SIF_FILE=/sdf/group/neutrino/images/larndsim_latest.sif
PARAM={param_spec}
seed=$SLURM_ARRAY_TASK_ID
max_abs_costheta_sel=0.966
min_abs_segz_sel=15.
batch_memory=32768
track_z_bound=28

singularity exec -B /sdf --nv ${{SIF_FILE}} \\
  python3 -m optimize.example_run \\
    --params ${{PARAM}} \\
    --input_file ${{INPUT_FILE}} \\
    --print_input \\
    --data_sz -1 \\
    --max_nbatch -1 \\
    --no-noise \\
    --num_workers 0 \\
    --track_len_sel 2 \\
    --track_z_bound ${{track_z_bound}} \\
    --max_abs_costheta_sel ${{max_abs_costheta_sel}} \\
    --min_abs_segz_sel ${{min_abs_segz_sel}} \\
    --random_ntrack \\
    --max_batch_len 100 \\
    --iterations 5000 \\
    --shift-no-fit {shift_param} \\
    --set-target-vals {target_val_str} \\
    --out_label interdep_5dim_plus_shift_{shift_param}_{shift_direction} \\
    --data_seed 2 \\
    --optimizer_fn Adam \\
    --loss_fn SDTW \\
    --link-vdrift-eField \\
    --batch_memory ${{batch_memory}} \\
    --skip_pixels \\
    --lr_scheduler ExponentialLR \\
    --lr_kw '{{"gamma" : 0.95 }}' \
'''

all_lrs = { 
            'Ab' : 1e-2,
            'kb' : 1e-2,
            'eField' : 1e-2,
            'tran_diff' : 1e-2,
            'long_diff' : 1e-2,
            'lifetime' : 5e-2
          }

for param in relevant_params:
    shift_param = param
    for direction in ['down', 'up']:
        shift_val = ranges[param][direction]
    
        target_val_str = ""
        lr_dict = {}
        for other in relevant_params:
            if other == param:
                target_val_str += f' {other} {shift_val}'
            else:
                nom_val = ranges[other]['nom']
                target_val_str += f' {other} {nom_val}'
                lr_dict[other] = all_lrs[other]

        target_val_str = target_val_str[1:]

        cwd = os.getcwd()
        base_dir = cwd[:cwd.find("optimize")]
        param_spec = os.path.join(base_dir, f"shift_{shift_param}_{direction}.yaml")
        with open(param_spec, "w") as file:
            yaml.dump(lr_dict, file)
        

        shell_script = template_string.format(param_spec=param_spec, 
                                              shift_param=shift_param, shift_direction=direction, 
                                              target_val_str=target_val_str)
        
        sh_fname = f"submit_sbatch_job_shift_{shift_param}_{direction}.sh"
        with open(sh_fname, "w") as file:
            file.write(shell_script)
