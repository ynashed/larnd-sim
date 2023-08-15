#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --job-name=min5-loss-landscape-larndsim
#SBATCH --output=output-%j.txt --error=output-%j.txt
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=1
#SBATCH --time=36:00:00
#SBATCH --array=1,2,3,4,5

seed=$SLURM_ARRAY_TASK_ID
seed_init=$SLURM_ARRAY_TASK_ID
data_seed=6

# INPUT_FILE=/sdf/home/b/bkroul/l-sim/h5/proton_no_nuclei.h5
# INPUT_FILE=/sdf/home/b/bkroul/l-sim/h5/proton_max-dEdx2.h5
INPUT_FILE=/sdf/home/b/bkroul/l-sim/h5/proton_min-dEdx5.h5
label=proton_min-5_loss_i=seed${seed}_dtseed${data_seed}

SIF_FILE=/sdf/group/neutrino/images/larndsim_latest.sif
PARAM=/sdf/home/b/bkroul/larnd-sim/optimize/scripts/param_list.yaml

max_abs_costheta_sel=0.966; min_abs_segz_sel=15; track_z_bound=28; track_len_sel=2; # dataio values
batch_memory=32768; max_grad_clip=1 
# label=proton_noise_multipar_adaptive_chunk_linkvE_adam_SDTW_gradclip1_gamma0-95_costheta${max_abs_costheta_sel}_segz${min_abs_segz_sel}_z_bound${track_z_bound}_len${track_len_sel}_dtseed${seed}_iseed${seed_init}__btch${batch_memory}MB

singularity exec -B /sdf --nv --bind /fs ${SIF_FILE} \
  python3 -m optimize.loss_landscape \
    --preload \
    --vary-init \
    --seed-init ${seed_init} \
    --no-noise-guess \
    --seed ${seed} \
    --data_seed ${data_seed} \
    --out_label ${label} \
    --params ${PARAM} \
    --input_file ${INPUT_FILE} \
    --print_input \
    --data_sz -1 \
    --max_nbatch -1 \
    --num_workers 0 \
    --track_len_sel ${track_len_sel} \
    --track_z_bound ${track_z_bound} \
    --max_abs_costheta_sel ${max_abs_costheta_sel} \
    --min_abs_segz_sel ${min_abs_segz_sel} \
    --random_ntrack \
    --max_batch_len 100 \
    --iterations 5000 \
    --optimizer_fn Adam \
    --loss_fn SDTW \
    --link-vdrift-eField \
    --batch_memory ${batch_memory} \
    --skip_pixels \
    --lr_scheduler ExponentialLR \
    --lr_kw '{"gamma" : 0.95 }' \
    --max_clip_norm_val ${max_grad_clip}