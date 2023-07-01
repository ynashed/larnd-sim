#!/bin/bash
#SBATCH --partition=ml
#SBATCH --job-name=larndsim-fit
#SBATCH --output=output-%j.txt --error=output-%j.txt
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=1
#SBATCH --time=72:00:00
#SBATCH --array=1,2,3,4,5

INPUT_FILE=/sdf/group/neutrino/cyifan/muon-sim/fake_data_S1/edepsim-output.h5
SIF_FILE=/sdf/group/neutrino/images/larndsim_latest.sif
seed=$SLURM_ARRAY_TASK_ID
PARAM=param_list.yaml
max_abs_costheta_sel=0.966
min_abs_segz_sel=15
track_z_bound=28
track_len_sel=2
batch_memory=32768
max_grad_clip=1

singularity exec -B /sdf --nv ${SIF_FILE} \
  python3 -m optimize.example_run \
    --params ${PARAM} \
    --input_file ${INPUT_FILE} \
    --print_input \
    --data_sz -1 \
    --max_nbatch -1 \
    --no-noise \
    --num_workers 0 \
    --track_len_sel ${track_len_sel} \
    --track_z_bound ${track_z_bound} \
    --max_abs_costheta_sel ${max_abs_costheta_sel} \
    --min_abs_segz_sel ${min_abs_segz_sel} \
    --random_ntrack \
    --max_batch_len 100 \
    --iterations 5000 \
    --out_label muon_no-noise_multipar_adaptive_chunk_linkvE_adam_SDTW_gradclip1_gamma0-95_max-abs-costheta${max_abs_costheta_sel}_min-abs-segz${min_abs_segz_sel}_track_z_bound${track_z_bound}_track_len${track_len_sel}_tgtseed${seed}_dtseed${seed}__btch${batch_memory}MB \
    --seed ${seed} \
    --data_seed ${seed} \
    --optimizer_fn Adam \
    --loss_fn SDTW \
    --link-vdrift-eField \
    --batch_memory ${batch_memory} \
    --skip_pixels \
    --lr_scheduler ExponentialLR \
    --lr_kw '{"gamma" : 0.95 }' \
    --max_clip_norm_val ${max_grad_clip}

