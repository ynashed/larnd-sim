#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --exclude=""
#SBATCH --account=neutrino:ml-dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --job-name=larndsim-fit
#SBATCH --output=output-%j.txt --error=output-%j.txt
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=1
#SBATCH --time=144:00:00
#SBATCH --array=1,2,3,4,5

seed=$SLURM_ARRAY_TASK_ID
seed_init=$SLURM_ARRAY_TASK_ID
data_seed=${seed}

#INPUT_FILE=/sdf/group/neutrino/cyifan/muon-sim/fake_data_S1/edepsim-output.h5
INPUT_FILE_TARGET=/sdf/group/neutrino/cyifan/muon-sim/fake_data_S1/edepsim_p_only_active_vol_5000.h5
INPUT_FILE_SIM=/sdf/group/neutrino/cyifan/muon-sim/fake_data_S1/edepsim_p_only_active_vol_5000_NIST_dEdx_cubic.h5

SIF_FILE=/sdf/group/neutrino/images/larndsim_latest.sif
PARAM=param_list.yaml

max_abs_costheta_sel=0.966
min_abs_segz_sel=15.
track_z_bound=28
track_len_sel=0
dx_low_limit=0.01
range_low_limit=0.15 #1.2
batch_memory=32768
max_grad_clip=1.
max_batch_len=200

label=p_only_contained_NIST_cubic_batch_len_${max_batch_len}_max_abs_costheta_${max_abs_costheta_sel}_min_abs_segz_${min_abs_segz_sel}_track_z_bound_${track_z_bound}_track_len_${track_len_sel}_dx_low_${dx_low_limit}_range_low_${range_low_limit}_no-noise-guess_tgtseed${seed}_dtseed${seed}

singularity exec -B /sdf --nv --bind /fs ${SIF_FILE} \
  python3 -m optimize.example_run \
    --params ${PARAM} \
    --input_file_sim ${INPUT_FILE_SIM} \
    --input_file_target ${INPUT_FILE_TARGET} \
    --data_sz -1 \
    --max_nbatch -1 \
    --no-noise-guess \
    --track_chunk 1\
    --pixel_chunk 1\
    --num_workers 0 \
    --track_len_sel ${track_len_sel} \
    --track_z_bound ${track_z_bound} \
    --max_abs_costheta_sel ${max_abs_costheta_sel} \
    --min_abs_segz_sel ${min_abs_segz_sel} \
    --dx_low_limit ${dx_low_limit} \
    --range_low_limit ${range_low_limit} \
    --max_batch_len ${max_batch_len} \
    --iterations 5000\
    --out_label ${label}\
    --seed ${seed} \
    --data_seed ${seed} \
    --random_ntrack \
    --optimizer_fn Adam \
    --loss_fn SDTW \
    --link-vdrift-eField \
    --batch_memory ${batch_memory} \
    --skip_pixels \
    --lr_scheduler ExponentialLR \
    --lr_kw '{"gamma" : 0.85 }' \
    --max_clip_norm_val ${max_grad_clip}
