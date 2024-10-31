#!/bin/bash

#SBATCH --partition=milano
#SBATCH --account=neutrino
#
#SBATCH --job-name=larndsim
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8g
#
#SBATCH --time=3:00:00

#BASE DECLARATIONS
TARGET_SEED=3
# PARAMS="tran_diff long_diff Ab kb lifetime"
BATCH_SIZE=500
ITERATIONS=4000
DATA_SEED=1
# INPUT_FILE=/sdf/group/neutrino/cyifan/muon-sim/fake_data_S1/edepsim-output.h5
INPUT_FILE=/home/pgranger/larnd-sim/jit_version/larnd-sim/data/mixed_sample/edepsim-output.h5
UUID=$(uuidgen)
#DECLARATIONS


PARAMS="param_list_lr.yaml"


# export JAX_LOG_COMPILES=1
# singularity exec --bind /sdf,$SCRATCH python-jax.sif python3 -m optimize.example_run \
python3 -m optimize.example_run \
    --print_input \
    --data_sz -1 \
    --max_nbatch 40 \
    --params ${PARAMS} \
    --input_file ${INPUT_FILE} \
    --track_len_sel 2 \
    --max_abs_costheta_sel 0.966 \
    --min_abs_segz_sel 15. \
    --no-noise \
    --data_seed ${DATA_SEED} \
    --num_workers 0 \
    --out_label seed${TARGET_SEED}_tdiff-vdrift_ds${DATA_SEED}_adam_SDTW_lr1e-2_5trk_test_${UUID} \
    --seed ${TARGET_SEED} \
    --optimizer_fn Adam \
    --random_ntrack \
    --iterations ${ITERATIONS} \
    --max_batch_len ${BATCH_SIZE} \
    --skip_pixels \
    --link-vdrift-eField \
    --batch_memory 32768 \
    --lr_scheduler exponential_decay \
    --lr_kw '{"decay_rate" : 0.996, "transition_steps": 10}' \
    --track_z_bound 28 \
    --max_clip_norm_val 1 \
    --loss_fn SDTW \
    --electron_sampling_resolution 0.005 \
    --number_pix_neighbors 3 \
    --signal_length 300 \
    --mode 'lut' \
    --lut_file /home/pgranger/larnd-sim/jit_version/original/build/lib/larndsim/bin/response_44.npy
    # --number_pix_neighbors 0 \
    # --signal_length 191 \
    # --mode 'parametrized'
    # --profile_gradient 
    # --loss_fn space_match
