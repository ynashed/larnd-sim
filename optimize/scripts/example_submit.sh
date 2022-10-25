#!/bin/bash
#SBATCH --partition=ml
#SBATCH --job-name=larndsim-fit
#SBATCH --output=output-%j.txt --error=output-%j.txt
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --array=1,2,3,4,5

INPUT_FILE=/sdf/group/neutrino/cyifan/muon-sim/fake_data_S1/edepsim-output.h5
SIF_FILE=/sdf/group/neutrino/images/larndsim_latest.sif
seed=$SLURM_ARRAY_TASK_ID
PARAM=Ab

singularity exec -B /sdf --nv ${SIF_FILE} \
  python3 -m optimize.example_run \
    --params ${PARAM} \
    --input_file ${INPUT_FILE} \
    --data_sz 5 \
    --batch_sz 5 \
    --no-noise \
    --num_workers 2 \
    --track_zlen_sel 0 \
    --out_label seed${seed}_${PARAM}_adam_SDTW_lr1e-2_5trk_test \
    --iterations 100 \
    --lr 1e-2 \
    --seed ${seed} \
    --optimizer_fn Adam \
    --loss_fn SDTW