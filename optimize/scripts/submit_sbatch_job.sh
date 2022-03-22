#!/bin/bash
#SBATCH --partition=ml
#SBATCH --job-name=larndsim-fit
#SBATCH --output=output-%j.txt --error=output-%j.txt
#SBATCH --nodes=1
#SBATCH --mem=131072

INPUT_FILE=/sdf/group/neutrino/cyifan/muon-sim/fake_data_S1/edepsim-output.h5
SIF_FILE=/sdf/group/neutrino/images/latest.sif

singularity exec -B /sdf --nv ${SIF_FILE} \
            python3 -u -m optimize.example_run \
            --params lifetime \
            --input_file ${INPUT_FILE} \
            --batch_sz 1 \
            --num_workers ${SLURM_NTASKS}
