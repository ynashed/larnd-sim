#!/bin/bash
#SBATCH --partition=ml
#SBATCH --job-name=larndsim
#SBATCH --output=output-%j.txt --error=output-%j.txt
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --array=0,1,2,3,4,5,6

SIF_FILE=/sdf/group/neutrino/images/larndsim_latest.sif

params=( Ab kb eField lifetime tran_diff long_diff noise )

param=${params[$SLURM_ARRAY_TASK_ID]}

if [ $param -eq "noise" ]
then
    singularity exec -B /sdf --nv ${SIF_FILE} \
        python3 collect_dtws.py --n-noise 10 
else
    singularity exec -B /sdf --nv ${SIF_FILE} \
        python3 collect_dtws.py --param ${param}
fi
