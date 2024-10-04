#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --job-name=dataio-cut-larndsim
#SBATCH --output=output-%j.txt --error=output-%j.txt
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00

# muon input file
INPUT_FILE=/fs/ddn//sdf/group/neutrino/cyifan/muon-sim/fake_data_S1/edepsim-output.h5
# proton input file
INPUT_FILE=/fs/ddn/sdf/group/neutrino/cyifan/muon-sim/larndsim_output/f1_1000_p_high_KE/edepsim-output.h5

SIF_FILE=/sdf/group/neutrino/images/larndsim_latest.sif
MIN_DEDX=10
OUT_LABEL=proton_no_nuclei_min${MIN_DEDX}
#change cutting min to max with --max
# dataio values
max_abs_costheta_sel=0.966; min_abs_segz_sel=15; track_z_bound=28; track_len_sel=2


singularity exec --bind /fs --nv ${SIF_FILE} \
  python3 make_dataio_cut.py ${INPUT_FILE} \
  --output ${OUT_LABEL} \
  --threshold ${MIN_DEDX} \
  --max_abs_costheta_sel ${max_abs_costheta_sel} \
  --min_abs_segz_sel ${min_abs_segz_sel} \
  --track_z_bound ${track_z_bound} \
  --track_len_sel ${track_len_sel} \



  
  

