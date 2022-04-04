#!/bin/bash
#SBATCH --partition=ml
#SBATCH --job-name=larndsim-fit
#SBATCH --output=output-%j.txt --error=output-%j.txt
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=64

INPUT_FILE=/sdf/group/neutrino/cyifan/muon-sim/fake_data_S1/edepsim-output.h5
SIF_FILE=/sdf/group/neutrino/images/larndsim_latest.sif

# Configuration
nproc_per_node=4
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )
master_node=${nodes_array[0]}
master_addr=$(srun --nodes=1 --ntasks=1 -w $master_node hostname --ip-address)
worker_num=$(($SLURM_JOB_NUM_NODES))

# Loop over nodes and submit training tasks
for ((  node_rank=0; node_rank<$worker_num; node_rank++ ))
do
  node=${nodes_array[$node_rank]}
  echo "Submitting node # $node_rank, $node"

  # Launch one SLURM task per node, and use torch distributed launch utility
  # to spawn training worker processes; one per GPU
  srun -N 1 -n 1 -w $node singularity exec -B /sdf --nv ${SIF_FILE} \
    python3 -m torch.distributed.launch \
    --nproc_per_node=$nproc_per_node --nnodes=$SLURM_JOB_NUM_NODES \
    --node_rank=$node_rank --master_addr=$master_addr \
    optimize/example_run.py \
            --params lifetime \
            --input_file ${INPUT_FILE}\
            --batch_sz 32 \
            --data_sz 1024 \
            --track_chunk 4 \
            --pixel_chunk 4 \
            --lr 1e-2 \
            --num_workers 16 &

  pids[${node_rank}]=$!
done

# Wait for completion
for pid in ${pids[*]}; do
    wait $pid
done
