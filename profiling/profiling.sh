#!/bin/bash

#export PYTHONPATH=$PYTHONPATH:/sdf/home/p/pgranger/larnd-sim/
# export SINGULARITYENV_APPEND_PATH=/sdf/home/p/pgranger/nsys/opt/nvidia/nsight-systems-cli/2022.4.1/target-linux-x64/
INPUT_FILE=/sdf/group/neutrino/cyifan/muon-sim/fake_data_S1/edepsim-output.h5
SIF_FILE=/sdf/group/neutrino/images/larndsim_latest.sif
seed=1
PARAM=Ab
#nsys profile -f true -c cudaProfilerApi --capture-range-end=stop-shutdown -o profile_loop --cuda-memory-usage=true
singularity exec -B /sdf --nv ${SIF_FILE} \
    python3 -m optimize.example_run \
    --params ${PARAM} \
    --input_file ${INPUT_FILE} \
    --data_sz 1 \
    --batch_sz 1 \
    --no-noise \
    --num_workers 2 \
    --track_len_sel 0 \
    --out_label seed${seed}_${PARAM}_adam_SDTW_lr1e-2_5trk_test \
    --iterations 1 \
    --print_input \
    --lr 1e-2 \
    --seed ${seed} \
    --optimizer_fn Adam \
    --loss_fn SDTW \
    --cpuprof
