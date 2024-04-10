#BASE DECLARATIONS
TARGET_SEED=2
# PARAMS="tran_diff long_diff Ab kb lifetime"
BATCH_SIZE=200
ITERATIONS=2500
DATA_SEED=1
INPUT_FILE=/home/pgranger/larnd-sim/jit_version/larnd-sim/data/muon/edepsim-output.h5
# INPUT_FILE=/home/pgranger/larnd-sim/jit_version/larnd-sim/examples/module0_corsika.h5
UUID=$(uuidgen)
#DECLARATIONS


PARAMS="param_list_lr.yaml"


# JAX_LOG_COMPILES=1
python3 -m optimize.example_run \
    --print_input \
    --data_sz -1 \
    --max_nbatch 10 \
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
    --loss_fn SDTW \
    --random_ntrack \
    --iterations ${ITERATIONS} \
    --max_batch_len ${BATCH_SIZE} \
    --skip_pixels \
    --link-vdrift-eField \
    --batch_memory 32768 \
    --lr_scheduler exponential_decay \
    --lr_kw '{"decay_rate" : 0.996, "transition_steps": 10}' \
    --track_z_bound 28 \
    --max_clip_norm_val 1
