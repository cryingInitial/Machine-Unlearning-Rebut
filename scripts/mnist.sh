SEEDS=(0 1 2 3 4 5 6 7)
EPS=(4.0)
GPUS=(0 1 2 3 4 5 6 7)
TARGET_TYPE="blank"

for eps_idx in ${!EPS[@]}
do
    for seed_idx in ${!SEEDS[@]}
    do
        mkdir -p exp_data/mnist_$TARGET_TYPE/seed${SEEDS[$seed_idx]}/
        # gpu = 4 * eps_idx + seed_idx
        gpu_idx=$((4 * eps_idx + seed_idx))
        CUDA_VISIBLE_DEVICES=${GPUS[$gpu_idx]} python3 audit_model.py --data_name mnist_half --model_name cnn --n_epochs 100 --lr 1.33e-4 --epsilon ${EPS[$eps_idx]} \
            --fixed_init --target_type $TARGET_TYPE --n_reps 256 \
            --seed ${SEEDS[$seed_idx]} --out exp_data/mnist_$TARGET_TYPE/seed${SEEDS[$seed_idx]}/ --block_size 10000 > exp_data/mnist_$TARGET_TYPE/seed${SEEDS[$seed_idx]}/${EPS[$eps_idx]}.log 2>&1 &
    done
done