#!/bin/bash

# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.




nvidia-smi
NUM_GPU=4

# Randomly set a port number
# If you encounter "address already used" error, just run again or manually set an available port id.
PORT_ID=$(expr $RANDOM + 1000)

# Allow multiple threads
export OMP_NUM_THREADS=8

#model=sentence-transformers/bert-base-nli-cls-token
model=roberta-base

env=torch1.12.1-cuda10.2
lr=1e-5
bs=64
# Use distributed data parallel
# If you only want to use one card, uncomment the following line and comment the line with "torch.distributed.launch"
# python3 -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID train.py \
python train.py \
    --model_name_or_path $model --fp16 --overwrite_cache \
    --train_file data/wiki1m_for_simcse.txt \
    --output_dir result/unsup-simcse-$model-lr_$lr-bs_$bs-$env-fp16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size $bs \
    --learning_rate $lr \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    "$@"
