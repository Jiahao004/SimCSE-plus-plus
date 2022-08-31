#!/bin/bash

# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.

export http_proxy="http://sys-proxy-rd-relay.byted.org:8118"
export https_proxy="http://sys-proxy-rd-relay.byted.org:8118"
export no_proxy="byted.org,bytedance.net,.byted.org,.bytedance.net,localhost,127.0.0.1,::1,10.0.0.0/8,127.0.0.0/8,fd00::/8,100.64.0.0/10,fe80::/10,172.16.0.0/12,169.254.0.0/16,192.168.0.0/16"



env=torch1.12.1_datasets1.18.3_cuda11

nvidia-smi
NUM_GPU=8

# Randomly set a port number
# If you encounter "address already used" error, just run again or manually set an available port id.
PORT_ID=$(expr $RANDOM + 1000)

# Allow multiple threads
export OMP_NUM_THREADS=8

model=sentence-transformers/bert-base-nli-cls-token
lr=1e-5
pos=1
# python3 -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID train_both.py \
python3 train_both.py \
    --model_name_or_path $model --pos_ratio $pos --fp16 \
    --train_file data/wiki1m_for_simcse.txt \
    --output_dir result/both-unsup-simcse-$model-$env-lr_$lr-pos_$pos \
    --num_train_epochs 1 \
    --per_device_train_batch_size 64 \
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
