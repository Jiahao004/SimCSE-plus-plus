#!/bin/bash

# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.



nvidia-smi

NUM_GPU=2

# Randomly set a port number
# If you encounter "address already used" error, just run again or manually set an available port id.
PORT_ID=$(expr $RANDOM + 1000)

# Allow multiple threads
export OMP_NUM_THREADS=8


lr=3e-5
bs=64
model=bert-base-uncased
env=torch1.10.2_datasets1.18.3_cuda10.2
pos=1

#python3 -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID train_re_vicreg.py \
python3 train_re_vicreg.py --fp16  \
    --model_name_or_path $model --pos_ratio $pos \
    --train_file data/wiki1m_for_simcse.txt \
    --output_dir result/re-unsup-simcse-$model-$env-lr_$lr-bs_$bs-vicreg_t5_w0.1_pos$pos \
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
