


export MODEL=gpt2
export MODEL_NAME=/data/transformers/gpt2
export BATCH=4
export OUTPUT=/data/jwwang/dialogGen/simpletod/gpt2

export TRAIN_FILE=./resources/gpt2/train.history_belief_dbsearch_action_sys
export TEST_FILE=./resources/gpt2/val.history_belief_dbsearch_action_sys


CUDA_VISIBLE_DEVICES=1 python main.py \
    --output_dir=$OUTPUT \
    --model_type=$MODEL \
    --model_name_or_path=$MODEL_NAME \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --evaluate_during_training \
    --save_steps 5000 \
    --logging_steps 5000 \
    --per_gpu_train_batch_size $BATCH \
    --num_train_epochs 5