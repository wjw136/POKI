CUDA_VISIBLE_DEVICES=3 python generate_dialogue.py \
    --checkpoint /data/jwwang/dialogGen/simpletod/gpt2/checkpoint-25000 \
    --decoding greedy \
    --use_db_search \
    --use_dynamic_db