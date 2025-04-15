#!/bin/bash
echo "Start running Prob-T Model test..."
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

srun -G 1 accelerate launch run_clm.py \
    --config_name configs/prob_t_tiny.json \
    --tokenizer_name TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --auto_find_batch_size \
    --gradient_accumulation_steps 1 \
    --block_size 2048 \
    --attn_implementation eager \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.015 \
    --learning_rate 3e-4 \
    --weight_decay 1e-1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --do_train \
    --do_eval \
    --do_predict \
    --num_train_epochs 1 \
    --save_total_limit 1 \
    --save_strategy steps \
    --save_steps 200 \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --logging_steps 50 \
    --load_best_model_at_end True \
    --metric_for_best_model eval_loss \
    --report_to none \
    --output_dir outputs/prob-t-tiny 