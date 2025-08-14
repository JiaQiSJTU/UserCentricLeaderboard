CUDA_VISIBLE_DEVICES=0 python3 src/train_rm.py \
    --model_name_or_path ./Qwen/Qwen3-4B \
    --data_path ./data/train_val_data.jsonl \
    --bf16 True \
    --output_dir ./outputs-model/Qwen3-4B-rm-random-random-wo-tie \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 32 \
    --eval_strategy "steps" \
    --eval_steps 50 \
    --save_strategy "best" \
    --save_total_limit 5 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to tensorboard \
    --tf32 True \
    --gradient_checkpointing True \
    --load_best_model_at_end True \
    --metric_for_best_model eval_loss \
    --greater_is_better False \
    --data_direction random \
    --data_noise random


