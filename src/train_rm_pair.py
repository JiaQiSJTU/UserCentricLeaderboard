# encoding = "utf-8"

import torch
import warnings
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser, EarlyStoppingCallback
from trl import (
    ModelConfig,
    RewardConfig,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from dataclasses import dataclass, field
import transformers
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
from utils.seeding import set_global_seed
set_global_seed(42)

from trainer.reward_pair_trainer import CustomRewardPairTrainer
from utils import safe_save_model_for_hf_trainer
from utils.metrics import compute_pair_accuracy as compute_metrics
from data import (
    RMPairPreferenceDataset,
    RMPairDataCollator,
)


@dataclass
class DataArguments:

    data_path: str = field(default="", metadata={"help": "Path to the training data."})
    data_direction: str = field(default="original", metadata={"help": "Data direction: random, original, reverse"})
    data_noise: str = field(default="none", metadata={"help": "Data noise: random, none, add, remove, replace"})
    data_choice: str = field(default="our", metadata={"help": "Data choice: our, baseline"})


if __name__ == "__main__":
    parser = HfArgumentParser((DataArguments, RewardConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=True)

    '''Model & Tokenizer'''
    print("loading model")
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        use_cache=False,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path, num_labels=2, trust_remote_code=model_args.trust_remote_code, attn_implementation="flash_attention_2", **model_kwargs
    )
    # Align padding tokens between tokenizer and model
    model.config.pad_token_id = tokenizer.pad_token_id
    # print(tokenizer.pad_token_id, tokenizer.eos_token_id)


    # If post-training a base model, use ChatML as the default template
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
        # model, tokenizer = setup_chat_format(model, tokenizer)

    if model_args.use_peft and model_args.lora_task_type != "SEQ_CLS":
        warnings.warn(
            "You are using a `task_type` that is different than `SEQ_CLS` for PEFT. This will lead to silent bugs"
            " Make sure to pass --lora_task_type SEQ_CLS when using this script with PEFT.",
            UserWarning,
        )

    training_args.max_length = 2048 # model.config.max_position_embeddings
    tokenizer.model_max_length = 2048 # model.config.max_position_embeddings
    tokenizer.padding_side = "right"

    if not hasattr(training_args, 'metric_for_best_model') or training_args.metric_for_best_model is None:
        training_args.metric_for_best_model = "eval_loss" 
        training_args.greater_is_better = False
        
    '''dataset'''
    print("loading dataset")
    if script_args.data_choice=="baseline":
        script_args.data_direction = "original"
        script_args.data_noise = "none"

    dataset = RMPairPreferenceDataset(script_args.data_path, tokenizer,script_args.data_direction, script_args.data_noise, script_args.data_choice)
    dataset = dataset.train_test_split(test_size=0.1) 
    data_collator = RMPairDataCollator(tokenizer)

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.0, # Optional: only stop if improvement is less than this
    )

    '''trainer'''
    trainer = CustomRewardPairTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=get_peft_config(model_args),
        data_collator=data_collator,
        compute_metrics=compute_metrics, 
        callbacks=[early_stopping_callback], # Add the early stopping callback
    )
    trainer.train()

    '''save model'''

    if training_args.eval_strategy != "no":
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
   