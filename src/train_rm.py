# encoding = "utf-8"

import torch
import warnings
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser, EarlyStoppingCallback, TrainerCallback
import trl
from trl import (
    ModelConfig,
    RewardConfig,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from dataclasses import dataclass, field
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE


from trainer.reward_trainer import CustomRewardTrainer
from utils import safe_save_model_for_hf_trainer
from utils.metrics import compute_reward_accuracy
from utils.seeding import set_global_seed
from data import RMPreferenceDataset, RMDataCollator

set_global_seed(42)


@dataclass
class DataArguments:

    data_path: str = field(default="", metadata={"help": "Path to the training data."})
    data_direction: str = field(default="original", metadata={"help": "Data direction: random, original, reverse"})
    data_noise: str = field(default="none", metadata={"help": "Data noise: random, none, add, remove, replace"})



if __name__ == "__main__":
    parser = HfArgumentParser((DataArguments, RewardConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=True)

    '''Model & Tokenizer'''
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
        model_args.model_name_or_path, num_labels=1, trust_remote_code=model_args.trust_remote_code, attn_implementation="flash_attention_2", **model_kwargs
    )
    # Align padding tokens between tokenizer and model
    model.config.pad_token_id = tokenizer.pad_token_id
    # print(tokenizer.pad_token_id, tokenizer.eos_token_id)


    # If post-training a base model, use ChatML as the default template
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

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
    dataset = RMPreferenceDataset(script_args.data_path, tokenizer,script_args.data_direction, script_args.data_noise)
    # print(len(dataset))
    # exit(0)
    dataset = dataset.train_test_split(test_size=0.1) 
    data_collator = RMDataCollator(tokenizer)

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.0, # Optional: only stop if improvement is less than this
    )

    # print(len(dataset["train"]), len(dataset["test"]))
    # exit(0)
    
    '''trainer'''
    trainer = CustomRewardTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=get_peft_config(model_args),
        data_collator=data_collator,
        compute_metrics=compute_reward_accuracy,
        callbacks=[early_stopping_callback]
    )
    trainer.train()

    '''save model'''

    if training_args.eval_strategy != "no":
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
   