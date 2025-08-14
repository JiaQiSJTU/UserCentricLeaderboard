# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import os
import warnings
from collections import defaultdict
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any, Callable, Optional, Union

import pandas as pd
import torch
import torch.nn as nn
from accelerate import PartialState
from accelerate.utils import gather_object
from datasets import Dataset
from transformers import (
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    is_wandb_available,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_pt_utils import nested_detach
from transformers.trainer_utils import EvalPrediction
from transformers.utils import is_peft_available, is_rich_available

from trl.data_utils import maybe_apply_chat_template
from trl.trainer.reward_config import RewardConfig
from trl.trainer.utils import (
    RewardDataCollatorWithPadding,
    # compute_accuracy,
    decode_and_strip_padding,
    disable_dropout_in_model,
    generate_model_card,
    get_comet_experiment_url,
    log_table_to_comet_experiment,
    print_rich_table,
)

import numpy as np
import warnings


if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training

if is_wandb_available():
    import wandb


class CustomRewardPairTrainer(Trainer):
    _tag_names = ["trl", "reward-trainer"]

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module]] = None,
        args: Optional[RewardConfig] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional[dict] = None,
    ):
        """
        Initialize RewardTrainer.

        Args:
            model (`transformers.PreTrainedModel`):
                The model to train, preferably an `AutoModelForSequenceClassification`.
            args (`RewardConfig`):
                The arguments to use for training.
            data_collator (`transformers.DataCollator`):
                The data collator to use for training. If None is specified, the default data collator (`RewardDataCollatorWithPadding`) will be used
                which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
            train_dataset (`datasets.Dataset`):
                The dataset to use for training.
            eval_dataset (`datasets.Dataset`):
                The dataset to use for evaluation.
            processing_class (`PreTrainedTokenizerBase` or `BaseImageProcessor` or `FeatureExtractionMixin` or `ProcessorMixin`, *optional*):
                Processing class used to process the data. If provided, will be used to automatically process the inputs
                for the model, and it will be saved along the model to make it easier to rerun an interrupted training or
                reuse the fine-tuned model.
            model_init (`Callable[[], transformers.PreTrainedModel]`):
                The model initializer to use for training. If None is specified, the default model initializer will be used.
            compute_metrics (`Callable[[transformers.EvalPrediction], dict]`, *optional* defaults to `compute_accuracy`):
                The metrics to use for evaluation. If no metrics are specified, the default metric (`compute_accuracy`) will be used.
            callbacks (`list[transformers.TrainerCallback]`):
                The callbacks to use for training.
            optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
                The optimizer and scheduler to use for training.
            preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
                The function to use to preprocess the logits before computing the metrics.
            peft_config (`dict`, defaults to `None`):
                The PEFT configuration to use for training. If you pass a PEFT configuration, the model will be wrapped in a PEFT model.
        """
        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            if not isinstance(model, PeftModel):
                if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_quantized", False):
                    _supports_gc_kwargs = "gradient_checkpointing_kwargs" in list(
                        inspect.signature(prepare_model_for_kbit_training).parameters
                    )

                    prepare_model_kwargs = {"use_gradient_checkpointing": args.gradient_checkpointing}

                    if not _supports_gc_kwargs and args.gradient_checkpointing_kwargs is not None:
                        warnings.warn(
                            "You passed `gradient_checkpointing_kwargs` in the trainer's kwargs, but your peft version does not support it. "
                            "please update to the latest version of peft to use `gradient_checkpointing_kwargs`.",
                            UserWarning,
                        )
                    elif _supports_gc_kwargs and args.gradient_checkpointing_kwargs is not None:
                        prepare_model_kwargs["gradient_checkpointing_kwargs"] = args.gradient_checkpointing_kwargs

                    model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)

                model = get_peft_model(model, peft_config)

        # Disable dropout in the model
        if args.disable_dropout:
            disable_dropout_in_model(model)



        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in Reward, the sampled data does not include the
        # "input_ids" key. Instead, the available keys are "input_ids_chosen" and "input_ids_rejected". As a result,
        # the trainer issues the warning: "Could not estimate the number of tokens of the input, floating-point
        # operations will not be computed." To suppress this warning, we set the "estimate_tokens" key in the model's
        # "warnings_issued" dictionary to True. This acts as a flag to indicate that the warning has already been
        # issued.
        model.warnings_issued["estimate_tokens"] = True


        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)
