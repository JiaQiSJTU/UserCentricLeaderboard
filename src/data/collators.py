# encoding = "utf-8"
from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import transformers

__all__ = ["RMDataCollator", "RMPairDataCollator"]

@dataclass
class RMDataCollator(object):

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids_chosen, attention_mask_chosen, input_ids_rejected, attention_mask_rejected = (
            tuple([instance[key] for instance in instances] for key in ("input_ids_chosen", "attention_mask_chosen", "input_ids_rejected", "attention_mask_rejected"))
        )

        input_ids_chosen = torch.nn.utils.rnn.pad_sequence(
            input_ids_chosen, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_mask_chosen = torch.nn.utils.rnn.pad_sequence(
            attention_mask_chosen, batch_first=True, padding_value=0
        )
        input_ids_rejected = torch.nn.utils.rnn.pad_sequence(
            input_ids_rejected, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_mask_rejected = torch.nn.utils.rnn.pad_sequence(
            attention_mask_rejected, batch_first=True, padding_value=0
        )

        return {
            "input_ids_chosen": input_ids_chosen,
            "attention_mask_chosen": attention_mask_chosen,
            "input_ids_rejected": input_ids_rejected,
            "attention_mask_rejected": attention_mask_rejected,
        }


@dataclass
class RMPairDataCollator(object):

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, attention_mask, labels = (
            tuple([instance[key] for instance in instances] for key in ("input_ids", "attention_mask", "label"))
        )

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = torch.stack(labels)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        } 