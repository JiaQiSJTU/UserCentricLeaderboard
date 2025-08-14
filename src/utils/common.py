# encoding = "utf-8"

import json
import random
from typing import List, Sequence

import torch
import transformers


__all__ = [
    "make_up_dialogue",
    "pad_sequence_left",
    "safe_save_model_for_hf_trainer",
    "criteria_processor",
]



def make_up_dialogue(utterances: List[dict]) -> str:
    """Format multi-turn dialogue as a JSON string for display in prompts."""
    outputs = []
    for utt in utterances[1:]:
        if utt["role"] == "user":
            outputs.append({"User": utt["content"]})
        else:
            outputs.append({"Model": utt["content"]})
    return json.dumps(outputs, indent=2)


def pad_sequence_left(sequences: Sequence[torch.Tensor], batch_first: bool = True, padding_value: int = 0):
    """Pad sequences on the left side."""
    reversed_sequences = [seq.flip(0) for seq in sequences]
    padded = torch.nn.utils.rnn.pad_sequence(reversed_sequences, batch_first=batch_first, padding_value=padding_value)
    return padded.flip(1)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Save a HuggingFace Trainer model to disk with CPU weights."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def criteria_processor(criteria: list, neg_criteria: list, neg_criteria_topic: list, cur_noise: str, criteria_topic_df):
    """Process the list of criteria according to the noise strategy."""
    if cur_noise == "remove":
        criteria = random.sample(criteria, int(len(criteria) * 0.5))
    elif cur_noise == "add":
        candidate_list = criteria_topic_df[~criteria_topic_df["Topic"].isin(neg_criteria_topic)]["Document"].tolist()
        criteria = criteria + random.sample(candidate_list, int(len(criteria) * 0.5))
        criteria = random.sample(criteria, len(criteria))
    elif cur_noise == "replace":
        replacements = random.sample(neg_criteria, int(len(criteria) * 0.5) - 1)
        replacements_idx = random.sample(range(len(criteria)), int(len(criteria) * 0.5) - 1)
        for idx, r_idx in enumerate(replacements_idx):
            criteria[r_idx] = replacements[idx]
    return criteria 