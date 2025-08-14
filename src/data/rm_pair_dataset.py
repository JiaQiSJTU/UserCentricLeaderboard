import json
import random
import pickle as pkl
from typing import Dict, Sequence

import torch
from torch.utils.data import Dataset, random_split
import transformers

from utils import criteria_processor, make_up_dialogue 
from utils.prompt import PAIR_REWARD_MODEL_PROMPT 
from utils.seeding import set_global_seed

set_global_seed(42)

class RMPairPreferenceDataset(Dataset):
    """Dataset for pairwise reward model (choose between two)"""

    def __init__(self, data_file: str, tokenizer: transformers.PreTrainedTokenizer, data_direction: str = "original", data_noise: str = "none", data_choice: str = "our"):
        self.data_direction = data_direction
        self.data_noise = data_noise
        self.data_choice = data_choice
        self.tokenizer = tokenizer
        self.criteria_topic_df = pkl.load(open("./data/criteria_topic_df.pkl", "rb"))
        self.data = self._load_data(data_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    # ---------------- private ---------------- #
    def _load_data(self, data_file: str):
        data = []
        length_filter_count = 0
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                line = json.loads(line.strip())

                # Filter: human_winner == "tie"
                if line["human_winner"] == "tie":
                    continue

                if self.data_choice == "baseline" and line["human_winner"] == "tie":
                    continue
                if self.data_direction != "random" and line["human_winner"] == "tie":
                    continue

                messages, label = self._build_pair_message(line)
                input_txt = self.tokenizer.apply_chat_template(messages, tokenize=False)
                input_ids = self.tokenizer(input_txt, return_tensors="pt", padding=True)

                if input_ids.input_ids.size(1) > self.tokenizer.model_max_length:
                    length_filter_count += 1
                    continue

                data.append({
                    "input_ids": input_ids.input_ids[0],
                    "attention_mask": input_ids.attention_mask[0],
                    "label": torch.tensor(label),
                })
        print(f"[RMPairDataset] filtered {length_filter_count} samples, {len(data)} samples left")
        return data

    def _build_pair_message(self, line: Dict):
        if self.data_direction == "random" or line["human_winner"] == "tie":
            cur_direction = random.choice(["original", "reverse"])
        else:
            cur_direction = self.data_direction

        cur_noise = random.choice(["none", "add", "remove", "replace"]) if self.data_noise == "random" else self.data_noise

        if cur_direction == "original":
            if self.data_choice == "our":
                criteria = line["a_criteria"] if line["human_winner"] == "model_a" else line["b_criteria"]
                chosen_model = "model_a" if line["human_winner"] == "model_a" else "model_b"
                rejected_model = "model_b" if chosen_model == "model_a" else "model_a"
                rejected_criteria = line["a_criteria"] if rejected_model == "model_a" else line["b_criteria"]
                rejected_criteria_topic = line["a_broad_criteria_topic"] if rejected_model == "model_a" else line["b_broad_criteria_topic"]
                criteria = criteria_processor(criteria, rejected_criteria, rejected_criteria_topic, cur_noise, self.criteria_topic_df)
            else:
                criteria = ["General user preference"]

            label = 0 if line["human_winner"] == "model_a" else 1
        else:  # reverse
            criteria = line["b_criteria"] if line["human_winner"] == "model_a" else line["a_criteria"]
            label = 1 if line["human_winner"] == "model_a" else 0
            chosen_model = "model_b" if line["human_winner"] == "model_a" else "model_a"
            rejected_model = "model_b" if chosen_model == "model_a" else "model_a"
            rejected_criteria = line["a_criteria"] if rejected_model == "model_a" else line["b_criteria"]
            rejected_criteria_topic = line["a_broad_criteria_topic"] if rejected_model == "model_a" else line["b_broad_criteria_topic"]
            criteria = criteria_processor(criteria, rejected_criteria, rejected_criteria_topic, cur_noise, self.criteria_topic_df)

        # Build prompt
        query = line["model_a"][0]["content"]
        messages = [
            {
                "role": "user",
                "content": PAIR_REWARD_MODEL_PROMPT.format(
                    criteria="\n".join([f"* {c}" for c in criteria]),
                    user_query=json.dumps({"User": query}),
                    model_a=make_up_dialogue(line["model_a"][:]),
                    model_b=make_up_dialogue(line["model_b"][:]),
                ),
            }
        ]
        return messages, label

    # ---------------- public ---------------- #
    def train_test_split(self, test_size: float = 0.1, random_seed: int = 42):
        train_size = int(len(self.data) * (1 - test_size))
        test_size = len(self.data) - train_size
        generator = torch.Generator().manual_seed(random_seed)
        train_ds, test_ds = random_split(self, [train_size, test_size], generator=generator)
        return {"train": train_ds, "test": test_ds} 