# encoding = "utf-8"
"""Dataset classes used for evaluation scripts."""

import json
from pathlib import Path
from typing import List, Dict, Any
import random

from torch.utils.data import Dataset

__all__ = ["PreferenceDataset"]


class PreferenceDataset(Dataset):
    """A simple dataset wrapper for jsonl files used in RM/RM-pair evaluation."""

    def __init__(self, data_file: str | Path, shuffle: bool = True):
        self.shuffle = shuffle
        self.data: List[Dict[str, Any]] = self._load_data(Path(data_file))

    def __len__(self) -> int:  
        return len(self.data)

    def __getitem__(self, idx: int):  
        return self.data[idx]

    def _load_data(self, data_file: Path):
        data: List[Dict[str, Any]] = []
        with data_file.open("r", encoding="utf-8") as f:
            for line in f:
                sample = json.loads(line.strip())
                order = random.choice(["12", "21"])

                if self.shuffle:
                    if order == "12":
                        model_a, model_b = sample["model_a"], sample["model_b"]
                        preference = "A" if sample["preference"] == "model_a" else "B"
                    else:
                        model_a, model_b = sample["model_b"], sample["model_a"]  
                        preference = "B" if sample["preference"] == "model_a" else "A"
                else:
                    model_a, model_b = sample["model_a"], sample["model_b"]
                    preference = "A" if sample["preference"] == "model_a" else "B"
                
                data.append(
                    {
                        "idx": sample["idx"],
                        "query": sample["prompt"],
                        "model_a": str(model_a),
                        "model_b": str(model_b),
                        "criteria": str(sample["criteria"]),
                        "preference": preference,
                    }
                )
        return data 