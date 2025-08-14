# encoding = "utf-8"
"""Dataset classes used for evaluation scripts."""

import json
from pathlib import Path
from typing import List, Dict, Any

from torch.utils.data import Dataset

__all__ = ["PreferenceDataset"]


class PreferenceDataset(Dataset):
    """A simple dataset wrapper for jsonl files used in RM/RM-pair evaluation."""

    def __init__(self, data_file: str | Path):
        self.data: List[Dict[str, Any]] = self._load_data(Path(data_file))

    # ---------------------------------------------------------------------
    def __len__(self) -> int:  # noqa: D401
        return len(self.data)

    def __getitem__(self, idx: int):  # noqa: D401
        return self.data[idx]

    # ------------------------------------------------------------------
    @staticmethod
    def _load_data(data_file: Path):
        data: List[Dict[str, Any]] = []
        with data_file.open("r", encoding="utf-8") as f:
            for line in f:
                sample = json.loads(line.strip())
                data.append(
                    {
                        "idx": sample["idx"],
                        "query": sample["prompt"],
                        "model_a": str(sample["model_a"]),
                        "model_b": str(sample["model_b"]),
                        "criteria": str(sample["criteria"]),
                        "preference": "A" if sample["preference"] == "model_a" else "B",
                    }
                )
        return data 