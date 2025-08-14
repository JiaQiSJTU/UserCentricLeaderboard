from __future__ import annotations
import numpy as np
from transformers.trainer_utils import EvalPrediction

__all__ = [
    "compute_reward_accuracy",
    "compute_pair_accuracy",
]


def compute_reward_accuracy(eval_pred: EvalPrediction) -> dict[str, float]:    
    predictions, labels = eval_pred
    data_size = len(predictions)

    # We want to see how much of the time rewards_chosen > rewards_rejected.
    equal_mask = predictions[:, 0] == predictions[:, 1]
    equal_predictions_count = int(equal_mask.sum())

    # Filter out equal predictions
    predictions = predictions[~equal_mask]
    labels = labels[~equal_mask]

    # Use the remaining predictions for accuracy calculation
    predictions = np.argmax(predictions, axis=1)
    accuracy = (np.array(predictions == labels, dtype=float).sum().item()/data_size) # regard the equal predictions as wrong

    return {"accuracy": accuracy}


def compute_pair_accuracy(eval_pred: EvalPrediction) -> dict[str, float]:
    """Pairwise binary classification accuracy."""
    predictions, labels = eval_pred
    pred_cls = np.argmax(predictions, axis=1)
    accuracy = float((pred_cls == labels).sum()) / len(predictions)
    return {"accuracy": accuracy}
