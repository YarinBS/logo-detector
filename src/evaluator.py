"""
Evaluator module
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def evaluate(model: nn.Module, dataloader: DataLoader) -> dict:
    """
    Evaluates the model on the given dataloader and returns a dictionary of evaluation metrics

    Parameters:
    - model (nn.Module): The trained model to evaluate.
    - dataloader (DataLoader): The DataLoader for the evaluation dataset.

    Returns:
    - dict: A dictionary containing evaluation metrics such as accuracy, precision, recall, and F1 score.
    """

    model.eval()
    tp = fp = tn = fn = 0

    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            predictions = torch.sigmoid(outputs).squeeze() > 0.5
            labels = labels.squeeze()

            tp += ((predictions == 1) & (labels == 1)).sum().item()
            fp += ((predictions == 1) & (labels == 0)).sum().item()
            tn += ((predictions == 0) & (labels == 0)).sum().item()
            fn += ((predictions == 0) & (labels == 1)).sum().item()
    
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }
