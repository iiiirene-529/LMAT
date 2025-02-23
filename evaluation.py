# evaluation.py

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score


def evaluate(model, dataloader, device, config):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].cpu().numpy()

            outputs1, outputs2 = model(input_ids=input_ids, attention_mask=attention_mask)
            probs1 = torch.softmax(outputs1, dim=1)[:, 1].cpu().numpy()
            probs2 = torch.softmax(outputs2, dim=1)[:, 1].cpu().numpy()

            # Averaging the probabilities from both classifiers
            probs = (probs1 + probs2) / 2
            preds = (probs >= 0.5).astype(int)

            all_labels.extend(labels)
            all_preds.extend(preds)
            all_probs.extend(probs)

    metrics = {}
    metrics['accuracy'] = accuracy_score(all_labels, all_preds)
    metrics['precision'] = precision_score(all_labels, all_preds)
    metrics['recall'] = recall_score(all_labels, all_preds)
    metrics['f1'] = f1_score(all_labels, all_preds)
    metrics['roc_auc'] = roc_auc_score(all_labels, all_probs)
    metrics['pr_auc'] = average_precision_score(all_labels, all_probs)

    return metrics
