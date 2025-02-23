# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from config import Config
from utils import save_checkpoint, load_checkpoint, setup_logging
from evaluation import evaluate
import logging
import pandas as pd
import os


def train_epoch(model, dataloader, criterion, optimizer, device, config):
    model.train()
    epoch_loss = 0
    all_labels = []
    all_preds = []
    all_probs = []

    for batch in tqdm(dataloader, desc='Training'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs1, outputs2 = model(input_ids=input_ids, attention_mask=attention_mask)

        loss1 = criterion(outputs1, labels)
        loss2 = criterion(outputs2, labels)
        loss = loss1 + loss2

        # Regularization
        l2_reg = torch.tensor(0.).to(device)
        for param in model.parameters():
            l2_reg += torch.norm(param, 2)
        loss += config.REGULARIZATION_COEFFICIENT * l2_reg

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
        optimizer.step()

        epoch_loss += loss.item()

        # Collect predictions for metrics
        probs1 = torch.softmax(outputs1, dim=1)[:, 1].detach().cpu().numpy()
        probs2 = torch.softmax(outputs2, dim=1)[:, 1].detach().cpu().numpy()
        probs = (probs1 + probs2) / 2
        preds = (probs >= 0.5).astype(int)

        all_labels.extend(labels.detach().cpu().numpy())
        all_preds.extend(preds)
        all_probs.extend(probs)

    # Calculate metrics
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    roc_auc = roc_auc_score(all_labels, all_probs)
    pr_auc = average_precision_score(all_labels, all_probs)

    return epoch_loss / len(dataloader), {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }


def train(model, train_loader, val_loader, config):
    device = config.DEVICE
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    best_val_f1 = 0.0
    metrics_history = []

    for epoch in range(config.NUM_EPOCHS):
        logging.info(f'Epoch {epoch + 1}/{config.NUM_EPOCHS}')
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, config)
        logging.info(f'Training Loss: {train_loss:.4f}')
        logging.info(f'Training Metrics: {train_metrics}')

        # Evaluate on Validation Set
        val_metrics = evaluate(model, val_loader, device, config)
        logging.info(f'Validation Metrics: {val_metrics}')

        # Save Metrics
        metrics_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_metrics['accuracy'],
            'train_precision': train_metrics['precision'],
            'train_recall': train_metrics['recall'],
            'train_f1': train_metrics['f1'],
            'train_roc_auc': train_metrics['roc_auc'],
            'train_pr_auc': train_metrics['pr_auc'],
            'val_accuracy': val_metrics['accuracy'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_f1': val_metrics['f1'],
            'val_roc_auc': val_metrics['roc_auc'],
            'val_pr_auc': val_metrics['pr_auc']
        })

        df_metrics = pd.DataFrame(metrics_history)
        if not os.path.exists('results'):
            os.makedirs('results')
        df_metrics.to_csv('results/training_metrics.csv', index=False)

        # Checkpointing
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            save_checkpoint(model, optimizer, epoch, best_val_f1, config)
            logging.info('Model checkpoint saved.')
