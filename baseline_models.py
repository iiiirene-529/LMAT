# baseline_models.py

import torch
import torch.nn as nn
from transformers import LlamaModel, GPT3Tokenizer, GPT3Model  # Hypothetical GPT-3 classes
from config import Config
from data_preprocessing import get_data_loaders
from evaluation import evaluate
from utils import setup_logging
import logging


class LlamaBaseline(nn.Module):
    def __init__(self, config):
        super(LlamaBaseline, self).__init__()
        self.llama = LlamaModel.from_pretrained(config.TOKENIZER_PATH)
        self.classifier = nn.Sequential(
            nn.Linear(self.llama.config.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Binary classification
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        outputs = self.llama(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch, seq, hidden_size)
        pooled = hidden_states.mean(dim=1)  # (batch, hidden_size)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)  # (batch, 2)
        return logits


class GPT3Baseline(nn.Module):
    def __init__(self, config):
        super(GPT3Baseline, self).__init__()
        self.gpt3 = GPT3Model.from_pretrained('gpt-3')  # Placeholder
        self.classifier = nn.Sequential(
            nn.Linear(self.gpt3.config.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Binary classification
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        outputs = self.gpt3(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch, seq, hidden_size)
        pooled = hidden_states.mean(dim=1)  # (batch, hidden_size)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)  # (batch, 2)
        return logits


def evaluate_baselines(config):
    setup_logging(config)
    logging.info('Evaluating Baseline Models')

    # Data Loaders
    _, _, test_loader = get_data_loaders(config)

    device = config.DEVICE

    # Initialize Baseline Models
    llama_baseline = LlamaBaseline(config).to(device)
    gpt3_baseline = GPT3Baseline(config).to(device)

    # Load pretrained weights if available
    # Example:
    # llama_baseline.load_state_dict(torch.load('path_to_llama_baseline.pth'))
    # gpt3_baseline.load_state_dict(torch.load('path_to_gpt3_baseline.pth'))

    # Define Loss Function
    criterion = nn.CrossEntropyLoss()

    # Evaluate Llama-7B Baseline
    logging.info('Evaluating Llama-7B Baseline')
    llama_metrics = evaluate(llama_baseline, test_loader, device, config)
    logging.info(f'Llama-7B Baseline Metrics: {llama_metrics}')

    # Evaluate GPT-3 Baseline
    logging.info('Evaluating GPT-3 Baseline')
    gpt3_metrics = evaluate(gpt3_baseline, test_loader, device, config)
    logging.info(f'GPT-3 Baseline Metrics: {gpt3_metrics}')

    # Save Results
    results = {
        'Model': ['Llama-7B (baseline)', 'GPT-3 (baseline)'],
        'Accuracy (%)': [llama_metrics['accuracy'] * 100, gpt3_metrics['accuracy'] * 100],
        'AUC': [llama_metrics['roc_auc'], gpt3_metrics['roc_auc']],
        'Recall': [llama_metrics['recall'], gpt3_metrics['recall']],
        'F1-Score': [llama_metrics['f1'], gpt3_metrics['f1']]
    }

    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv('results/baseline_performance.csv', index=False)
    logging.info('Baseline performance results saved to results/baseline_performance.csv')


if __name__ == '__main__':
    config = Config()
    evaluate_baselines(config)
