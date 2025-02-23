# ablation_study.py

import torch
import torch.nn as nn
from transformers import LlamaModel
from config import Config
from data_preprocessing import get_data_loaders
from evaluation import evaluate
from utils import setup_logging
import logging

from model import LMAT_ND, MetaAttention, DualClassificationLayer


class LMAT_ND_Variant(nn.Module):
    def __init__(self, config, remove_meta_attention=False, remove_dual_classification=False):
        super(LMAT_ND_Variant, self).__init__()
        self.llama = LlamaModel.from_pretrained(config.TOKENIZER_PATH)
        self.remove_meta_attention = remove_meta_attention
        self.remove_dual_classification = remove_dual_classification

        if not self.remove_meta_attention:
            self.meta_attention = MetaAttention(config.META_ATTENTION_DIM)
        if not self.remove_dual_classification:
            self.dual_classifier = DualClassificationLayer(
                embed_dim=self.llama.config.hidden_size,
                hidden_dim=config.DUAL_CLASSIFIER_DIM
            )

        self.dynamic_mha = nn.Identity()
        # Using Identity for simplicity; you can implement DynamicMultiHeadAttention if needed
        self.dropout = nn.Dropout(0.1)
        if self.remove_dual_classification:
            self.classifier = nn.Sequential(
                nn.Linear(self.llama.config.hidden_size, 128),
                nn.ReLU(),
                nn.Linear(128, 2)
            )

    def forward(self, input_ids, attention_mask):
        outputs = self.llama(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch, seq, hidden_size)

        if not self.remove_meta_attention:
            # Simulate attention matrix
            A = torch.softmax(torch.randn(hidden_states.size(0), self.llama.config.num_attention_heads, hidden_states.size(1), hidden_states.size(1)), dim=-1).to(hidden_states.device)
            C = hidden_states.mean(dim=1)  # (batch, hidden_size)
            A_meta = self.meta_attention(A, C)
            # Normally, A_meta would be used to adjust attention, but for simplicity, we'll skip further steps
            H = hidden_states.mean(dim=1)
        else:
            H = hidden_states.mean(dim=1)

        H = self.dynamic_mha(H)
        H = self.dropout(H)

        if not self.remove_dual_classification:
            y1, y2 = self.dual_classifier(H)
            return y1, y2
        else:
            logits = self.classifier(H)
            return logits


def conduct_ablation_study(config):
    setup_logging(config)
    logging.info('Conducting Ablation Study')

    _, _, test_loader = get_data_loaders(config)

    device = config.DEVICE

    # Define model variants
    variants = {
        'full_model': {'remove_meta_attention': False, 'remove_dual_classification': False},
        'without_meta_attention': {'remove_meta_attention': True, 'remove_dual_classification': False},
        'without_dual_classification': {'remove_meta_attention': False, 'remove_dual_classification': True},
        'without_both_components': {'remove_meta_attention': True, 'remove_dual_classification': True}
    }

    results = []

    for variant_name, params in variants.items():
        logging.info(f'Evaluating {variant_name}')
        model = LMAT_ND_Variant(config, **params).to(device)

        # Load pretrained weights if available
        # Example:
        # model.load_state_dict(torch.load(f'path_to_{variant_name}.pth'))

        # Define Loss Function
        criterion = nn.CrossEntropyLoss()

        # Evaluate
        if not params['remove_dual_classification']:
            outputs = model(input_ids=torch.randint(0, 1000, (1, config.MAX_SEQ_LENGTH)).to(device),
                            attention_mask=torch.ones((1, config.MAX_SEQ_LENGTH)).to(device))
            # Placeholder to ensure proper forward
            pass

        metrics = evaluate(model, test_loader, device, config)
        logging.info(f'{variant_name} Metrics: {metrics}')

        results.append({
            'Model Variant': variant_name.replace('_', ' ').title(),
            'Accuracy (%)': metrics['accuracy'] * 100,
            'AUC': metrics['roc_auc'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1']
        })

    # Convert results to DataFrame and save
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv('results/ablation_study_results.csv', index=False)
    logging.info('Ablation study results saved to results/ablation_study_results.csv')


if __name__ == '__main__':
    config = Config()
    conduct_ablation_study(config)
