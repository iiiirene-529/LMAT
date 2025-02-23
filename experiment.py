# experiment.py

import argparse
from config import Config
from baseline_models import evaluate_baselines
from ablation_study import conduct_ablation_study
from model import LMAT_ND
from data_preprocessing import get_data_loaders
from train import train
from evaluation import evaluate
from utils import load_checkpoint, setup_logging
import logging
import pandas as pd
import os


def train_lmat_nd(config):
    setup_logging(config)
    logging.info('Training LMAT-ND Model')

    train_loader, val_loader, test_loader = get_data_loaders(config)

    model = LMAT_ND(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    # Load checkpoint if exists
    model, optimizer, start_epoch, best_f1 = load_checkpoint(model, optimizer, config)

    # Train the model
    train(model, train_loader, val_loader, config)

    # Evaluate on Test Set
    model, _, _, _ = load_checkpoint(model, optimizer, config)
    test_metrics = evaluate(model, test_loader, config.DEVICE, config)
    logging.info(f'LMAT-ND Test Metrics: {test_metrics}')

    # Save LMAT-ND Test Metrics
    lmat_metrics = {
        'Model': 'LMAT-ND (full model)',
        'Accuracy (%)': test_metrics['accuracy'] * 100,
        'AUC': test_metrics['roc_auc'],
        'Recall': test_metrics['recall'],
        'F1-Score': test_metrics['f1']
    }
    df_lmat = pd.DataFrame([lmat_metrics])
    if not os.path.exists('results'):
        os.makedirs('results')
    df_lmat.to_csv('results/lmat_nd_performance.csv', index=False)
    logging.info('LMAT-ND performance results saved to results/lmat_nd_performance.csv')


def main():
    parser = argparse.ArgumentParser(description='LMAT-ND Experimentation Pipeline')
    parser.add_argument('--task', type=str, required=True, choices=['train_lmat', 'evaluate_baselines', 'ablation_study', 'plot_metrics'], help='Task to perform')

    args = parser.parse_args()
    config = Config()

    if args.task == 'train_lmat':
        train_lmat_nd(config)
    elif args.task == 'evaluate_baselines':
        evaluate_baselines(config)
    elif args.task == 'ablation_study':
        conduct_ablation_study(config)
    elif args.task == 'plot_metrics':
        from plot_metrics import plot_training_metrics, generate_performance_tables
        plot_training_metrics(config)
        generate_performance_tables()
    else:
        print('Invalid task selected.')


if __name__ == '__main__':
    main()
