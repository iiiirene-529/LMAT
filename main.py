# main.py

import torch
from config import Config
from data_preprocessing import get_data_loaders
from model import LMAT_ND
from train import train
from utils import setup_logging, load_checkpoint
from evaluation import evaluate


def main():
    config = Config()
    setup_logging(config)
    logging.info('Starting LMAT-ND Training Pipeline')

    # Data Loaders
    train_loader, val_loader, test_loader = get_data_loaders(config)
    logging.info('Data loaders prepared.')

    # Model Initialization
    model = LMAT_ND(config)
    logging.info('Model initialized.')

    # Load Checkpoint if exists
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    model, optimizer, start_epoch, best_f1 = load_checkpoint(model, optimizer, config)

    # Training
    train(model, train_loader, val_loader, config)

    # Load Best Model for Testing
    model, optimizer, _, _ = load_checkpoint(model, optimizer, config)

    # Evaluation on Test Set
    test_metrics = evaluate(model, test_loader, config.DEVICE, config)
    logging.info(f'Test Metrics: {test_metrics}')


if __name__ == '__main__':
    main()
