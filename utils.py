# utils.py

import torch
import os
import logging

def save_checkpoint(model, optimizer, epoch, best_f1, config):
    if not os.path.exists(config.MODEL_SAVE_PATH):
        os.makedirs(config.MODEL_SAVE_PATH)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_f1': best_f1
    }
    torch.save(checkpoint, os.path.join(config.MODEL_SAVE_PATH, 'best_model.pth'))

def load_checkpoint(model, optimizer, config):
    checkpoint_path = os.path.join(config.MODEL_SAVE_PATH, 'best_model.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        best_f1 = checkpoint['best_f1']
        logging.info(f'Loaded checkpoint from epoch {epoch} with F1-Score {best_f1}')
        return model, optimizer, epoch, best_f1
    else:
        logging.info('No checkpoint found, initializing from scratch.')
        return model, optimizer, 0, 0.0

def setup_logging(config):
    if not os.path.exists('logs'):
        os.makedirs('logs')
    logging.basicConfig(
        filename=config.LOG_PATH,
        filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.INFO
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
