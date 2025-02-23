# config.py

import os

class Config:
    # Paths
    DATA_PATH = os.path.join('data', 'raw')
    PROCESSED_DATA_PATH = os.path.join('data', 'processed')
    MODEL_SAVE_PATH = os.path.join('models')
    LOG_PATH = os.path.join('logs', 'training.log')
    TOKENIZER_PATH = 'facebook/llama-7b'  # Pretrained Llama-7B tokenizer

    # Data Preprocessing
    MAX_SEQ_LENGTH = 512
    BATCH_SIZE = 8
    NUM_WORKERS = 4

    # Model Hyperparameters
    META_ATTENTION_DIM = 256
    DYNAMIC_MULTI_HEADS = 12
    DYNAMIC_HEAD_DIM = 64
    DUAL_CLASSIFIER_DIM = 128

    # Training Settings
    NUM_EPOCHS = 10
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 1e-2
    REGULARIZATION_COEFFICIENT = 1e-4
    GRADIENT_CLIP = 1.0

    # Evaluation Metrics
    METRICS = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']

    # Other Settings
    SEED = 42
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
