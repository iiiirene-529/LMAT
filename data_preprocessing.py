# data_preprocessing.py

import os
import re
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import LlamaTokenizer
from config import Config
import pandas as pd


class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze()  # (max_length)
        attention_mask = encoding['attention_mask'].squeeze()  # (max_length)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label
        }


def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def filter_content(df):
    # Assuming 'content' column contains the text
    # and 'type' column indicates 'news' or 'non-news'
    df_filtered = df[df['type'] == 'news']
    return df_filtered


def load_and_preprocess_data(filepath, tokenizer, max_length):
    df = pd.read_csv(filepath)
    df = filter_content(df)
    df['normalized'] = df['content'].apply(normalize_text)
    df = df.dropna(subset=['normalized'])
    texts = df['normalized'].tolist()
    labels = df['label'].apply(lambda x: 1 if x == 'AI-generated' else 0).tolist()
    return NewsDataset(texts, labels, tokenizer, max_length)


def get_data_loaders(config):
    tokenizer = LlamaTokenizer.from_pretrained(config.TOKENIZER_PATH)

    # Load training data
    train_dataset = load_and_preprocess_data(
        os.path.join(config.DATA_PATH, 'train.csv'),
        tokenizer,
        config.MAX_SEQ_LENGTH
    )

    # Load validation data
    val_dataset = load_and_preprocess_data(
        os.path.join(config.DATA_PATH, 'val.csv'),
        tokenizer,
        config.MAX_SEQ_LENGTH
    )

    # Load test data
    test_dataset = load_and_preprocess_data(
        os.path.join(config.DATA_PATH, 'test.csv'),
        tokenizer,
        config.MAX_SEQ_LENGTH
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )

    return train_loader, val_loader, test_loader
