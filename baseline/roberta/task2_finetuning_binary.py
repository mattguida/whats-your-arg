#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import (
    AutoModelForCausalLM, AutoTokenizer, pipeline, BertTokenizer, LongformerTokenizerFast, 
    LongformerForSequenceClassification, Trainer, TrainingArguments, 
    LongformerConfig, RobertaTokenizer, RobertaForSequenceClassification,
    RobertaModel, RobertaTokenizer, RobertaConfig, AdamW)
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold, StratifiedShuffleSplit 
from huggingface_hub import login
from tqdm import tqdm
from collections import Counter
import pandas as pd
import accelerate
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import nltk
import gc
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from collections import defaultdict
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, precision_recall_fscore_support, confusion_matrix
from transformers import EvalPrediction
import os
import re
import csv
import json
import joblib
import glob
import warnings
warnings.filterwarnings('ignore')



# In[3]:


tokenizer = RobertaTokenizer.from_pretrained('roberta-base')


# In[4]:

def prepare_data(data_path):
    df = pd.read_csv(data_path, index_col=0)
    label_mapping = {1: 0, 5: 1}
    df['label'] = df['label'].map(label_mapping)
    df['combined'] = df['text'] + ' [SEP] ' + df['argument']
    return df

def create_train_test_split(df, test_size=0.2):
    return train_test_split(
        df, 
        test_size=test_size, 
        stratify=df['label'],
        random_state=42
    )



# In[8]:


class ArgumentsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def get_trainer(model, train_dataset, val_dataset, output_dir):
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True
    )
    
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

def run_kfold_training(X, y, tokenizer, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_metrics = []
    confusion_matrices = []
    best_model = None
    best_f1 = 0
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nTraining Fold {fold + 1}")
        
        # Split and tokenize data
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        train_encodings = tokenizer(list(X_train), truncation=True, padding=True, return_tensors='pt')
        val_encodings = tokenizer(list(X_val), truncation=True, padding=True, return_tensors='pt')
        
        train_dataset = ArgumentsDataset(train_encodings, y_train.to_numpy())
        val_dataset = ArgumentsDataset(val_encodings, y_val.to_numpy())
        
        model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2, device_map="auto")
        trainer = get_trainer(model, train_dataset, val_dataset, f'./results/fold-{fold}')
        
        trainer.train()
        metrics = trainer.evaluate()
        all_metrics.append(metrics)
        
        if metrics['eval_f1'] > best_f1:
            best_f1 = metrics['eval_f1']
            best_model = model
            
        predictions = trainer.predict(val_dataset)
        preds = predictions.predictions.argmax(-1)
        fold_confusion_matrix = confusion_matrix(y_val.to_numpy(), preds)
        confusion_matrices.append(fold_confusion_matrix)
        
        print(f"Fold {fold + 1} metrics:", metrics)
        print(f"Fold {fold + 1} Confusion Matrix:\n", fold_confusion_matrix)
    
    return best_model, all_metrics, confusion_matrices

def evaluate_test_set(model, X_test, y_test, tokenizer):
    test_encodings = tokenizer(list(X_test), truncation=True, padding=True, return_tensors='pt')
    test_dataset = ArgumentsDataset(test_encodings, y_test.to_numpy())
    
    trainer = Trainer(model=model, compute_metrics=compute_metrics)
    test_results = trainer.predict(test_dataset)
    
    test_metrics = {
        'accuracy': test_results.metrics['test_accuracy'],
        'f1': test_results.metrics['test_f1'],
        'precision': test_results.metrics['test_precision'],
        'recall': test_results.metrics['test_recall']
    }
    
    test_preds = test_results.predictions.argmax(-1)
    test_confusion_matrix = confusion_matrix(y_test.to_numpy(), test_preds)
    
    return test_metrics, test_confusion_matrix

def main():

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    df = prepare_data('data/fine_tuning/task2_binary_finetune.csv')
    train_val_df, test_df = create_train_test_split(df)
    
    best_model, all_metrics, confusion_matrices = run_kfold_training(
        train_val_df['combined'], 
        train_val_df['label'],
        tokenizer
    )
    
    avg_metrics = {key: np.mean([fold_metric[key] for fold_metric in all_metrics]) 
                  for key in all_metrics[0].keys()}
    avg_confusion_matrix = np.mean(confusion_matrices, axis=0)
    
    print("\nAverage metrics across all folds:", avg_metrics)
    print("\nAverage confusion matrix across all folds:")
    print(avg_confusion_matrix)
    
    test_metrics, test_confusion_matrix = evaluate_test_set(
        best_model, 
        test_df['combined'], 
        test_df['label'],
        tokenizer
    )
    
    print("\nFinal Test Set Metrics:", test_metrics)
    print("\nTest Set Confusion Matrix:")
    print(test_confusion_matrix)

if __name__ == "__main__":
    main()