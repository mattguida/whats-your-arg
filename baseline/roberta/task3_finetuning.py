#!/usr/bin/env python
# coding: utf-8
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, pipeline, BertTokenizer, LongformerTokenizerFast, 
    LongformerForSequenceClassification, Trainer, TrainingArguments, 
    LongformerConfig, RobertaTokenizer, RobertaForSequenceClassification, RobertaForQuestionAnswering, RobertaModel, RobertaTokenizer, RobertaConfig, AdamW)
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold, StratifiedShuffleSplit 
from huggingface_hub import login
from tqdm import tqdm
from collections import Counter
from rouge_score import rouge_scorer
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
from simpletransformers.question_answering import QuestionAnsweringModel, QuestionAnsweringArgs
warnings.filterwarnings('ignore')

with open('data/fine_tuning/task3_formatted.jsonl', 'r') as f:
    data = json.load(f)

train_args = {
    'overwrite_output_dir': True,
    "evaluate_during_training": True, 
    "max_seq_length": 512,
    "num_train_epochs": 10,
    "save_model_every_epoch": False,
    "save_eval_checkpoints": False,
    "n_best_size": 16,
    "train_batch_size": 16,
    "eval_batch_size": 16
}

rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)

train_data, test_data = train_test_split(data, test_size=0.2, shuffle=True, random_state=42)

rouge_scores = []

kf = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_inner_idx, val_inner_idx) in enumerate(kf.split(train_data)):
    print(f"\nTraining fold {fold + 1}/{kf.n_splits}...")

    train_fold = [train_data[i] for i in train_inner_idx]
    val_fold = [train_data[i] for i in val_inner_idx]

    model = QuestionAnsweringModel(
        "roberta",
        "roberta-base",
        args=train_args,
        device_map="auto"
    )
    
    model.train_model(train_fold, eval_data=val_fold)

    val_result, val_texts = model.eval_model(val_fold)
    print(f"Validation Results (Fold {fold + 1}):", val_result)

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for sample_id, sample_data in val_texts['similar_text'].items():
        true_answer = sample_data['truth']
        predicted_answer = sample_data['predicted']
        
        scores = rouge_scorer.score(true_answer, predicted_answer)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)

    for sample_id, sample_data in val_texts['correct_text'].items():
        true_answer = sample_data['truth']
        predicted_answer = sample_data['predicted']
        
        scores = rouge_scorer.score(true_answer, predicted_answer)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)

    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores)
    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)

    print(f"Validation ROUGE Scores (Fold {fold + 1}):")
    print(f"Average ROUGE-1: {avg_rouge1:.4f}")
    print(f"Average ROUGE-2: {avg_rouge2:.4f}")
    print(f"Average ROUGE-L: {avg_rougeL:.4f}")

    rouge_scores.append({
        'fold': fold,
        'rouge1': avg_rouge1,
        'rouge2': avg_rouge2,
        'rougeL': avg_rougeL
    })

avg_rouge1_all = sum(fold['rouge1'] for fold in rouge_scores) / len(rouge_scores)
avg_rouge2_all = sum(fold['rouge2'] for fold in rouge_scores) / len(rouge_scores)
avg_rougeL_all = sum(fold['rougeL'] for fold in rouge_scores) / len(rouge_scores)

print("\nAverage Validation ROUGE Scores across all folds:")
print(f"Average ROUGE-1: {avg_rouge1_all:.4f}")
print(f"Average ROUGE-2: {avg_rouge2_all:.4f}")
print(f"Average ROUGE-L: {avg_rougeL_all:.4f}")

print("\nEvaluating on held-out test set...")
final_result, final_texts = model.eval_model(test_data)
print("Final Test Set Results:", final_result)

test_rouge1_scores = []
test_rouge2_scores = []
test_rougeL_scores = []

for sample_id, sample_data in final_texts['similar_text'].items():
    true_answer = sample_data['truth']
    predicted_answer = sample_data['predicted']
    
    scores = rouge_scorer.score(true_answer, predicted_answer)
    test_rouge1_scores.append(scores['rouge1'].fmeasure)
    test_rouge2_scores.append(scores['rouge2'].fmeasure)
    test_rougeL_scores.append(scores['rougeL'].fmeasure)

for sample_id, sample_data in final_texts['correct_text'].items():
    true_answer = sample_data['truth']
    predicted_answer = sample_data['predicted']
    
    scores = rouge_scorer.score(true_answer, predicted_answer)
    test_rouge1_scores.append(scores['rouge1'].fmeasure)
    test_rouge2_scores.append(scores['rouge2'].fmeasure)
    test_rougeL_scores.append(scores['rougeL'].fmeasure)

final_rouge1 = sum(test_rouge1_scores) / len(test_rouge1_scores)
final_rouge2 = sum(test_rouge2_scores) / len(test_rouge2_scores)
final_rougeL = sum(test_rougeL_scores) / len(test_rougeL_scores)

print("\nFinal Test Set ROUGE Scores:")
print(f"ROUGE-1: {final_rouge1:.4f}")
print(f"ROUGE-2: {final_rouge2:.4f}")
print(f"ROUGE-L: {final_rougeL:.4f}")
