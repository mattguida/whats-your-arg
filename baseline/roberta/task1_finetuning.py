
#!/usr/bin/env python
# coding: utf-8

# In[1]:

#!/usr/bin/env python
# coding: utf-8

from transformers import (
    RobertaTokenizer, RobertaForSequenceClassification, 
    Trainer, TrainingArguments
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('data/fine_tuning/task1_finetune_data.csv', index_col=0)
df['combined'] = df['text'] + ' [SEP] ' + df['argument']

# First, create a held-out test set (20% of data)
X = df['combined']
y = df['label']
np.random.seed(42)
test_size = 0.2
indices = np.random.permutation(len(X))
test_idx = indices[:int(test_size * len(X))]
train_val_idx = indices[int(test_size * len(X)):]

X_test = X.iloc[test_idx]
y_test = y.iloc[test_idx]
X_train_val = X.iloc[train_val_idx]
y_train_val = y.iloc[train_val_idx]

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

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Prepare test dataset (will be used after k-fold training)
test_encodings = tokenizer(list(X_test), truncation=True, padding=True, return_tensors='pt')
test_dataset = ArgumentsDataset(test_encodings, y_test.to_numpy())

# Now perform k-fold on the training+validation data
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

all_metrics = []
confusion_matrices = []
test_predictions_all = []
test_labels_all = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_val, y_train_val)):
    print(f"\nTraining Fold {fold + 1}")
    
    X_train, X_val = X_train_val.iloc[train_idx], X_train_val.iloc[val_idx]
    y_train, y_val = y_train_val.iloc[train_idx], y_train_val.iloc[val_idx]
    
    train_encodings = tokenizer(list(X_train), truncation=True, padding=True, return_tensors='pt')
    val_encodings = tokenizer(list(X_val), truncation=True, padding=True, return_tensors='pt')
    
    train_dataset = ArgumentsDataset(train_encodings, y_train.to_numpy())
    val_dataset = ArgumentsDataset(val_encodings, y_val.to_numpy())
    
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', device_map="auto")
    
    training_args = TrainingArguments(
        output_dir=f'./results/fold-{fold}',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f'./logs/fold-{fold}',
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    # Train the model
    trainer.train()
    
    # Evaluate on test set
    test_predictions = trainer.predict(test_dataset)
    test_preds = test_predictions.predictions.argmax(-1)
    test_labels = y_test.to_numpy()
    
    # Store predictions and labels for this fold
    test_predictions_all.append(test_preds)
    test_labels_all.append(test_labels)
    
    # Calculate metrics for this fold on test set
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, test_preds, average='macro')
    acc = accuracy_score(test_labels, test_preds)
    
    fold_metrics = {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
    all_metrics.append(fold_metrics)
    
    # Calculate confusion matrix for this fold
    fold_confusion_matrix = confusion_matrix(test_labels, test_preds)
    confusion_matrices.append(fold_confusion_matrix)
    
    # Print example correct predictions from test set
    correct_indices = np.where(test_preds == test_labels)[0]
    correct_examples = X_test.iloc[correct_indices][:5]
    print("\nSample correctly classified examples from test set:")
    for i, example in enumerate(correct_examples):
        print(f"{i + 1}. {example}")

# Calculate and print average metrics across all folds (on test set)
avg_metrics = {key: np.mean([fold_metric[key] for fold_metric in all_metrics]) for key in all_metrics[0].keys()}
print("\nAverage metrics across all folds (on test set):", avg_metrics)