#!/usr/bin/env python3
"""
Evaluate synthetic data quality using XGBoost with comprehensive metrics.

Metrics: Accuracy, Precision, Recall, F1, AUC-ROC, Log Loss
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss
)
import xgboost as xgb

warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'sex',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
]


def load_datasets():
    """Load original and synthetic datasets."""
    datasets = {}
    
    df_original = pd.read_csv('adult_census.csv', header=None, names=COLUMNS, skipinitialspace=True)
    datasets['Original'] = df_original
    
    files = [
        ('GReaT', 'synthetic_great.csv'),
        ('CTGAN-3k', 'synthetic_ctgan_3k.csv'),
        ('SDG-3k', 'synthetic_sdg_3k.csv')
    ]
    
    for name, filepath in files:
        try:
            datasets[name] = pd.read_csv(filepath, skipinitialspace=True)
        except FileNotFoundError:
            print(f"Warning: {filepath} not found")
    
    return datasets


def preprocess(df):
    """Preprocess data for ML."""
    df = df.dropna()
    
    le_dict = {}
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype(str).str.strip()
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            le_dict[col] = le
    
    return df, le_dict


def evaluate_dataset(name, df):
    """Train and evaluate XGBoost on dataset."""
    df, le_dict = preprocess(df)
    
    if len(df) < 100:
        return None
    
    X = df.drop('income', axis=1)
    y = df['income']
    
    # Get unique classes for log_loss
    classes = np.unique(y)
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=SEED, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=SEED
        )
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=SEED,
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=8
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted'),
        'samples': len(df)
    }
    
    # AUC-ROC (only for binary classification)
    if len(classes) == 2:
        # Get the index of the positive class (class with higher label value)
        positive_class_idx = np.where(model.classes_ == max(model.classes_))[0][0]
        metrics['auc_roc'] = roc_auc_score(y_test, y_pred_proba[:, positive_class_idx])
        # Log loss with explicit labels
        metrics['log_loss'] = log_loss(y_test, y_pred_proba, labels=classes)
    else:
        metrics['auc_roc'] = 0.5
        metrics['log_loss'] = log_loss(y_test, y_pred_proba, labels=classes)
    
    return metrics


def print_results(results):
    """Print comprehensive comparison results."""
    print("\n" + "="*90)
    print("COMPREHENSIVE EVALUATION RESULTS (XGBoost)")
    print("="*90)
    
    print(f"\n{'Dataset':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AUC-ROC':<10} {'LogLoss':<10}")
    print("-"*90)
    
    baseline = results.get('Original', {})
    
    for name, metrics in results.items():
        if metrics:
            print(f"{name:<15} "
                  f"{metrics['accuracy']:<10.4f} "
                  f"{metrics['precision']:<10.4f} "
                  f"{metrics['recall']:<10.4f} "
                  f"{metrics['f1']:<10.4f} "
                  f"{metrics['auc_roc']:<10.4f} "
                  f"{metrics['log_loss']:<10.4f}")
    
    print("\n" + "="*90)
    print("COMPARISON VS ORIGINAL")
    print("="*90)
    
    if baseline:
        print(f"\n{'Dataset':<15} {'Accuracy':<12} {'F1':<12} {'AUC-ROC':<12}")
        print("-"*60)
        
        for name, metrics in results.items():
            if name != 'Original' and metrics:
                acc_diff = (metrics['accuracy'] - baseline['accuracy']) * 100
                f1_diff = (metrics['f1'] - baseline['f1']) * 100
                auc_diff = (metrics['auc_roc'] - baseline['auc_roc']) * 100
                
                print(f"{name:<15} "
                      f"{metrics['accuracy']:.4f} ({acc_diff:+.2f}%)  "
                      f"{metrics['f1']:.4f} ({f1_diff:+.2f}%)  "
                      f"{metrics['auc_roc']:.4f} ({auc_diff:+.2f}%)")


def main():
    print("Loading datasets...")
    datasets = load_datasets()
    
    print("Evaluating with XGBoost...")
    results = {}
    
    for name, df in datasets.items():
        print(f"  Evaluating {name}...", end=" ")
        metrics = evaluate_dataset(name, df)
        if metrics:
            results[name] = metrics
            print(f"Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
        else:
            print("Skipped")
    
    print_results(results)
    
    print("\n" + "="*90)
    print("EVALUATION COMPLETE")
    print("="*90)


if __name__ == "__main__":
    main()
