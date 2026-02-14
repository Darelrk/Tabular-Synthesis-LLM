#!/usr/bin/env python3
"""
Evaluate synthetic data quality using Random Forest classifier.

Usage:
    python evaluate.py
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

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
    
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype(str).str.strip()
            df[col] = LabelEncoder().fit_transform(df[col])
    
    return df


def evaluate_dataset(name, df):
    """Train and evaluate Random Forest on dataset."""
    df = preprocess(df)
    
    if len(df) < 100:
        return None
    
    X = df.drop('income', axis=1)
    y = df['income']
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=SEED, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=SEED
        )
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=SEED, n_jobs=8)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'samples': len(df)
    }


def print_results(results, baseline_accuracy):
    """Print comparison results."""
    print("\nResults:")
    print("-" * 60)
    print(f"{'Dataset':<15} {'Accuracy':<12} {'F1-Score':<12} {'vs Original':<12}")
    print("-" * 60)
    
    for name, metrics in results.items():
        if metrics:
            diff = (metrics['accuracy'] - baseline_accuracy) * 100
            diff_str = f"{diff:+.2f}%"
            print(f"{name:<15} {metrics['accuracy']:<12.4f} {metrics['f1']:<12.4f} {diff_str:<12}")


def main():
    print("Loading datasets...")
    datasets = load_datasets()
    
    print("Evaluating...")
    results = {}
    
    for name, df in datasets.items():
        print(f"  {name}...", end=" ")
        metrics = evaluate_dataset(name, df)
        if metrics:
            results[name] = metrics
            print(f"Accuracy: {metrics['accuracy']:.4f}")
        else:
            print("Skipped")
    
    baseline = results.get('Original', {}).get('accuracy', 0)
    print_results(results, baseline)
    
    print("\nConclusion:")
    if 'GReaT' in results:
        great_acc = results['GReaT']['accuracy']
        diff = (great_acc - baseline) * 100
        print(f"GReaT achieves {great_acc:.2%} accuracy ({diff:+.2f}% vs original)")


if __name__ == "__main__":
    main()
