#!/usr/bin/env python3
"""
Phase 3 v2: ML Quality Testing - Compare 3000 rows synthetic data
"""

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
SEED = 42
np.random.seed(SEED)

print("=" * 70)
print("PHASE 3 v2: ML QUALITY TESTING (3000 Rows Comparison)")
print("=" * 70)

# Load all datasets
print("\n[1] Loading datasets...")
datasets = {}

# Original data
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
           'marital_status', 'occupation', 'relationship', 'race', 'sex',
           'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']

df_original = pd.read_csv('adult_census.csv', header=None, names=columns, skipinitialspace=True)
datasets['Original'] = df_original
print(f"    [OK] Original: {len(df_original)} rows")

# Load all synthetic datasets (old 30k and new 3k)
synthetic_files = [
    ('CTGAN-30k', 'synthetic_ctgan.csv'),
    ('CTGAN-3k', 'synthetic_ctgan_3k.csv'),
    ('SDG-30k', 'synthetic_sdg.csv'),
    ('SDG-3k', 'synthetic_sdg_3k.csv'),
    ('GReaT', 'synthetic_great.csv')
]

for name, file in synthetic_files:
    try:
        df = pd.read_csv(file, skipinitialspace=True)
        datasets[name] = df
        print(f"    [OK] {name}: {len(df)} rows")
    except FileNotFoundError:
        print(f"    [MISSING] {name}: {file}")

print(f"\n    Total datasets: {len(datasets)}")

# Preprocess function
def preprocess_data(df):
    """Preprocess data for ML"""
    df_processed = df.copy()
    df_processed = df_processed.dropna()
    
    le_dict = {}
    for col in df_processed.columns:
        if not pd.api.types.is_numeric_dtype(df_processed[col]):
            df_processed[col] = df_processed[col].astype(str).str.strip()
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
            le_dict[col] = le
    
    return df_processed, le_dict

# Test each dataset
print("\n[2] Training and evaluating models...")
print("-" * 70)

results = {}

for name, df in datasets.items():
    print(f"\n{'='*50}")
    print(f"Dataset: {name}")
    print(f"{'='*50}")
    
    df_processed, _ = preprocess_data(df)
    
    if len(df_processed) < 100:
        print(f"    [WARN] Skipping - too few samples ({len(df_processed)})")
        continue
    
    X = df_processed.drop('income', axis=1)
    y = df_processed['income']
    
    # Train/test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=SEED, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=SEED
        )
    
    print(f"    Training samples: {len(X_train)}")
    print(f"    Test samples: {len(X_test)}")
    
    # Train Random Forest
    start_time = time.time()
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=SEED,
        n_jobs=8
    )
    rf.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Predict
    y_pred = rf.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'train_time': train_time,
        'n_samples': len(df_processed)
    }
    
    results[name] = metrics
    
    print(f"    [DONE] Training completed in {train_time:.2f}s")
    print(f"    Accuracy: {metrics['accuracy']:.4f}")
    print(f"    F1-Score: {metrics['f1']:.4f}")

# Comparison Report
print("\n" + "=" * 70)
print("COMPARISON RESULTS - 3000 vs 30000 Rows")
print("=" * 70)

if results:
    comparison_df = pd.DataFrame(results).T
    comparison_df = comparison_df.round(4)
    
    print("\nDetailed Metrics:")
    print(comparison_df.to_string())
    
    # Compare 3k vs 30k
    if 'Original' in results:
        print("\n" + "-" * 70)
        print("Quality Score (Relative to Original)")
        print("-" * 70)
        
        original_metrics = results['Original']
        
        for name, metrics in results.items():
            if name != 'Original':
                acc_diff = (metrics['accuracy'] - original_metrics['accuracy']) * 100
                f1_diff = (metrics['f1'] - original_metrics['f1']) * 100
                
                print(f"\n{name}:")
                print(f"    Accuracy: {metrics['accuracy']:.4f} ({acc_diff:+.2f}%)")
                print(f"    F1-Score: {metrics['f1']:.4f} ({f1_diff:+.2f}%)")
                
                if abs(acc_diff) < 5 and abs(f1_diff) < 5:
                    print(f"    [GOOD] QUALITY: Good (within 5% of original)")
                elif abs(acc_diff) < 10 and abs(f1_diff) < 10:
                    print(f"    [WARN] QUALITY: Moderate (within 10% of original)")
                else:
                    print(f"    [POOR] QUALITY: Poor (>10% difference)")
    
    # Summary comparison CTGAN
    if 'CTGAN-30k' in results and 'CTGAN-3k' in results:
        print("\n" + "=" * 70)
        print("CTGAN COMPARISON: 30k vs 3k")
        print("=" * 70)
        ctgan_30k = results['CTGAN-30k']
        ctgan_3k = results['CTGAN-3k']
        
        print(f"CTGAN-30k: Accuracy={ctgan_30k['accuracy']:.4f}, F1={ctgan_30k['f1']:.4f}")
        print(f"CTGAN-3k:  Accuracy={ctgan_3k['accuracy']:.4f}, F1={ctgan_3k['f1']:.4f}")
        
        acc_diff = (ctgan_3k['accuracy'] - ctgan_30k['accuracy']) * 100
        f1_diff = (ctgan_3k['f1'] - ctgan_30k['f1']) * 100
        print(f"\nDifference (3k vs 30k):")
        print(f"    Accuracy: {acc_diff:+.2f}%")
        print(f"    F1-Score: {f1_diff:+.2f}%")
    
    # Summary comparison SDG
    if 'SDG-30k' in results and 'SDG-3k' in results:
        print("\n" + "=" * 70)
        print("SDG COMPARISON: 30k vs 3k")
        print("=" * 70)
        sdg_30k = results['SDG-30k']
        sdg_3k = results['SDG-3k']
        
        print(f"SDG-30k: Accuracy={sdg_30k['accuracy']:.4f}, F1={sdg_30k['f1']:.4f}")
        print(f"SDG-3k:  Accuracy={sdg_3k['accuracy']:.4f}, F1={sdg_3k['f1']:.4f}")
        
        acc_diff = (sdg_3k['accuracy'] - sdg_30k['accuracy']) * 100
        f1_diff = (sdg_3k['f1'] - sdg_30k['f1']) * 100
        print(f"\nDifference (3k vs 30k):")
        print(f"    Accuracy: {acc_diff:+.2f}%")
        print(f"    F1-Score: {f1_diff:+.2f}%")
    
    # Save report
    report_file = 'phase3_v2_ml_report.txt'
    with open(report_file, 'w') as f:
        f.write("PHASE 3 v2: ML QUALITY TESTING REPORT (3000 vs 30000 Rows)\n")
        f.write("=" * 70 + "\n\n")
        f.write("Summary:\n")
        f.write(comparison_df.to_string())
        
        if 'Original' in results:
            f.write("\n\nQuality Assessment (Relative to Original):\n")
            f.write("-" * 70 + "\n")
            for name, metrics in results.items():
                if name != 'Original':
                    acc_diff = (metrics['accuracy'] - original_metrics['accuracy']) * 100
                    f1_diff = (metrics['f1'] - original_metrics['f1']) * 100
                    f.write(f"\n{name}:\n")
                    f.write(f"  Accuracy Diff: {acc_diff:+.2f}%\n")
                    f.write(f"  F1-Score Diff: {f1_diff:+.2f}%\n")
    
    print(f"\n[DONE] Report saved to: {report_file}")

print("\n" + "=" * 70)
print("PHASE 3 v2 COMPLETE!")
print("=" * 70)
