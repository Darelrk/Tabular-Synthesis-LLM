#!/usr/bin/env python3
"""
Phase 2 v2: SDG (SDV) - Generate 3000 rows with balanced distribution
Approach: Train on full dataset, sample 3000 rows
"""

import pandas as pd
import numpy as np
import time
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

# Set random seeds
SEED = 42
np.random.seed(SEED)

print("=" * 60)
print("PHASE 2 v2: SDG - Generate 3000 Synthetic Rows")
print("=" * 60)

# Load original data
print("\n[1] Loading original Adult dataset...")
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
           'marital_status', 'occupation', 'relationship', 'race', 'sex',
           'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']

df_original = pd.read_csv('adult_census.csv', header=None, names=columns, skipinitialspace=True)
print(f"    Original data: {len(df_original)} rows")
print(f"    Income distribution:")
print(df_original['income'].value_counts(normalize=True).round(4))

# Preprocess
print("\n[2] Preprocessing data...")
df_clean = df_original.copy()
df_clean = df_clean.dropna()
print(f"    After cleaning: {len(df_clean)} rows")

# Create metadata
print("\n[3] Creating metadata...")
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df_clean)

# Detect categorical columns automatically
for col in df_clean.columns:
    if df_clean[col].dtype == 'object':
        metadata.update_column(column_name=col, sdtype='categorical')
    else:
        metadata.update_column(column_name=col, sdtype='numerical')

print("    Metadata created successfully")

# Train SDG on full dataset
print("\n[4] Training SDG on full dataset (this may take 2-3 minutes)...")
start_time = time.time()

synthesizer = CTGANSynthesizer(metadata, epochs=300, verbose=True)
synthesizer.fit(df_clean)

train_time = time.time() - start_time
print(f"\n    Training completed in {train_time:.2f} seconds")

# Generate 3000 rows
print("\n[5] Generating 3000 synthetic rows...")
start_time = time.time()
synthetic_data = synthesizer.sample(num_rows=3000)
gen_time = time.time() - start_time

print(f"    Generated {len(synthetic_data)} rows in {gen_time:.2f} seconds")

# Verify distribution
print("\n[6] Verifying income distribution...")
print("Original distribution:")
orig_dist = df_clean['income'].value_counts(normalize=True)
print(orig_dist.round(4))

print("\nSynthetic distribution:")
syn_dist = synthetic_data['income'].value_counts(normalize=True)
print(syn_dist.round(4))

# Calculate difference
print("\nDistribution difference:")
for income_val in orig_dist.index:
    diff = abs(orig_dist[income_val] - syn_dist.get(income_val, 0))
    print(f"    {income_val}: {diff:.4f} ({diff*100:.2f}%)")

# Save
output_file = 'synthetic_sdg_3k.csv'
synthetic_data.to_csv(output_file, index=False)
print(f"\n[7] Saved to: {output_file}")

# Sample preview
print("\n[8] Sample of generated data:")
print(synthetic_data.head(10).to_string())

print("\n" + "=" * 60)
print("DONE! SDG 3000 rows generated successfully")
print("=" * 60)
