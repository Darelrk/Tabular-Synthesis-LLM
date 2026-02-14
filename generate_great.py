#!/usr/bin/env python3
"""
Generate synthetic data using be_great with OpenRouter (Arcee model)

Usage:
    python generate_great.py
"""

import pandas as pd
import numpy as np
import time
import os

# Set API key directly (PowerShell compatible)
os.environ["OPENAI_API_KEY"] = "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"

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
import random
random.seed(SEED)

from great_openrouter import GReaT_OpenRouter

print("=" * 60)
print("Generating Synthetic Data with be_great + OpenRouter (Arcee)")
print("=" * 60)

# Load original data
print("\n[1] Loading original Adult dataset...")
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 
           'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 
           'hours_per_week', 'native_country', 'income']
df_original = pd.read_csv('adult_census.csv', header=None, names=columns)
print(f"    Original data: {df_original.shape[0]} rows, {df_original.shape[1]} columns")
print(f"    Columns: {list(df_original.columns)}")

# Initialize GReaT with OpenRouter (Arcee model)
print("\n[2] Initializing GReaT_OpenRouter with Arcee model...")
great = GReaT_OpenRouter(
    llm="arcee-ai/trinity-large-preview:free",
    temperature=0.7,
    max_tokens=600,
    experiment_dir="great_arcee_model"
)

# Fit on original data
print("\n[3] Fitting model on original data...")
great.fit(df_original)
print("    Model fitted successfully!")

# Generate synthetic data
target_rows = 3000
print(f"\n[4] Generating {target_rows} synthetic rows...")
print("    (Using batch_size=10 for faster generation)")

start_time = time.time()

synthetic_df = great.sample(
    n_samples=target_rows,
    batch_size=10,
    temperature=0.7
)

elapsed = time.time() - start_time
print(f"\n[5] Generation complete!")
print(f"    Time elapsed: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
print(f"    Rows generated: {len(synthetic_df)}")

# Save to CSV
output_file = 'synthetic_great.csv'
synthetic_df.to_csv(output_file, index=False)
print(f"\n[6] Saved to: {output_file}")

# Show sample
print("\n[7] Sample of generated data:")
print(synthetic_df.head(10).to_string())

print("\n" + "=" * 60)
print("DONE!")
print("=" * 60)
