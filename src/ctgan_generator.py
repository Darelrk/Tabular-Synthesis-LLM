#!/usr/bin/env python3
"""
CTGAN Generator - Generate synthetic tabular data using CTGAN.

Usage:
    python ctgan_generator.py
"""

import time
import numpy as np
import pandas as pd
from ctgan import CTGAN

SEED = 42
np.random.seed(SEED)

COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'sex',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
]

CATEGORICAL_COLUMNS = [
    'workclass', 'education', 'marital_status', 'occupation',
    'relationship', 'race', 'sex', 'native_country', 'income'
]


def load_data(filepath: str) -> pd.DataFrame:
    """Load and return original dataset."""
    return pd.read_csv(filepath, header=None, names=COLUMNS, skipinitialspace=True)


def train_ctgan(df: pd.DataFrame, epochs: int = 300) -> CTGAN:
    """Train CTGAN model on dataset."""
    model = CTGAN(epochs=epochs, verbose=True)
    model.fit(df, discrete_columns=CATEGORICAL_COLUMNS)
    return model


def generate_synthetic_data(model: CTGAN, n_samples: int = 3000) -> pd.DataFrame:
    """Generate synthetic data from trained model."""
    return model.sample(n_samples)


def main():
    print("Loading dataset...")
    df = load_data('adult_census.csv')
    df_clean = df.dropna()
    print(f"Loaded {len(df_clean)} rows after cleaning")
    
    print("Training CTGAN (this may take 20-30 minutes)...")
    start = time.time()
    model = train_ctgan(df_clean, epochs=300)
    print(f"Training completed in {time.time() - start:.1f} seconds")
    
    print("Generating 3000 synthetic rows...")
    start = time.time()
    synthetic = generate_synthetic_data(model, n_samples=3000)
    print(f"Generated in {time.time() - start:.1f} seconds")
    
    output_file = 'synthetic_ctgan_3k.csv'
    synthetic.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")
    
    print("\nIncome distribution comparison:")
    print("Original:", df_clean['income'].value_counts(normalize=True).round(4).to_dict())
    print("Synthetic:", synthetic['income'].value_counts(normalize=True).round(4).to_dict())


if __name__ == "__main__":
    main()
