#!/usr/bin/env python3
"""
SDG Generator - Generate synthetic tabular data using SDV/CTGANSynthesizer.

Usage:
    python sdg_generator.py
"""

import time
import numpy as np
import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

SEED = 42
np.random.seed(SEED)

COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'sex',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
]


def load_data(filepath: str) -> pd.DataFrame:
    """Load and return original dataset."""
    return pd.read_csv(filepath, header=None, names=COLUMNS, skipinitialspace=True)


def create_metadata(df: pd.DataFrame) -> SingleTableMetadata:
    """Create metadata for SDV synthesizer."""
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
    
    for col in df.columns:
        sdtype = 'categorical' if df[col].dtype == 'object' else 'numerical'
        metadata.update_column(column_name=col, sdtype=sdtype)
    
    return metadata


def train_synthesizer(df: pd.DataFrame, metadata: SingleTableMetadata, epochs: int = 300):
    """Train SDV synthesizer on dataset."""
    synthesizer = CTGANSynthesizer(metadata, epochs=epochs, verbose=True)
    synthesizer.fit(df)
    return synthesizer


def generate_synthetic_data(synthesizer, n_samples: int = 3000) -> pd.DataFrame:
    """Generate synthetic data from trained synthesizer."""
    return synthesizer.sample(num_rows=n_samples)


def main():
    print("Loading dataset...")
    df = load_data('adult_census.csv')
    df_clean = df.dropna()
    print(f"Loaded {len(df_clean)} rows after cleaning")
    
    print("Creating metadata...")
    metadata = create_metadata(df_clean)
    
    print("Training SDV synthesizer (this may take 20-30 minutes)...")
    start = time.time()
    synthesizer = train_synthesizer(df_clean, metadata, epochs=300)
    print(f"Training completed in {time.time() - start:.1f} seconds")
    
    print("Generating 3000 synthetic rows...")
    start = time.time()
    synthetic = generate_synthetic_data(synthesizer, n_samples=3000)
    print(f"Generated in {time.time() - start:.1f} seconds")
    
    output_file = 'synthetic_sdg_3k.csv'
    synthetic.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")
    
    print("\nIncome distribution comparison:")
    print("Original:", df_clean['income'].value_counts(normalize=True).round(4).to_dict())
    print("Synthetic:", synthetic['income'].value_counts(normalize=True).round(4).to_dict())


if __name__ == "__main__":
    main()
