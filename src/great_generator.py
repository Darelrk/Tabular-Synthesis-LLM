#!/usr/bin/env python3
"""
GReaT Generator - Generate synthetic tabular data using LLM via OpenRouter.

Usage:
    export OPENAI_API_KEY="your-key-here"
    python great_generator.py
"""

import os
import time
import random
import numpy as np
import pandas as pd
from great_openrouter import GReaT_OpenRouter

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'sex',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
]


def load_data(filepath: str) -> pd.DataFrame:
    """Load and return original dataset."""
    return pd.read_csv(filepath, header=None, names=COLUMNS, skipinitialspace=True)


def generate_synthetic_data(
    df: pd.DataFrame,
    n_samples: int = 3000,
    batch_size: int = 10
) -> pd.DataFrame:
    """Generate synthetic data using GReaT with OpenRouter."""
    synthesizer = GReaT_OpenRouter(
        llm="arcee-ai/trinity-large-preview:free",
        temperature=0.7,
        max_tokens=600
    )
    
    synthesizer.fit(df)
    return synthesizer.sample(n_samples=n_samples, batch_size=batch_size)


def main():
    print("Loading dataset...")
    df_original = load_data('adult_census.csv')
    print(f"Loaded {len(df_original)} rows with {len(df_original.columns)} columns")
    
    print("Generating synthetic data...")
    start = time.time()
    synthetic = generate_synthetic_data(df_original, n_samples=3000)
    elapsed = time.time() - start
    
    print(f"Generated {len(synthetic)} rows in {elapsed:.1f} seconds")
    
    output_file = 'synthetic_great.csv'
    synthetic.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")
    
    print("\nSample output:")
    print(synthetic.head())


if __name__ == "__main__":
    main()
