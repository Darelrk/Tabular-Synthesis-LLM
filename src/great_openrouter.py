"""
GReaT OpenRouter Integration

Subclass of be_great.GReaT that uses OpenRouter API instead of local models.
"""

import os
import re
import json
import time
import random
import logging
import warnings
from typing import Optional, Union, List, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from be_great.great import GReaT
from be_great.great_utils import (
    _array_to_dataframe,
    _get_column_distribution,
    bcolors,
)


class GReaT_OpenRouter(GReaT):
    """
    GReaT with OpenRouter API integration.
    
    Uses OpenAI-compatible API to generate synthetic data with LLMs
    instead of loading local HuggingFace models.
    
    Args:
        llm: Model identifier (default: arcee-ai/trinity-large-preview:free)
        api_key: OpenRouter API key (or from OPENAI_API_KEY env var)
        base_url: API endpoint (default: https://openrouter.ai/api/v1)
        temperature: Generation temperature (0.0-1.0)
        max_tokens: Max tokens per request
    """
    
    def __init__(
        self,
        llm: str = "arcee-ai/trinity-large-preview:free",
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        temperature: float = 0.7,
        max_tokens: int = 500,
        float_precision: Optional[int] = None,
        **kwargs
    ):
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("API key required. Set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(base_url=base_url, api_key=self.api_key)
        self.llm = llm
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.float_precision = float_precision
        
        self.columns = None
        self.num_cols = None
        self.conditional_col = None
        self.conditional_col_dist = None
        self.col_stats = None
        self._is_fitted = False
        
        logging.info(f"Initialized with model: {llm}")
    
    def fit(self, data: Union[pd.DataFrame, np.ndarray], column_names: Optional[List[str]] = None, conditional_col: Optional[str] = None, **kwargs):
        """Extract metadata from training data."""
        df = _array_to_dataframe(data, columns=column_names)
        self._update_column_info(df)
        self._update_conditional_info(df, conditional_col)
        self._is_fitted = True
        logging.info(f"Fitted on {len(df)} rows, {len(self.columns)} columns")
        return self
    
    def sample(self, n_samples: int, batch_size: int = 10, **kwargs) -> pd.DataFrame:
        """Generate synthetic samples using API."""
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        synthetic_data = []
        rows_generated = 0
        
        with tqdm(total=n_samples, desc="Generating") as pbar:
            while rows_generated < n_samples:
                current_batch = min(batch_size, n_samples - rows_generated)
                
                try:
                    batch = self._generate_batch(current_batch)
                    synthetic_data.extend(batch)
                    rows_generated += len(batch)
                    pbar.update(len(batch))
                    time.sleep(0.5)
                except Exception as e:
                    logging.error(f"Batch error: {e}")
                    continue
        
        if not synthetic_data:
            return pd.DataFrame(columns=self.columns)
        
        df = pd.DataFrame(synthetic_data[:n_samples])
        if self.columns:
            df = df[self.columns]
        
        for col in self.num_cols:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except (ValueError, TypeError):
                pass
        
        return df
    
    def _generate_batch(self, n_rows: int) -> List[Dict]:
        """Generate a batch via API."""
        prompt = self._create_prompt(n_rows)
        
        response = self.client.chat.completions.create(
            model=self.llm,
            messages=[
                {"role": "system", "content": "Generate realistic tabular data rows."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        text = response.choices[0].message.content
        return self._parse_text(text, n_rows)
    
    def _create_prompt(self, n_rows: int) -> str:
        """Create generation prompt."""
        descriptions = []
        for col in self.columns:
            if col in self.num_cols:
                stats = self.col_stats.get(col, {})
                desc = f"{col} (numeric: {stats.get('min', 0):.0f}-{stats.get('max', 0):.0f})"
            else:
                cats = self.col_stats.get(col, {}).get('categories', [])
                desc = f"{col} ({len(cats)} categories)"
            descriptions.append(desc)
        
        format_str = '; '.join(f'{col} is [value]' for col in self.columns)
        
        return f"""Generate {n_rows} realistic rows.

Columns: {', '.join(descriptions)}

Format: {format_str}

Requirements:
- {n_rows} complete rows
- One row per line
- Realistic values

Generate now:"""
    
    def _parse_text(self, text: str, expected: int) -> List[Dict]:
        """Parse API response."""
        rows = []
        lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
        
        for line in lines:
            if len(rows) >= expected:
                break
            
            row = {}
            for part in line.split(';'):
                match = re.match(r'^(.+?)\s+is\s+(.+)$', part.strip())
                if match:
                    col, val = match.group(1).strip(), match.group(2).strip()
                    if val.lower() in ['nan', 'null', 'none', '']:
                        val = None
                    row[col] = val
            
            if row:
                for col in self.columns:
                    if col not in row:
                        row[col] = None
                rows.append(row)
        
        return rows
    
    def _update_column_info(self, df: pd.DataFrame):
        """Update column statistics."""
        self.columns = df.columns.tolist()
        self.num_cols = df.select_dtypes(include=np.number).columns.tolist()
        self.col_stats = {}
        
        for col in self.columns:
            if col in self.num_cols:
                self.col_stats[col] = {
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                }
            else:
                self.col_stats[col] = {
                    'categories': df[col].dropna().unique().astype(str).tolist()
                }
    
    def _update_conditional_info(self, df: pd.DataFrame, conditional_col: Optional[str] = None):
        """Update conditional column info."""
        self.conditional_col = conditional_col or df.columns[-1]
        self.conditional_col_dist = _get_column_distribution(df, self.conditional_col)
