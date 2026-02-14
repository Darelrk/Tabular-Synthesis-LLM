#!/usr/bin/env python3
"""
GReaT OpenRouter Integration
Subclass of be_great.GReaT that uses OpenRouter API instead of local HuggingFace models
"""

import warnings
import json
import typing as tp
import logging
import re
import random
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

# OpenAI client for OpenRouter
from openai import OpenAI

# Import original be_great utilities
from be_great.great import GReaT
from be_great.great_utils import (
    _array_to_dataframe,
    _get_column_distribution,
    _convert_text_to_tabular_data,
    _partial_df_to_prompts,
    bcolors,
)


class GReaT_OpenRouter(GReaT):
    """GReaT Class with OpenRouter API Integration

    This subclass overrides the original GReaT to use OpenRouter API (OpenAI-compatible)
    instead of loading local HuggingFace models. This allows using powerful LLMs like
    StepFun, GPT-4, Claude, etc. via API.

    Attributes:
        llm (str): Model identifier for OpenRouter (e.g., "stepfun/step-3.5-flash:free")
        api_key (str): OpenRouter API key
        base_url (str): OpenRouter API base URL
        client (OpenAI): OpenAI client instance
        columns (list): List of all features/columns of the tabular dataset
        num_cols (list): List of all numerical features/columns
    """

    def __init__(
        self,
        llm: str = "arcee-ai/trinity-large-preview:free",
        api_key: str = None,
        base_url: str = "https://openrouter.ai/api/v1",
        experiment_dir: str = "trainer_great_openrouter",
        temperature: float = 0.7,
        max_tokens: int = 500,
        float_precision: tp.Optional[int] = None,
        **kwargs
    ):
        """Initializes GReaT_OpenRouter.

        Args:
            llm: Model identifier for OpenRouter (e.g., "arcee-ai/trinity-large-preview:free")
            api_key: OpenRouter API key. If None, will try to get from OPENAI_API_KEY env var
            base_url: OpenRouter API base URL
            experiment_dir: Directory for saving metadata (not used for API models)
            temperature: Temperature for generation (0.0 to 1.0)
            max_tokens: Maximum tokens per generation
            float_precision: Number of decimal places for floating point numbers
        """
        # Get API key from parameter or environment
        import os
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError(
                "API key required. Provide api_key parameter or set OPENAI_API_KEY environment variable."
            )
        
        # Initialize OpenAI client for OpenRouter
        self.base_url = base_url
        self.client = OpenAI(
            base_url=base_url,
            api_key=self.api_key
        )
        
        # Store model identifier
        self.llm = llm
        self.experiment_dir = experiment_dir
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.float_precision = float_precision
        
        # Initialize metadata (will be populated in fit())
        self.columns = None
        self.num_cols = None
        self.conditional_col = None
        self.conditional_col_dist = None
        self.col_stats = None
        
        # Track training status
        self._is_fitted = False
        
        logging.info(f"GReaT_OpenRouter initialized with model: {llm}")

    def fit(
        self,
        data: tp.Union[pd.DataFrame, np.ndarray],
        column_names: tp.Optional[tp.List[str]] = None,
        conditional_col: tp.Optional[str] = None,
        **kwargs
    ):
        """Prepare GReaT_OpenRouter using tabular data.

        For API-based models, this skips actual training and only extracts
        column information and statistics from the data.

        Args:
            data: Pandas DataFrame or Numpy Array containing tabular data
            column_names: Feature names if data is Numpy Array
            conditional_col: Column to use as conditional feature

        Returns:
            self
        """
        df = _array_to_dataframe(data, columns=column_names)
        self._update_column_information(df)
        self._update_conditional_information(df, conditional_col)
        
        self._is_fitted = True
        logging.info(f"GReaT_OpenRouter fitted on {len(df)} rows with {len(self.columns)} columns")
        return self

    def sample(
        self,
        n_samples: int,
        start_col: tp.Optional[str] = "",
        start_col_dist: tp.Optional[tp.Union[dict, list]] = None,
        temperature: tp.Optional[float] = None,
        max_length: int = 100,
        device: str = "cuda",  # Ignored for API
        batch_size: int = 1,  # Rows per API call
        **kwargs
    ) -> pd.DataFrame:
        """Generate synthetic tabular data using OpenRouter API.

        Args:
            n_samples: Number of synthetic samples to generate
            start_col: Starting column (optional)
            start_col_dist: Distribution for starting column
            temperature: Temperature override (uses self.temperature if None)
            max_length: Ignored (for API compatibility)
            device: Ignored (API doesn't use local device)
            batch_size: Number of rows to generate per API call (1-10 recommended)

        Returns:
            pd.DataFrame: DataFrame containing n_samples rows of generated data
        """
        if not self._is_fitted:
            raise ValueError("Model has not been fitted yet. Please call fit() first.")
        
        temp = temperature if temperature is not None else self.temperature
        
        # Determine starting column
        if start_col and start_col_dist is None:
            raise ValueError(f"Start column {start_col} given but no distribution provided.")
        
        start_col = start_col if start_col else self.conditional_col
        start_col_dist = start_col_dist if start_col_dist else self.conditional_col_dist
        
        # Generate samples
        synthetic_data = []
        rows_generated = 0
        
        with tqdm(total=n_samples, desc="Generating samples") as pbar:
            while rows_generated < n_samples:
                # Determine batch size
                current_batch = min(batch_size, n_samples - rows_generated)
                
                try:
                    # Generate batch
                    batch_data = self._generate_batch(
                        n_rows=current_batch,
                        start_col=start_col,
                        start_col_dist=start_col_dist,
                        temperature=temp
                    )
                    
                    synthetic_data.extend(batch_data)
                    rows_generated += len(batch_data)
                    pbar.update(len(batch_data))
                    
                    # Rate limiting - sleep briefly between calls
                    time.sleep(0.5)
                    
                except Exception as e:
                    logging.error(f"Error generating batch: {e}")
                    # Continue with next batch
                    continue
        
        # Convert to DataFrame
        if synthetic_data:
            df = pd.DataFrame(synthetic_data[:n_samples])
            
            # Ensure columns match original order
            if self.columns:
                df = df[self.columns]
            
            # Convert numerical columns
            for col in self.num_cols:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except (ValueError, TypeError) as e:
                    logging.warning(f"Could not convert column {col} to numeric: {e}")
            
            return df
        else:
            logging.error("No samples were generated")
            return pd.DataFrame(columns=self.columns)

    def _generate_batch(
        self,
        n_rows: int,
        start_col: str,
        start_col_dist: tp.Union[dict, list],
        temperature: float
    ) -> tp.List[dict]:
        """Generate a batch of rows using OpenRouter API."""
        
        # Create prompt
        prompt = self._create_generation_prompt(
            n_rows=n_rows,
            start_col=start_col,
            start_col_dist=start_col_dist
        )
        
        # Call API
        try:
            response = self.client.chat.completions.create(
                model=self.llm,
                messages=[
                    {"role": "system", "content": "You are a tabular data generator. Generate realistic, diverse data rows in the exact format specified."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=self.max_tokens,
                n=1
            )
            
            # Parse response
            generated_text = response.choices[0].message.content
            return self._parse_generated_text(generated_text, n_rows)
            
        except Exception as e:
            logging.error(f"API call failed: {e}")
            raise

    def _create_generation_prompt(
        self,
        n_rows: int,
        start_col: str,
        start_col_dist: tp.Union[dict, list]
    ) -> str:
        """Create a prompt for the LLM to generate tabular data."""
        
        # Build column descriptions
        col_descriptions = []
        for col in self.columns:
            if col in self.num_cols:
                stats = self.col_stats.get(col, {})
                min_val = stats.get('min', 'unknown')
                max_val = stats.get('max', 'unknown')
                col_descriptions.append(f"{col} (numeric, range: {min_val}-{max_val})")
            else:
                stats = self.col_stats.get(col, {})
                categories = stats.get('categories', [])
                if len(categories) <= 5:
                    cat_str = ', '.join(categories)
                    col_descriptions.append(f"{col} (categorical: {cat_str})")
                else:
                    col_descriptions.append(f"{col} (categorical, {len(categories)} categories)")
        
        # Starting value sampling
        if isinstance(start_col_dist, dict):
            # Categorical
            values = list(start_col_dist.keys())
            weights = list(start_col_dist.values())
            start_value = random.choices(values, weights=weights, k=1)[0]
        elif isinstance(start_col_dist, list):
            # Continuous
            start_value = random.choice(start_col_dist)
        else:
            start_value = None
        
        prompt = f"""Generate {n_rows} realistic rows of tabular data.

Columns ({len(self.columns)}):
{chr(10).join(f'- {desc}' for desc in col_descriptions)}

Format each row EXACTLY as:
{'; '.join(f'{col} is [value]' for col in self.columns)}

Requirements:
- Generate {n_rows} complete rows
- Each row on a new line
- Use realistic, diverse values
- Match the data types and ranges

{f'Start each row with: {start_col} is {start_value}' if start_value else ''}

Generate {n_rows} rows now:
"""
        return prompt

    def _parse_generated_text(self, text: str, expected_rows: int) -> tp.List[dict]:
        """Parse generated text into list of dictionaries."""
        rows = []
        
        # Split by newlines and process each line
        lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
        
        for line in lines:
            if len(rows) >= expected_rows:
                break
                
            row_data = {}
            
            # Try to parse "column is value; column is value; ..." format
            # Split by semicolon
            parts = [p.strip() for p in line.split(';')]
            
            for part in parts:
                # Look for "column is value" pattern
                match = re.match(r'^(.+?)\s+is\s+(.+)$', part.strip())
                if match:
                    col_name = match.group(1).strip()
                    value = match.group(2).strip()
                    
                    # Clean up value
                    if value.lower() in ['nan', 'null', 'none', '']:
                        value = None
                    
                    row_data[col_name] = value
            
            # Only add if we got at least some columns
            if row_data:
                # Fill missing columns with None
                for col in self.columns:
                    if col not in row_data:
                        row_data[col] = None
                rows.append(row_data)
        
        return rows

    def _update_column_information(self, df: pd.DataFrame):
        """Update column information from DataFrame."""
        self.columns = df.columns.to_list()
        self.num_cols = df.select_dtypes(include=np.number).columns.to_list()
        
        # Compute per-column statistics
        self.col_stats = {}
        for col in self.columns:
            if col in self.num_cols:
                self.col_stats[col] = {
                    "type": "numeric",
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                }
            else:
                self.col_stats[col] = {
                    "type": "categorical",
                    "categories": df[col].dropna().unique().astype(str).tolist(),
                }

    def _update_conditional_information(
        self, df: pd.DataFrame, conditional_col: tp.Optional[str] = None
    ):
        """Update conditional column information."""
        assert conditional_col is None or isinstance(conditional_col, str), \
            f"The column name has to be a string and not {type(conditional_col)}"
        assert conditional_col is None or conditional_col in df.columns, \
            f"The column name {conditional_col} is not in the feature names"
        
        self.conditional_col = conditional_col if conditional_col else df.columns[-1]
        self.conditional_col_dist = _get_column_distribution(df, self.conditional_col)

    def save(self, path: str):
        """Save GReaT_OpenRouter metadata."""
        import fsspec
        
        fs = fsspec.filesystem(fsspec.utils.get_protocol(path))
        if fs.exists(path):
            warnings.warn(f"Directory {path} already exists and is overwritten now.")
        else:
            fs.mkdir(path)
        
        # Save attributes
        with fs.open(path + "/config.json", "w") as f:
            attributes = {
                "llm": self.llm,
                "base_url": self.base_url,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "float_precision": self.float_precision,
                "columns": self.columns,
                "num_cols": self.num_cols,
                "conditional_col": self.conditional_col,
                "conditional_col_dist": self.conditional_col_dist,
                "col_stats": self.col_stats,
                "_is_fitted": self._is_fitted,
            }
            json.dump(attributes, f)
        
        logging.info(f"GReaT_OpenRouter metadata saved to {path}")

    @classmethod
    def load_from_dir(cls, path: str, api_key: str = None):
        """Load GReaT_OpenRouter from directory."""
        import fsspec
        import os
        
        fs = fsspec.filesystem(fsspec.utils.get_protocol(path))
        assert fs.exists(path), f"Directory {path} does not exist."
        
        # Load attributes
        with fs.open(path + "/config.json", "r") as f:
            attributes = json.load(f)
        
        # Create new instance
        instance = cls(
            llm=attributes["llm"],
            api_key=api_key or os.environ.get('OPENAI_API_KEY'),
            base_url=attributes.get("base_url", "https://openrouter.ai/api/v1"),
            temperature=attributes.get("temperature", 0.7),
            max_tokens=attributes.get("max_tokens", 500),
            float_precision=attributes.get("float_precision"),
        )
        
        # Restore attributes
        instance.columns = attributes.get("columns")
        instance.num_cols = attributes.get("num_cols")
        instance.conditional_col = attributes.get("conditional_col")
        instance.conditional_col_dist = attributes.get("conditional_col_dist")
        instance.col_stats = attributes.get("col_stats")
        instance._is_fitted = attributes.get("_is_fitted", False)
        
        return instance


# Convenience function for quick usage
def create_great_openrouter(
    api_key: str = None,
    model: str = "arcee-ai/trinity-large-preview:free",
    **kwargs
) -> GReaT_OpenRouter:
    """Create a GReaT_OpenRouter instance with specified parameters.
    
    Args:
        api_key: OpenRouter API key
        model: Model identifier
        **kwargs: Additional arguments for GReaT_OpenRouter
    
    Returns:
        GReaT_OpenRouter instance
    """
    return GReaT_OpenRouter(
        llm=model,
        api_key=api_key,
        **kwargs
    )
