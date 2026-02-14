# Methodology

## Dataset

**Adult Census Income** from UCI Machine Learning Repository
- 32,561 rows, 15 features
- Mix of numerical and categorical data
- Binary classification target (income <=50K vs >50K)

## Experimental Design

### Training Strategy
All models trained on full dataset (32k rows) to learn complete distribution.

### Generation
Each framework generates 3,000 synthetic samples:
- GReaT: 3,000 rows (via OpenRouter API)
- CTGAN: 3,000 rows
- SDV: 3,000 rows

### Evaluation
XGBoost Classifier (100 trees, max_depth=6):
1. Train on synthetic data
2. Test on held-out real data (20% split)
3. Compare accuracy, F1-score, AUC-ROC, and other metrics

## Why This Design?

**Fair Comparison**: All models trained on same data, evaluated on same test set.

**Real-world Scenario**: Tests if synthetic data can replace real data for ML training.

**Quality Metric**: ML utility more important than statistical similarity.

## Key Decisions

1. **3K not 30K**: Smaller balanced dataset > larger imbalanced
2. **80/20 Split**: Standard ML evaluation practice
3. **Random Forest**: Robust baseline, good for tabular data
4. **Stratified Split**: Maintains class distribution
