# Quick Start Guide

Get started with ConvE training in 5 minutes!

## Prerequisites

```bash
# Install Python dependencies
pip install -r requirements.txt
```

## Quick Start with Pipeline Script

The easiest way to run the complete pipeline:

```bash
# Run the complete pipeline (preprocessing + training + evaluation)
./run_pipeline.sh <path_to_your_data> <output_directory>
```

**Example:**
```bash
./run_pipeline.sh ./data/raw ./output
```

Your data directory should contain:
- `train.tsv` - Training triples
- `valid.tsv` - Validation triples
- `test.tsv` - Test triples
- `node_dict` - Entity to index mapping
- `rel_dict` - Relation to index mapping
- `edge_map.json` - (Optional) Detailed predicate mapping

## Manual Step-by-Step

### 1. Preprocess Data

```bash
python preprocess.py \
  --train data/raw/train.tsv \
  --valid data/raw/valid.tsv \
  --test data/raw/test.tsv \
  --node-dict data/raw/node_dict \
  --rel-dict data/raw/rel_dict \
  --output-dir data/processed
```

### 2. Train Model

```bash
python train.py \
  --train data/processed/train.txt \
  --valid data/processed/valid.txt \
  --test data/processed/test.txt \
  --entity-to-id data/processed/train_entity_to_id.tsv \
  --relation-to-id data/processed/train_relation_to_id.tsv \
  --output-dir output/model \
  --num-epochs 100 \
  --batch-size 256
```

### 3. Make Predictions

```bash
# Single query
python predict.py \
  --model-dir output/model \
  --query "CHEBI:17833" "predicate:0" \
  --top-k 10

# Batch queries from file
python predict.py \
  --model-dir output/model \
  --query-file queries.txt \
  --output predictions.json
```

### 4. Run Examples

```bash
python example.py output/model data/processed/train.txt data/processed/test.txt
```

## Expected Results

After training, you'll find:

```
output/
├── processed/                    # Preprocessed data
│   ├── train.txt
│   ├── valid.txt
│   ├── test.txt
│   ├── train_entity_to_id.tsv
│   └── train_relation_to_id.tsv
└── model/                        # Trained model and results
    ├── trained_model.pkl         # Trained model
    ├── config.json               # Configuration
    ├── test_results.json         # Detailed results
    ├── test_results.csv          # Results in CSV
    └── losses.tsv                # Training losses
```

## Understanding Results

### Key Metrics

- **MRR (Mean Reciprocal Rank)**: Higher is better (0-1 range)
  - 0.3+ is considered good for most datasets
- **Hits@10**: Percentage of correct predictions in top-10
  - 50%+ is considered good

### Example Output

```
Evaluation Results:
  Mean Rank: 45.23
  Mean Reciprocal Rank: 0.3456
  Hits@1: 0.2134
  Hits@3: 0.2876
  Hits@10: 0.4523
```

## Common Issues

### Out of Memory

```bash
# Reduce batch size
python train.py ... --batch-size 128

# Use CPU
python train.py ... --no-gpu
```

### Slow Training

```bash
# Use GPU if available (default)
# Check GPU usage: nvidia-smi

# Reduce epochs for quick testing
python train.py ... --num-epochs 10
```

### Unknown Entities/Relations

Ensure your test data only contains entities and relations that appear in the training data. The preprocessing script automatically filters unknown entities.

### Label Smoothing Error

If you see:
```
pykeen.losses.UnsupportedLabelSmoothingError: MarginRankingLoss does not support label smoothing.
```

**Solution:** Update to the latest version (this is fixed). Or disable label smoothing:
```bash
python train.py ... --label-smoothing 0
```

**Note:** Label smoothing (default: 0.1) is recommended as it improves generalization.

## Next Steps

1. **Tune Hyperparameters**: Adjust embedding dimensions, dropout rates, learning rate
2. **Analyze Results**: Review per-triple predictions in `test_results.csv`
3. **TracIn Analysis**: Understand which training examples influence predictions
4. **Cross-Method Comparison**: Compare results with other models using the same indices

## Need Help?

See [README.md](README.md) for detailed documentation.

## Quick Reference

| Task | Command |
|------|---------|
| Full pipeline | `./run_pipeline.sh <data> <output>` |
| Preprocess | `python preprocess.py --train ... --valid ... --test ...` |
| Train | `python train.py --train ... --valid ... --test ...` |
| Predict | `python predict.py --model-dir ... --query <h> <r>` |
| Examples | `python example.py <model_dir>` |
