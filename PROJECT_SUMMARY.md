# ConvE with PyKEEN - Project Summary

This repository contains a complete implementation of ConvE using PyKEEN with custom evaluation and TracIn support.

## What Was Created

### Core Implementation Files

1. **preprocess.py** - Data preprocessing
   - Converts custom triple format to PyKEEN format
   - Preserves exact entity/relation indices from node_dict and rel_dict
   - Handles train/valid/test splits
   - Validates data integrity

2. **model.py** - ConvE model wrapper
   - Extends PyKEEN's ConvE implementation
   - Adds gradient tracking for TracIn
   - Provides factory functions for model creation
   - Model loading utilities

3. **train.py** - Training script
   - Complete training pipeline using PyKEEN
   - Configurable hyperparameters
   - Early stopping support
   - Automatic evaluation on test set
   - Saves detailed results and configuration

4. **evaluate.py** - Custom evaluation module
   - Per-triple predictions and rankings
   - Detailed metrics (MR, MRR, Hits@k)
   - Filtered evaluation (removes known triples)
   - Top-k predictions for each query
   - Exports results in JSON and CSV formats

5. **tracin.py** - TracIn implementation
   - Computes training data influence on predictions
   - Per-sample gradient tracking
   - Self-influence analysis for training data
   - Full test set analysis support

### Utility Scripts

6. **predict.py** - Prediction script
   - Make predictions for new queries
   - Single query or batch mode
   - Exports predictions in JSON/CSV

7. **example.py** - Comprehensive examples
   - Model evaluation examples
   - Prediction examples
   - TracIn analysis examples
   - Embedding analysis examples

8. **run_tracin.py** - Standalone TracIn analysis
   - Three modes: test, self, single
   - Configurable analysis parameters
   - Progress tracking

9. **run_pipeline.sh** - Complete pipeline script
   - One-command execution
   - Preprocessing + training + evaluation
   - Error handling and progress reporting

### Documentation

10. **README.md** - Comprehensive documentation
    - Installation instructions
    - Usage examples for all features
    - Configuration options
    - Troubleshooting guide
    - Citations

11. **QUICKSTART.md** - Quick start guide
    - 5-minute getting started
    - Common commands
    - Expected results
    - Quick reference table

12. **requirements.txt** - Python dependencies
    - PyTorch, PyKEEN, NumPy, Pandas, etc.

13. **.gitignore** - Git ignore rules
    - Python, data, model, and OS files

## Key Features

### 1. Data Compatibility
- Preserves your exact node_dict and rel_dict indices
- Ensures cross-method comparison capability
- Handles missing entities/relations gracefully

### 2. Comprehensive Evaluation
- Standard metrics: MR, MRR, Hits@k
- Per-triple detailed results
- Top-k predictions for each query
- Filtered evaluation (best practice)

### 3. TracIn Support
- Influence analysis for training data
- Self-influence computation
- Per-triple or batch analysis
- Computationally optimized

### 4. Easy to Use
- One-line pipeline script
- Sensible defaults
- Clear error messages
- Comprehensive examples

## File Structure

```
conve_pykeen/
├── README.md                 # Main documentation
├── QUICKSTART.md            # Quick start guide
├── PROJECT_SUMMARY.md       # This file
├── requirements.txt         # Dependencies
├── .gitignore              # Git ignore rules
│
├── preprocess.py           # Data preprocessing
├── model.py                # ConvE model
├── train.py                # Training script
├── evaluate.py             # Evaluation module
├── tracin.py               # TracIn implementation
│
├── predict.py              # Prediction utility
├── example.py              # Usage examples
├── run_tracin.py           # TracIn utility
└── run_pipeline.sh         # Complete pipeline
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run complete pipeline
./run_pipeline.sh <data_dir> <output_dir>
```

## Example Usage

### Training
```bash
python train.py \
  --train data/processed/train.txt \
  --valid data/processed/valid.txt \
  --test data/processed/test.txt \
  --entity-to-id data/processed/train_entity_to_id.tsv \
  --relation-to-id data/processed/train_relation_to_id.tsv \
  --output-dir output/model
```

### Prediction
```bash
python predict.py \
  --model-dir output/model \
  --query "CHEBI:17833" "predicate:0"
```

### TracIn Analysis
```bash
python run_tracin.py \
  --model-dir output/model \
  --train data/processed/train.txt \
  --test data/processed/test.txt \
  --output tracin_results.json \
  --mode single \
  --test-indices 0 1 2
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- PyKEEN 1.10+
- NumPy, Pandas
- Optional: CUDA for GPU acceleration

## What Makes This Implementation Special

1. **PyKEEN Integration**: Leverages PyKEEN's robust training infrastructure
2. **Index Preservation**: Maintains your original entity/relation indices
3. **Detailed Evaluation**: Goes beyond standard metrics with per-triple analysis
4. **TracIn Support**: Unique capability to analyze training data influence
5. **Production Ready**: Error handling, logging, validation
6. **Well Documented**: Comprehensive docs with examples

## Performance Expectations

- **Training**: ~5-10 minutes per epoch on GPU (dataset dependent)
- **Evaluation**: ~1-2 minutes for 1000 test triples
- **TracIn**: ~30-60 seconds per test triple (very expensive)

## Next Steps

1. Read [QUICKSTART.md](QUICKSTART.md) for immediate usage
2. Review [README.md](README.md) for detailed documentation
3. Run [example.py](example.py) to see all features in action
4. Start with the pipeline script for your data

## Support

For issues or questions:
1. Check the troubleshooting section in README.md
2. Review the examples in example.py
3. Examine the code comments for implementation details

## License

This project is provided as-is for research purposes.
