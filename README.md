# ConvE with PyKEEN

This repository provides a complete implementation of ConvE (Convolutional Knowledge Graph Embeddings) using PyKEEN, with support for:

- Custom data preprocessing from your existing triple format
- Detailed per-triple evaluation and predictions
- TracIn analysis for understanding training data influence
- Cross-method comparison using fixed entity/relation indices

## Features

- **PyKEEN Integration**: Leverages PyKEEN's robust training pipeline and optimizations
- **Custom Evaluation**: Get predictions and rankings for each test triple
- **TracIn Support**: Analyze which training examples influence specific predictions
- **Compatible Data Format**: Maintains your existing node_dict and rel_dict indices for cross-method comparison
- **Comprehensive Metrics**: MR, MRR, Hits@k with filtered evaluation

## Installation

### Local Installation

```bash
# Clone the repository
git clone <repository-url>
cd conve_pykeen

# Install dependencies
pip install -r requirements.txt
```

### Kubernetes Deployment

For running on Kubernetes with GPU support:

```bash
# Deploy pod with automatic setup
./k8s-helper.sh deploy

# Attach to pod
./k8s-helper.sh attach

# See K8S_DEPLOYMENT.md for detailed instructions
```

### Requirements

- Python 3.8+
- PyTorch 2.0+
- PyKEEN 1.10+
- CUDA (optional, for GPU acceleration)
- Kubernetes cluster (optional, for K8s deployment)

## Data Format

This implementation expects the following input format:

### Triple Files (TSV format)
```
CHEBI:68579     predicate:0     MONDO:0002251
CHEBI:17833     predicate:0     HP:0002153
CHEBI:74947     predicate:1     HP:0001288
```

### node_dict (Entity to Index Mapping)
```
CHEBI:17833     0
HP:0002153      1
CHEBI:74947     2
HP:0001288      3
```

### rel_dict (Relation to Index Mapping)
```
predicate:0     0
predicate:1     1
predicate:2     2
```

### edge_map.json (Optional - Detailed Predicate Mapping)
```json
{
  "{\"predicate\": \"biolink:contributes_to\", ...}": "predicate:0",
  "{\"predicate\": \"biolink:causes\", ...}": "predicate:1"
}
```

## Usage

### 1. Data Preprocessing

Convert your data to PyKEEN format while preserving the exact indices from node_dict and rel_dict:

```bash
python preprocess.py \
  --train /path/to/train.tsv \
  --valid /path/to/valid.tsv \
  --test /path/to/test.tsv \
  --node-dict /path/to/node_dict \
  --rel-dict /path/to/rel_dict \
  --edge-map /path/to/edge_map.json \
  --output-dir ./data/processed
```

**Options:**
- `--no-validate`: Skip validation of entities/relations (faster, but may include invalid triples)
  ```bash
  python preprocess.py ... --no-validate  # Skip validation checks
  ```

**What it does:**
- By default, validates all entities exist in node_dict and all relations exist in rel_dict
- Reports warnings for unknown entities/relations and filters them out
- Use `--no-validate` to skip these checks if your data is already validated

This will create:
- `train.txt`, `valid.txt`, `test.txt` - Processed triples
- `train_entity_to_id.tsv` - Entity mapping
- `train_relation_to_id.tsv` - Relation mapping

### 2. Training

Train the ConvE model with default hyperparameters:

```bash
python train.py \
  --train ./data/processed/train.txt \
  --valid ./data/processed/valid.txt \
  --test ./data/processed/test.txt \
  --entity-to-id ./data/processed/train_entity_to_id.tsv \
  --relation-to-id ./data/processed/train_relation_to_id.tsv \
  --output-dir ./output/conve_model
```

#### Training Options

**Model Hyperparameters:**
```bash
--embedding-dim 200              # Embedding dimension (must equal height * width)
--embedding-height 10            # Height of reshaped embedding
--embedding-width 20             # Width of reshaped embedding
--output-channels 32             # Number of convolutional filters
--kernel-height 3                # Convolution kernel height
--kernel-width 3                 # Convolution kernel width
--input-dropout 0.2              # Input dropout rate
--feature-map-dropout 0.2        # Feature map dropout rate
--output-dropout 0.3             # Output layer dropout rate
```

**Training Hyperparameters:**
```bash
--num-epochs 100                 # Number of training epochs
--batch-size 256                 # Training batch size
--learning-rate 0.001            # Learning rate (Adam optimizer)
--label-smoothing 0.1            # Label smoothing parameter (0.0 to disable)
--patience 10                    # Early stopping patience
--no-early-stopping              # Disable early stopping
```

**About Label Smoothing:**
Label smoothing is a regularization technique that prevents overconfident predictions:
- Default: `0.1` (10% smoothing) - recommended for most cases
- Softens hard labels: `1.0 → 0.9`, `0.0 → 0.1`
- Improves generalization and prevents overfitting
- Set to `0.0` to disable: `--label-smoothing 0`
- Uses `BCEWithLogitsLoss` which supports label smoothing

**Example - Disable label smoothing:**
```bash
python train.py ... --label-smoothing 0
```

**Other Options:**
```bash
--no-gpu                         # Use CPU instead of GPU
--random-seed 42                 # Random seed for reproducibility
--track-gradients                # Enable gradient tracking for TracIn
```

### 3. Evaluation

Evaluate a trained model and get detailed per-triple predictions:

```python
from pykeen.pipeline import PipelineResult
from evaluate import evaluate_model
from pykeen.triples import TriplesFactory

# Load trained model
result = PipelineResult.from_directory('./output/conve_model')

# Load test data
test_triples = TriplesFactory.from_path(
    path='./data/processed/test.txt',
    entity_to_id=result.training.entity_to_id,
    relation_to_id=result.training.relation_to_id
)

# Evaluate with detailed results
results = evaluate_model(
    model=result.model,
    test_triples=test_triples,
    training_triples=result.training,
    validation_triples=result.validation,
    filter_triples=True,
    output_path='./output/detailed_results.json'
)

# Access metrics
print(f"MRR: {results['metrics']['mean_reciprocal_rank']:.4f}")
print(f"Hits@10: {results['metrics']['hits@10']:.4f}")

# Access per-triple results
for triple_result in results['per_triple_results'][:5]:
    print(f"Triple: ({triple_result['head']}, {triple_result['relation']}, {triple_result['tail']})")
    print(f"  Score: {triple_result['score']:.4f}")
    print(f"  Rank: {triple_result['rank']}")
    print(f"  Top predictions: {triple_result['top_predictions'][:3]}")
```

### 4. TracIn Analysis

Analyze which training examples influence test predictions:

```python
from pykeen.pipeline import PipelineResult
from tracin import TracInAnalyzer
from pykeen.triples import TriplesFactory

# Load trained model
result = PipelineResult.from_directory('./output/conve_model')

# Load data
train_triples = result.training
test_triples = result.testing

# Create TracIn analyzer
analyzer = TracInAnalyzer(model=result.model)

# Analyze a single test triple
test_triple = (0, 1, 5)  # (head_id, relation_id, tail_id)
influences = analyzer.compute_influences_for_test_triple(
    test_triple=test_triple,
    training_triples=train_triples,
    learning_rate=0.001,
    top_k=10
)

# Print most influential training triples
print(f"Top influential training triples for {test_triple}:")
for inf in influences[:5]:
    print(f"  Train triple: ({inf['train_head']}, {inf['train_relation']}, {inf['train_tail']})")
    print(f"  Influence: {inf['influence']:.6f}")

# Analyze full test set (warning: computationally expensive)
full_analysis = analyzer.analyze_test_set(
    test_triples=test_triples,
    training_triples=train_triples,
    learning_rate=0.001,
    top_k=10,
    max_test_triples=100,  # Limit for speed
    output_path='./output/tracin_analysis.json'
)

# Compute self-influence for training examples
self_influences = analyzer.compute_self_influence(
    training_triples=train_triples,
    learning_rate=0.001,
    output_path='./output/self_influences.json'
)
```

## Output Files

After training and evaluation, you'll find:

```
output/conve_model/
├── config.json                    # Training configuration
├── trained_model.pkl              # Trained PyKEEN model
├── training_triples/             # Training data
├── validation_triples/           # Validation data
├── testing_triples/              # Test data
├── test_results.json             # Detailed test results
├── test_results.csv              # Test results in CSV format
└── losses.tsv                    # Training losses
```

## Example Workflow

```bash
# 1. Preprocess data
python preprocess.py \
  --train data/raw/train.tsv \
  --valid data/raw/valid.tsv \
  --test data/raw/test.tsv \
  --node-dict data/raw/node_dict \
  --rel-dict data/raw/rel_dict \
  --output-dir data/processed

# 2. Train model
python train.py \
  --train data/processed/train.txt \
  --valid data/processed/valid.txt \
  --test data/processed/test.txt \
  --entity-to-id data/processed/train_entity_to_id.tsv \
  --relation-to-id data/processed/train_relation_to_id.tsv \
  --output-dir output/model \
  --num-epochs 100 \
  --batch-size 256

# 3. Results will be automatically generated in output/model/
```

## Understanding the Results

### Evaluation Metrics

- **Mean Rank (MR)**: Average rank of correct entities (lower is better)
- **Mean Reciprocal Rank (MRR)**: Average of 1/rank (higher is better, range 0-1)
- **Hits@k**: Percentage of correct entities in top-k predictions (higher is better)

### Per-Triple Results

Each test triple gets:
- `score`: Model's confidence score for the triple
- `rank`: Position of the correct entity in ranked predictions
- `reciprocal_rank`: 1/rank
- `hits@k`: Binary indicators for top-k accuracy
- `top_predictions`: Top-10 predicted entities with scores

### TracIn Influence Scores

- **Positive influence**: Training example pushes prediction toward correctness
- **Negative influence**: Training example pushes prediction away from correctness
- **Magnitude**: Indicates strength of influence
- **Self-influence**: Measures example's importance/difficulty

## Cross-Method Comparison

This implementation preserves your original node_dict and rel_dict indices, ensuring compatibility with other methods. The entity and relation IDs in predictions can be directly compared across different models.

```python
# Example: Compare predictions from different methods
conve_predictions = load_predictions('conve_results.json')
other_predictions = load_predictions('other_method_results.json')

# Entity IDs are compatible
assert conve_predictions[0]['head'] == other_predictions[0]['head']
```

## Model Architecture

ConvE uses 2D convolution over reshaped entity and relation embeddings:

1. Entity and relation embeddings are reshaped into 2D matrices
2. Concatenated along width dimension
3. 2D convolution applied
4. Flattened and projected back to embedding dimension
5. Scored against all entities via dot product

## Performance Tips

1. **GPU Usage**: Use `--no-gpu` flag if you encounter memory issues
2. **Batch Size**: Reduce `--batch-size` if out of memory
3. **TracIn**: Very computationally expensive - use `max_test_triples` parameter
4. **Early Stopping**: Enabled by default, monitors Hits@10 on validation set

## Troubleshooting

### Out of Memory Error
```bash
# Reduce batch size
python train.py ... --batch-size 128

# Use CPU
python train.py ... --no-gpu
```

### Embedding Dimension Error
Ensure `embedding_dim = embedding_height × embedding_width`:
```bash
# Valid configurations:
--embedding-dim 200 --embedding-height 10 --embedding-width 20
--embedding-dim 100 --embedding-height 10 --embedding-width 10
```

### Index Mismatch
Ensure you use the same node_dict and rel_dict for all train/valid/test splits.

### Label Smoothing Error

**Error:**
```
pykeen.losses.UnsupportedLabelSmoothingError: MarginRankingLoss does not support label smoothing.
```

**Solution:**
This has been fixed in the current version. The model now uses `BCEWithLogitsLoss` which supports label smoothing.

If you still encounter this error:
```bash
# Option 1: Disable label smoothing
python train.py ... --label-smoothing 0

# Option 2: Update to latest version
git pull origin main
```

The default label smoothing (0.1) is recommended as it improves generalization.

## Citation

If you use this implementation, please cite:

**ConvE:**
```bibtex
@inproceedings{dettmers2018conve,
  title={Convolutional 2d knowledge graph embeddings},
  author={Dettmers, Tim and Minervini, Pasquale and Stenetorp, Pontus and Riedel, Sebastian},
  booktitle={AAAI},
  year={2018}
}
```

**PyKEEN:**
```bibtex
@article{ali2021pykeen,
  title={PyKEEN 1.0: A Python Library for Training and Evaluating Knowledge Graph Embeddings},
  author={Ali, Mehdi and others},
  journal={Journal of Machine Learning Research},
  year={2021}
}
```

**TracIn:**
```bibtex
@inproceedings{pruthi2020estimating,
  title={Estimating Training Data Influence by Tracing Gradient Descent},
  author={Pruthi, Garima and Liu, Frederick and Kale, Satyen and Sundararajan, Mukund},
  booktitle={NeurIPS},
  year={2020}
}
```

## License

This project is provided as-is for research purposes.
