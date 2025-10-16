# Label Smoothing Documentation Summary

This document summarizes all the label smoothing documentation added to the repository.

## What is Label Smoothing?

Label smoothing is a regularization technique that:
- Prevents the model from becoming overconfident in predictions
- Softens hard labels: `1.0 → 0.9`, `0.0 → 0.1` (with smoothing = 0.1)
- Improves model generalization and reduces overfitting
- Formula: `smoothed_label = (1 - α) * label + α / num_classes`

## Default Configuration

- **Default value**: `0.1` (10% smoothing)
- **Loss function**: `BCEWithLogitsLoss` (supports label smoothing)
- **Recommended**: Keep enabled for better generalization

## Files Updated

### 1. train.py (Code Fix)
**Location**: Lines 203-233

**Changes**:
- Explicitly set `loss='BCEWithLogitsLoss'`
- Made label smoothing conditional (only added if > 0)
- Added logging when label smoothing is used

**Code snippet**:
```python
training_kwargs = {
    'num_epochs': num_epochs,
    'batch_size': batch_size,
}

if label_smoothing > 0:
    training_kwargs['label_smoothing'] = label_smoothing
    logger.info(f"Using label smoothing: {label_smoothing}")

result = pipeline(
    ...
    loss='BCEWithLogitsLoss',  # Supports label smoothing
    training_kwargs=training_kwargs,
    ...
)
```

### 2. README.md
**Sections Updated**:

#### A. Training Options (Lines 149-170)
Added detailed explanation:
```markdown
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
```

#### B. Troubleshooting Section (Lines 390-409)
Added error solution:
```markdown
### Label Smoothing Error

**Error:**
```
pykeen.losses.UnsupportedLabelSmoothingError: MarginRankingLoss does not support label smoothing.
```

**Solution:**
This has been fixed in the current version. The model now uses `BCEWithLogitsLoss`.

If you still encounter this error:
```bash
# Option 1: Disable label smoothing
python train.py ... --label-smoothing 0

# Option 2: Update to latest version
git pull origin main
```

The default label smoothing (0.1) is recommended as it improves generalization.
```

### 3. QUICKSTART.md
**Section**: Common Issues (Lines 150-162)

Added troubleshooting:
```markdown
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
```

### 4. COMPLETE_SETUP.md
**Sections Updated**:

#### A. Training Example (Lines 166-179)
Added parameter to example:
```bash
python train.py \
  --train data/processed/train.txt \
  --valid data/processed/valid.txt \
  --test data/processed/test.txt \
  --entity-to-id data/processed/train_entity_to_id.tsv \
  --relation-to-id data/processed/train_relation_to_id.tsv \
  --output-dir output/model \
  --num-epochs 100 \
  --batch-size 256 \
  --learning-rate 0.001 \
  --label-smoothing 0.1  # Optional: 0.0 to disable, default is 0.1
```

#### B. Troubleshooting Section (Lines 265-275)
Added error solution:
```markdown
### Label Smoothing Error
```
pykeen.losses.UnsupportedLabelSmoothingError: MarginRankingLoss does not support label smoothing.
```

**Fixed in current version.** The model now uses `BCEWithLogitsLoss`. If you still see this:
```bash
python train.py ... --label-smoothing 0  # Disable label smoothing
```

Label smoothing (default: 0.1) helps prevent overfitting and is recommended.
```

## Usage Examples

### With Label Smoothing (Default)
```bash
python train.py \
  --train ./processed/train.txt \
  --valid ./processed/valid.txt \
  --test ./processed/test.txt \
  --entity-to-id ./processed/train_entity_to_id.tsv \
  --relation-to-id ./processed/train_relation_to_id.tsv \
  --output-dir ./output/conve_model
# Label smoothing = 0.1 by default
```

### Without Label Smoothing
```bash
python train.py \
  --train ./processed/train.txt \
  --valid ./processed/valid.txt \
  --test ./processed/test.txt \
  --entity-to-id ./processed/train_entity_to_id.tsv \
  --relation-to-id ./processed/train_relation_to_id.tsv \
  --output-dir ./output/conve_model \
  --label-smoothing 0
```

### Custom Label Smoothing
```bash
python train.py \
  --train ./processed/train.txt \
  --valid ./processed/valid.txt \
  --test ./processed/test.txt \
  --entity-to-id ./processed/train_entity_to_id.tsv \
  --relation-to-id ./processed/train_relation_to_id.tsv \
  --output-dir ./output/conve_model \
  --label-smoothing 0.05  # 5% smoothing
```

## Command Line Options

```bash
--label-smoothing 0.1    # Default: 0.1 (10% smoothing)
--label-smoothing 0.0    # Disable label smoothing
--label-smoothing 0.05   # Light smoothing (5%)
--label-smoothing 0.2    # Heavy smoothing (20%)
```

## When to Use

### Use Default (0.1) When:
- ✅ Training a new model
- ✅ Dataset has class imbalance
- ✅ Want better generalization
- ✅ Following best practices

### Disable (0.0) When:
- ⚠️ Debugging training issues
- ⚠️ Comparing with baseline that doesn't use it
- ⚠️ Very small dataset (< 1000 examples)
- ⚠️ Already using heavy regularization

### Increase (0.15-0.2) When:
- 📊 Model is overfitting
- 📊 Very confident but wrong predictions
- 📊 Large dataset with many classes

## Technical Details

### Loss Function
- **Used**: `BCEWithLogitsLoss` (Binary Cross-Entropy with Logits)
- **Why**: Supports label smoothing unlike `MarginRankingLoss`
- **Stable**: Combines sigmoid and BCE for numerical stability

### Formula
For binary classification with smoothing α:
```
positive_label = (1 - α) * 1.0 + α * 0.5 = 1 - α/2
negative_label = (1 - α) * 0.0 + α * 0.5 = α/2

With α = 0.1:
  positive → 0.95
  negative → 0.05
```

### Implementation
The label smoothing is applied during training by PyKEEN's training loop:
```python
if label_smoothing > 0:
    training_kwargs['label_smoothing'] = label_smoothing
```

## Error Resolution

### Original Error
```
pykeen.losses.UnsupportedLabelSmoothingError: MarginRankingLoss does not support label smoothing.
```

### Root Cause
PyKEEN's default loss function for ConvE was `MarginRankingLoss`, which doesn't support label smoothing.

### Fix Applied
1. Explicitly specify `loss='BCEWithLogitsLoss'`
2. Make label smoothing conditional
3. Add informative logging

### Status
✅ **FIXED** in current version

## Recommendations

1. **Default users**: Keep `--label-smoothing 0.1` (recommended)
2. **Experimenting**: Try values between 0.05 and 0.2
3. **Debugging**: Use `--label-smoothing 0` temporarily
4. **Production**: Use default 0.1 for best generalization

## References

- Original ConvE paper: Dettmers et al., 2018
- Label smoothing: Szegedy et al., "Rethinking the Inception Architecture", 2016
- PyKEEN documentation: https://pykeen.readthedocs.io/

## Summary

✅ Label smoothing is now:
- **Documented** in 4 key markdown files
- **Working** with BCEWithLogitsLoss
- **Configurable** via command line
- **Recommended** for better model performance

All markdown files have been updated with comprehensive documentation about label smoothing! 📚✨
