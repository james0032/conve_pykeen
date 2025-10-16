"""
Training script for ConvE model using PyKEEN.

This script provides a complete training pipeline with:
1. Data loading from preprocessed files
2. Model training with configurable hyperparameters
3. Validation during training
4. Model checkpointing
5. Final evaluation with detailed metrics
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Optional

import torch
from pykeen.datasets import Dataset
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from pykeen.training import SLCWATrainingLoop
from pykeen.evaluation import RankBasedEvaluator
from pykeen.stoppers import EarlyStopper

from evaluate import evaluate_model
from model import ConvEWithGradients

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_triples_factory(
    triples_path: str,
    entity_to_id_path: str,
    relation_to_id_path: str,
    create_inverse_triples: bool = False
) -> TriplesFactory:
    """Load triples factory from preprocessed files.

    Args:
        triples_path: Path to triples file (TSV format)
        entity_to_id_path: Path to entity-to-ID mapping
        relation_to_id_path: Path to relation-to-ID mapping
        create_inverse_triples: Whether to create inverse triples

    Returns:
        TriplesFactory instance
    """
    logger.info(f"Loading triples from {triples_path}")

    # Load entity and relation mappings
    entity_to_id = {}
    with open(entity_to_id_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                entity, idx = parts
                entity_to_id[entity] = int(idx)

    relation_to_id = {}
    with open(relation_to_id_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                relation, idx = parts
                relation_to_id[relation] = int(idx)

    logger.info(f"  Loaded {len(entity_to_id)} entities")
    logger.info(f"  Loaded {len(relation_to_id)} relations")

    # Create triples factory
    factory = TriplesFactory.from_path(
        path=triples_path,
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
        create_inverse_triples=create_inverse_triples
    )

    logger.info(f"  Loaded {factory.num_triples} triples")

    return factory


def train_model(
    train_path: str,
    valid_path: str,
    test_path: str,
    entity_to_id_path: str,
    relation_to_id_path: str,
    output_dir: str,
    # Model hyperparameters
    embedding_dim: int = 200,
    output_channels: int = 32,
    kernel_height: int = 3,
    kernel_width: int = 3,
    input_dropout: float = 0.2,
    feature_map_dropout: float = 0.2,
    output_dropout: float = 0.3,
    embedding_height: int = 10,
    embedding_width: int = 20,
    # Training hyperparameters
    num_epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    label_smoothing: float = 0.1,
    # Other options
    use_gpu: bool = True,
    random_seed: int = 42,
    save_checkpoints: bool = True,
    checkpoint_frequency: int = 10,
    early_stopping: bool = True,
    patience: int = 10,
    track_gradients: bool = False
):
    """Train ConvE model.

    Args:
        train_path: Path to training triples
        valid_path: Path to validation triples
        test_path: Path to test triples
        entity_to_id_path: Path to entity mapping
        relation_to_id_path: Path to relation mapping
        output_dir: Output directory for model and results
        embedding_dim: Dimension of embeddings
        output_channels: Number of output channels in convolution
        kernel_height: Height of convolutional kernel
        kernel_width: Width of convolutional kernel
        input_dropout: Dropout rate for input
        feature_map_dropout: Dropout rate for feature maps
        output_dropout: Dropout rate for output layer
        embedding_height: Height of reshaped embeddings
        embedding_width: Width of reshaped embeddings
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        label_smoothing: Label smoothing parameter
        use_gpu: Whether to use GPU
        random_seed: Random seed for reproducibility
        save_checkpoints: Whether to save model checkpoints
        checkpoint_frequency: Save checkpoint every N epochs
        early_stopping: Whether to use early stopping
        patience: Patience for early stopping
        track_gradients: Whether to track gradients for TracIn
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Set device
    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Load data
    logger.info("Loading datasets...")
    training = load_triples_factory(train_path, entity_to_id_path, relation_to_id_path)
    validation = load_triples_factory(valid_path, entity_to_id_path, relation_to_id_path)
    testing = load_triples_factory(test_path, entity_to_id_path, relation_to_id_path)

    # Save configuration
    config = {
        'data': {
            'train_path': train_path,
            'valid_path': valid_path,
            'test_path': test_path,
            'num_entities': training.num_entities,
            'num_relations': training.num_relations,
            'num_train_triples': training.num_triples,
            'num_valid_triples': validation.num_triples,
            'num_test_triples': testing.num_triples,
        },
        'model': {
            'embedding_dim': embedding_dim,
            'output_channels': output_channels,
            'kernel_height': kernel_height,
            'kernel_width': kernel_width,
            'input_dropout': input_dropout,
            'feature_map_dropout': feature_map_dropout,
            'output_dropout': output_dropout,
            'embedding_height': embedding_height,
            'embedding_width': embedding_width,
        },
        'training': {
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'label_smoothing': label_smoothing,
            'random_seed': random_seed,
        }
    }

    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved configuration to {config_path}")

    # Train model using PyKEEN pipeline
    logger.info("Starting training...")

    # Prepare training kwargs - label smoothing only works with certain losses
    training_kwargs = {
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'checkpoint_directory': '/workspace/data/robokop/CGGD_alltreat/checkpoints',  # where to store checkpoints
        'checkpoint_frequency': 2,                # save every 2 epochs
        'checkpoint_name': 'conve_checkpoint',    # base name
        'checkpoint_on_failure': True,            # also save if interrupted
    }

    # Only add label smoothing if it's non-zero and using BCEWithLogits loss
    if label_smoothing > 0:
        training_kwargs['label_smoothing'] = label_smoothing
        logger.info(f"Using label smoothing: {label_smoothing}")

    result = pipeline(
        # Data
        training=training,
        validation=validation,
        testing=testing,
        # Model
        model='ConvE',
        model_kwargs={
            'embedding_dim': embedding_dim,
            'output_channels': output_channels,
            'kernel_height': kernel_height,
            'kernel_width': kernel_width,
            'input_dropout': input_dropout,
            'feature_map_dropout': feature_map_dropout,
            'output_dropout': output_dropout,
            'embedding_height': embedding_height,
            'embedding_width': embedding_width,
        },
        # Loss function - use BCEWithLogitsLoss which supports label smoothing
        loss='BCEWithLogitsLoss',
        # Training
        training_loop='sLCWA',
        training_kwargs=training_kwargs,
        # Optimizer
        optimizer='Adam',
        optimizer_kwargs={
            'lr': learning_rate,
        },
        # Evaluation
        evaluator='RankBasedEvaluator',
        evaluator_kwargs={
            'filtered': True,
        },
        evaluation_kwargs={
            'batch_size': batch_size,
        },
        # Stopping
        stopper='early' if early_stopping else None,
        stopper_kwargs={
            'patience': patience,
            'frequency': 1000,
            'metric': 'hits@10',
        } if early_stopping else None,
        # Other
        random_seed=random_seed,
        device=device,
    )

    logger.info("Training completed!")

    # Save model
    model_path = os.path.join(output_dir, 'model.pkl')
    result.save_to_directory(output_dir)
    logger.info(f"Saved model to {output_dir}")

    # Evaluate on test set with detailed metrics
    logger.info("Evaluating on test set...")
    test_results_path = os.path.join(output_dir, 'test_results.json')
    detailed_results = evaluate_model(
        model=result.model,
        test_triples=testing,
        training_triples=training,
        validation_triples=validation,
        batch_size=batch_size,
        filter_triples=True,
        output_path=test_results_path,
        device=device
    )

    # Print final results
    metrics = detailed_results['metrics']
    logger.info("=" * 60)
    logger.info("Final Test Results:")
    logger.info("=" * 60)
    logger.info(f"Mean Rank: {metrics['mean_rank']:.2f}")
    logger.info(f"Mean Reciprocal Rank: {metrics['mean_reciprocal_rank']:.4f}")
    logger.info(f"Hits@1: {metrics['hits@1']:.4f}")
    logger.info(f"Hits@3: {metrics['hits@3']:.4f}")
    logger.info(f"Hits@5: {metrics['hits@5']:.4f}")
    logger.info(f"Hits@10: {metrics['hits@10']:.4f}")
    logger.info("=" * 60)

    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train ConvE model using PyKEEN'
    )

    # Data arguments
    parser.add_argument(
        '--train', type=str, required=True,
        help='Path to training triples file'
    )
    parser.add_argument(
        '--valid', type=str, required=True,
        help='Path to validation triples file'
    )
    parser.add_argument(
        '--test', type=str, required=True,
        help='Path to test triples file'
    )
    parser.add_argument(
        '--entity-to-id', type=str, required=True,
        help='Path to entity-to-ID mapping file'
    )
    parser.add_argument(
        '--relation-to-id', type=str, required=True,
        help='Path to relation-to-ID mapping file'
    )
    parser.add_argument(
        '--output-dir', '-o', type=str, required=True,
        help='Output directory for model and results'
    )

    # Model hyperparameters
    parser.add_argument('--embedding-dim', type=int, default=200)
    parser.add_argument('--output-channels', type=int, default=32)
    parser.add_argument('--kernel-height', type=int, default=3)
    parser.add_argument('--kernel-width', type=int, default=3)
    parser.add_argument('--input-dropout', type=float, default=0.2)
    parser.add_argument('--feature-map-dropout', type=float, default=0.2)
    parser.add_argument('--output-dropout', type=float, default=0.3)
    parser.add_argument('--embedding-height', type=int, default=10)
    parser.add_argument('--embedding-width', type=int, default=20)

    # Training hyperparameters
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--label-smoothing', type=float, default=0.1)

    # Other options
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU usage')
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument('--no-early-stopping', action='store_true')
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--track-gradients', action='store_true',
                       help='Track gradients for TracIn analysis')

    return parser.parse_args()


def main():
    args = parse_args()

    train_model(
        train_path=args.train,
        valid_path=args.valid,
        test_path=args.test,
        entity_to_id_path=args.entity_to_id,
        relation_to_id_path=args.relation_to_id,
        output_dir=args.output_dir,
        embedding_dim=args.embedding_dim,
        output_channels=args.output_channels,
        kernel_height=args.kernel_height,
        kernel_width=args.kernel_width,
        input_dropout=args.input_dropout,
        feature_map_dropout=args.feature_map_dropout,
        output_dropout=args.output_dropout,
        embedding_height=args.embedding_height,
        embedding_width=args.embedding_width,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        label_smoothing=args.label_smoothing,
        use_gpu=not args.no_gpu,
        random_seed=args.random_seed,
        early_stopping=not args.no_early_stopping,
        patience=args.patience,
        track_gradients=args.track_gradients
    )


if __name__ == '__main__':
    main()
