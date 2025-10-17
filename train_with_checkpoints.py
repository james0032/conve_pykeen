"""
Training script for ConvE with proper checkpoint support.

This version uses PyKEEN's lower-level API to enable checkpoint functionality.
"""

import argparse
import json
import logging
import os
from pathlib import Path

import torch
from pykeen.datasets import Dataset
from pykeen.models import ConvE
from pykeen.training import SLCWATrainingLoop
from pykeen.evaluation import RankBasedEvaluator
from pykeen.stoppers import EarlyStopper
from pykeen.triples import TriplesFactory
from pykeen.trackers import MLFlowResultTracker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_triples_factory(
    triples_path: str,
    entity_to_id_path: str,
    relation_to_id_path: str,
) -> TriplesFactory:
    """Load triples factory from preprocessed files."""
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
    )

    logger.info(f"  Loaded {factory.num_triples} triples")
    return factory


def train_with_checkpoints(
    train_path: str,
    valid_path: str,
    test_path: str,
    entity_to_id_path: str,
    relation_to_id_path: str,
    output_dir: str,
    checkpoint_dir: str,
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
    checkpoint_frequency: int = 5,
    eval_frequency: int = 1,
    # Other options
    use_gpu: bool = True,
    random_seed: int = 42,
    early_stopping: bool = True,
    patience: int = 10,
):
    """Train ConvE model with checkpoint support.

    Args:
        train_path: Path to training triples
        valid_path: Path to validation triples
        test_path: Path to test triples
        entity_to_id_path: Path to entity-to-ID mapping
        relation_to_id_path: Path to relation-to-ID mapping
        output_dir: Output directory for models and results
        checkpoint_dir: Directory to save checkpoints
        embedding_dim: Embedding dimension
        output_channels: Number of convolution output channels
        kernel_height: Convolution kernel height
        kernel_width: Convolution kernel width
        input_dropout: Input dropout rate
        feature_map_dropout: Feature map dropout rate
        output_dropout: Output dropout rate
        embedding_height: Height of reshaped embeddings
        embedding_width: Width of reshaped embeddings
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for Adam optimizer
        checkpoint_frequency: Save checkpoint every N epochs (default: 5)
        eval_frequency: Evaluate on validation set every N epochs (default: 1)
        use_gpu: Whether to use GPU
        random_seed: Random seed for reproducibility
        early_stopping: Whether to use early stopping
        patience: Early stopping patience (number of evaluations)
    """

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Set device
    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Load data
    logger.info("Loading datasets...")
    training = load_triples_factory(train_path, entity_to_id_path, relation_to_id_path)
    validation = load_triples_factory(valid_path, entity_to_id_path, relation_to_id_path)
    testing = load_triples_factory(test_path, entity_to_id_path, relation_to_id_path)

    # Create model
    logger.info("Creating ConvE model...")
    model = ConvE(
        triples_factory=training,
        embedding_dim=embedding_dim,
        output_channels=output_channels,
        kernel_height=kernel_height,
        kernel_width=kernel_width,
        input_dropout=input_dropout,
        feature_map_dropout=feature_map_dropout,
        output_dropout=output_dropout,
        embedding_height=embedding_height,
        embedding_width=embedding_width,
        random_seed=random_seed,
    ).to(device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Create training loop with checkpoint support
    logger.info("Creating training loop...")
    training_loop = SLCWATrainingLoop(
        model=model,
        triples_factory=training,
        optimizer=optimizer,
    )

    # Create evaluator
    evaluator = RankBasedEvaluator(filtered=True)

    # Create early stopper if requested
    stopper = None
    if early_stopping:
        logger.info(f"Using early stopping with patience={patience}, eval_frequency={eval_frequency}")
        stopper = EarlyStopper(
            model=model,
            evaluator=evaluator,
            training_triples_factory=training,
            evaluation_triples_factory=validation,
            frequency=eval_frequency,  # Evaluate every N epochs
            patience=patience,
            relative_delta=0.001,
            metric='hits@10',
            larger_is_better=True,
        )

    # Training with manual checkpoint saving
    logger.info("Starting training...")
    logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")

    best_hits_at_10 = 0.0

    for epoch in range(1, num_epochs + 1):
        # Train for one epoch
        losses = training_loop.train(
            triples_factory=training,
            num_epochs=1,
            batch_size=batch_size,
        )

        avg_loss = sum(losses) / len(losses) if losses else 0.0
        logger.info(f"Epoch {epoch}/{num_epochs} - Avg Loss: {avg_loss:.4f}")

        # Save checkpoint at specified frequency
        if epoch % checkpoint_frequency == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f'checkpoint_epoch_{epoch:03d}.pt'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Evaluate and early stopping check
        if stopper is not None:
            should_stop = stopper.should_stop()
            if should_stop:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

            # Track best model
            results = stopper.best_metric_value
            if results > best_hits_at_10:
                best_hits_at_10 = results
                best_model_path = os.path.join(output_dir, 'best_model.pt')
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"New best model saved (Hits@10: {best_hits_at_10:.4f})")

    # Save final model
    final_model_path = os.path.join(output_dir, 'final_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': {
            'embedding_dim': embedding_dim,
            'output_channels': output_channels,
            'num_epochs': epoch,
        }
    }, final_model_path)
    logger.info(f"Saved final model: {final_model_path}")

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_results = evaluator.evaluate(
        model=model,
        mapped_triples=testing.mapped_triples,
        batch_size=batch_size,
        additional_filter_triples=[
            training.mapped_triples,
            validation.mapped_triples,
        ],
    )

    # Save results
    results_dict = test_results.to_dict()
    results_path = os.path.join(output_dir, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)

    logger.info("=" * 60)
    logger.info("Final Test Results:")
    logger.info(f"  Mean Rank: {results_dict['mean_rank']:.2f}")
    logger.info(f"  MRR: {results_dict['mean_reciprocal_rank']:.4f}")
    logger.info(f"  Hits@1: {results_dict['hits_at_1']:.4f}")
    logger.info(f"  Hits@3: {results_dict['hits_at_3']:.4f}")
    logger.info(f"  Hits@10: {results_dict['hits_at_10']:.4f}")
    logger.info("=" * 60)

    return model


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train ConvE with checkpoint support'
    )

    # Data arguments
    parser.add_argument('--train', type=str, required=True)
    parser.add_argument('--valid', type=str, required=True)
    parser.add_argument('--test', type=str, required=True)
    parser.add_argument('--entity-to-id', type=str, required=True)
    parser.add_argument('--relation-to-id', type=str, required=True)
    parser.add_argument('--output-dir', '-o', type=str, required=True)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')

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
    parser.add_argument('--checkpoint-frequency', type=int, default=5,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--eval-frequency', type=int, default=1,
                       help='Evaluate on validation set every N epochs (for early stopping)')

    # Other options
    parser.add_argument('--no-gpu', action='store_true')
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument('--no-early-stopping', action='store_true')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience (number of evaluations without improvement)')

    return parser.parse_args()


def main():
    args = parse_args()

    train_with_checkpoints(
        train_path=args.train,
        valid_path=args.valid,
        test_path=args.test,
        entity_to_id_path=args.entity_to_id,
        relation_to_id_path=args.relation_to_id,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
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
        checkpoint_frequency=args.checkpoint_frequency,
        eval_frequency=args.eval_frequency,
        use_gpu=not args.no_gpu,
        random_seed=args.random_seed,
        early_stopping=not args.no_early_stopping,
        patience=args.patience,
    )


if __name__ == '__main__':
    main()
