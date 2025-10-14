"""
Standalone script for running TracIn analysis on a trained model.

This script provides a convenient interface for analyzing training data
influence on test predictions.
"""

import argparse
import logging
from pathlib import Path

from pykeen.pipeline import PipelineResult
from pykeen.triples import TriplesFactory

from tracin import TracInAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_tracin_analysis(
    model_dir: str,
    train_path: str,
    test_path: str,
    output_path: str,
    mode: str = 'test',
    test_triple_indices: list = None,
    max_test_triples: int = None,
    top_k: int = 10,
    learning_rate: float = 0.001,
    device: str = 'cpu'
):
    """Run TracIn analysis.

    Args:
        model_dir: Directory containing trained model
        train_path: Path to training triples
        test_path: Path to test triples
        output_path: Path to save results
        mode: Analysis mode ('test', 'self', or 'single')
        test_triple_indices: Specific test triple indices to analyze (for single mode)
        max_test_triples: Maximum number of test triples to analyze
        top_k: Number of top influential triples to return
        learning_rate: Learning rate used during training
        device: Device to run on
    """
    logger.info("Loading model and data...")

    # Load model
    result = PipelineResult.from_directory(model_dir)

    # Load training triples
    train_triples = TriplesFactory.from_path(
        path=train_path,
        entity_to_id=result.training.entity_to_id,
        relation_to_id=result.training.relation_to_id
    )
    logger.info(f"Loaded {len(train_triples)} training triples")

    # Create analyzer
    analyzer = TracInAnalyzer(model=result.model, device=device)

    if mode == 'self':
        # Compute self-influence for training set
        logger.info("Computing self-influences for training set...")
        logger.info("This will take a while...")

        influences = analyzer.compute_self_influence(
            training_triples=train_triples,
            learning_rate=learning_rate,
            output_path=output_path
        )

        # Print top-10
        logger.info("\nTop-10 training examples by self-influence:")
        for i, inf in enumerate(influences[:10]):
            logger.info(f"  {i+1}. ({inf['head']}, {inf['relation']}, {inf['tail']})")
            logger.info(f"     Self-influence: {inf['self_influence']:.6f}")

    elif mode == 'test':
        # Analyze test set
        test_triples = TriplesFactory.from_path(
            path=test_path,
            entity_to_id=result.training.entity_to_id,
            relation_to_id=result.training.relation_to_id
        )
        logger.info(f"Loaded {len(test_triples)} test triples")

        logger.info("Analyzing influence on test predictions...")
        logger.info("WARNING: This is computationally expensive!")

        if max_test_triples:
            logger.info(f"Limiting analysis to first {max_test_triples} test triples")

        analysis = analyzer.analyze_test_set(
            test_triples=test_triples,
            training_triples=train_triples,
            learning_rate=learning_rate,
            top_k=top_k,
            max_test_triples=max_test_triples,
            output_path=output_path
        )

        logger.info(f"\nAnalyzed {analysis['num_test_triples']} test triples")
        logger.info(f"Results saved to {output_path}")

    elif mode == 'single':
        # Analyze specific test triples
        test_triples = TriplesFactory.from_path(
            path=test_path,
            entity_to_id=result.training.entity_to_id,
            relation_to_id=result.training.relation_to_id
        )

        if test_triple_indices is None:
            test_triple_indices = [0]  # Default to first test triple

        results = []
        for idx in test_triple_indices:
            if idx >= len(test_triples):
                logger.warning(f"Test triple index {idx} out of range (max: {len(test_triples)-1})")
                continue

            test_triple = tuple(int(x) for x in test_triples.mapped_triples[idx])
            logger.info(f"\nAnalyzing test triple {idx}: {test_triple}")

            influences = analyzer.compute_influences_for_test_triple(
                test_triple=test_triple,
                training_triples=train_triples,
                learning_rate=learning_rate,
                top_k=top_k
            )

            logger.info(f"Top-{min(5, top_k)} influential training triples:")
            for i, inf in enumerate(influences[:5]):
                logger.info(f"  {i+1}. ({inf['train_head']}, {inf['train_relation']}, {inf['train_tail']})")
                logger.info(f"     Influence: {inf['influence']:.6f}")

            results.append({
                'test_triple': test_triple,
                'test_triple_index': idx,
                'influences': influences
            })

        # Save results
        import json
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nResults saved to {output_path}")

    else:
        raise ValueError(f"Unknown mode: {mode}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run TracIn analysis on trained ConvE model'
    )

    parser.add_argument(
        '--model-dir', type=str, required=True,
        help='Directory containing trained model'
    )
    parser.add_argument(
        '--train', type=str, required=True,
        help='Path to training triples file'
    )
    parser.add_argument(
        '--test', type=str,
        help='Path to test triples file (required for test/single modes)'
    )
    parser.add_argument(
        '--output', '-o', type=str, required=True,
        help='Output path for results (JSON)'
    )

    parser.add_argument(
        '--mode', type=str, default='test',
        choices=['test', 'self', 'single'],
        help='Analysis mode: test (full test set), self (training self-influence), single (specific triples)'
    )

    parser.add_argument(
        '--test-indices', type=int, nargs='+',
        help='Test triple indices to analyze (for single mode)'
    )
    parser.add_argument(
        '--max-test-triples', type=int,
        help='Maximum number of test triples to analyze (for speed)'
    )
    parser.add_argument(
        '--top-k', type=int, default=10,
        help='Number of top influential triples to return per test triple'
    )
    parser.add_argument(
        '--learning-rate', type=float, default=0.001,
        help='Learning rate used during training'
    )
    parser.add_argument(
        '--device', type=str, default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to run on'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Validate arguments
    if args.mode in ['test', 'single'] and not args.test:
        parser.error(f"--test is required for mode '{args.mode}'")

    if args.mode == 'single' and not args.test_indices:
        logger.info("No test indices specified, will analyze first test triple")

    # Run analysis
    run_tracin_analysis(
        model_dir=args.model_dir,
        train_path=args.train,
        test_path=args.test,
        output_path=args.output,
        mode=args.mode,
        test_triple_indices=args.test_indices,
        max_test_triples=args.max_test_triples,
        top_k=args.top_k,
        learning_rate=args.learning_rate,
        device=args.device
    )

    logger.info("\nTracIn analysis completed!")


if __name__ == '__main__':
    main()
