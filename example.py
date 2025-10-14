"""
Example script demonstrating the complete ConvE workflow.

This script shows how to:
1. Load a trained model
2. Make predictions
3. Evaluate on test data
4. Perform TracIn analysis
"""

import logging
from pathlib import Path

import torch
from pykeen.pipeline import PipelineResult
from pykeen.triples import TriplesFactory

from evaluate import evaluate_model, DetailedEvaluator
from tracin import TracInAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_evaluation(model_dir: str, test_data_path: str):
    """Example: Evaluate model on test data with detailed metrics.

    Args:
        model_dir: Path to trained model directory
        test_data_path: Path to test triples file
    """
    logger.info("=" * 60)
    logger.info("Example 1: Model Evaluation")
    logger.info("=" * 60)

    # Load model
    result = PipelineResult.from_directory(model_dir)
    logger.info(f"Loaded model from {model_dir}")

    # Load test data
    test_triples = TriplesFactory.from_path(
        path=test_data_path,
        entity_to_id=result.training.entity_to_id,
        relation_to_id=result.training.relation_to_id
    )
    logger.info(f"Loaded {len(test_triples)} test triples")

    # Evaluate
    eval_results = evaluate_model(
        model=result.model,
        test_triples=test_triples,
        training_triples=result.training,
        validation_triples=result.validation,
        filter_triples=True,
        output_path='example_evaluation.json'
    )

    # Print results
    metrics = eval_results['metrics']
    logger.info("\nEvaluation Metrics:")
    logger.info(f"  Mean Rank: {metrics['mean_rank']:.2f}")
    logger.info(f"  MRR: {metrics['mean_reciprocal_rank']:.4f}")
    logger.info(f"  Hits@1: {metrics['hits@1']:.4f}")
    logger.info(f"  Hits@10: {metrics['hits@10']:.4f}")

    # Show some example predictions
    logger.info("\nExample Predictions (first 3 test triples):")
    for i, triple_result in enumerate(eval_results['per_triple_results'][:3]):
        logger.info(f"\n  Test Triple {i+1}:")
        logger.info(f"    (h={triple_result['head']}, r={triple_result['relation']}, t={triple_result['tail']})")
        logger.info(f"    Score: {triple_result['score']:.4f}")
        logger.info(f"    Rank: {triple_result['rank']}")
        logger.info(f"    Top-3 predictions:")
        for pred in triple_result['top_predictions'][:3]:
            logger.info(f"      Entity {pred['entity']}: {pred['score']:.4f}")


def example_prediction(model_dir: str):
    """Example: Make predictions for new queries.

    Args:
        model_dir: Path to trained model directory
    """
    logger.info("\n" + "=" * 60)
    logger.info("Example 2: Making Predictions")
    logger.info("=" * 60)

    # Load model
    result = PipelineResult.from_directory(model_dir)
    entity_to_id = result.training.entity_to_id
    relation_to_id = result.training.relation_to_id

    # Create reverse mappings
    id_to_entity = {v: k for k, v in entity_to_id.items()}
    id_to_relation = {v: k for k, v in relation_to_id.items()}

    # Create evaluator
    evaluator = DetailedEvaluator(model=result.model, filter_triples=False)

    # Example queries (using actual entity/relation labels)
    # Note: Replace these with actual entities/relations from your data
    example_entities = list(entity_to_id.keys())[:5]
    example_relations = list(relation_to_id.keys())[:2]

    if example_entities and example_relations:
        head = example_entities[0]
        relation = example_relations[0]

        head_id = entity_to_id[head]
        relation_id = relation_to_id[relation]

        logger.info(f"\nQuery: ({head}, {relation}, ?)")

        # Make prediction
        predictions = evaluator.predict_tails(
            head=head_id,
            relation=relation_id,
            top_k=5
        )

        logger.info(f"Top-5 Predicted Tails:")
        for i, (entity_id, score) in enumerate(predictions):
            entity_label = id_to_entity[entity_id]
            logger.info(f"  {i+1}. {entity_label} (score: {score:.4f})")
    else:
        logger.warning("No entities/relations available for prediction example")


def example_tracin(model_dir: str, train_data_path: str, test_data_path: str):
    """Example: Perform TracIn analysis.

    Args:
        model_dir: Path to trained model directory
        train_data_path: Path to training triples file
        test_data_path: Path to test triples file
    """
    logger.info("\n" + "=" * 60)
    logger.info("Example 3: TracIn Analysis")
    logger.info("=" * 60)
    logger.info("WARNING: TracIn is computationally expensive!")
    logger.info("This example analyzes only 1 test triple for demonstration.")

    # Load model
    result = PipelineResult.from_directory(model_dir)
    logger.info(f"Loaded model from {model_dir}")

    # Load data
    train_triples = TriplesFactory.from_path(
        path=train_data_path,
        entity_to_id=result.training.entity_to_id,
        relation_to_id=result.training.relation_to_id
    )

    test_triples = TriplesFactory.from_path(
        path=test_data_path,
        entity_to_id=result.training.entity_to_id,
        relation_to_id=result.training.relation_to_id
    )

    # Create analyzer
    analyzer = TracInAnalyzer(model=result.model)

    # Analyze first test triple
    if len(test_triples) > 0:
        test_triple = tuple(int(x) for x in test_triples.mapped_triples[0])
        logger.info(f"\nAnalyzing test triple: {test_triple}")
        logger.info("Computing influences (this may take a while)...")

        influences = analyzer.compute_influences_for_test_triple(
            test_triple=test_triple,
            training_triples=train_triples,
            learning_rate=0.001,
            top_k=5
        )

        logger.info(f"\nTop-5 Most Influential Training Triples:")
        for i, inf in enumerate(influences):
            logger.info(f"  {i+1}. Train triple: ({inf['train_head']}, {inf['train_relation']}, {inf['train_tail']})")
            logger.info(f"     Influence: {inf['influence']:.6f}")
    else:
        logger.warning("No test triples available for TracIn example")


def example_embeddings(model_dir: str):
    """Example: Extract and analyze embeddings.

    Args:
        model_dir: Path to trained model directory
    """
    logger.info("\n" + "=" * 60)
    logger.info("Example 4: Embedding Analysis")
    logger.info("=" * 60)

    # Load model
    result = PipelineResult.from_directory(model_dir)

    # Create evaluator
    evaluator = DetailedEvaluator(model=result.model)

    # Get embeddings
    entity_embeddings = evaluator.get_entity_embeddings()
    relation_embeddings = evaluator.get_relation_embeddings()

    logger.info(f"\nEntity embeddings shape: {entity_embeddings.shape}")
    logger.info(f"Relation embeddings shape: {relation_embeddings.shape}")

    # Example: Find most similar entities (by cosine similarity)
    if len(entity_embeddings) > 1:
        # Normalize embeddings
        entity_embeddings_norm = entity_embeddings / entity_embeddings.norm(dim=1, keepdim=True)

        # Compute similarity for first entity
        target_idx = 0
        similarities = torch.matmul(
            entity_embeddings_norm[target_idx].unsqueeze(0),
            entity_embeddings_norm.t()
        ).squeeze()

        # Get top-5 most similar
        top_k = min(5, len(similarities))
        top_similarities, top_indices = torch.topk(similarities, k=top_k)

        id_to_entity = {v: k for k, v in result.training.entity_to_id.items()}
        target_entity = id_to_entity[target_idx]

        logger.info(f"\nMost similar entities to '{target_entity}':")
        for i, (idx, sim) in enumerate(zip(top_indices, top_similarities)):
            if idx != target_idx:  # Skip self
                entity = id_to_entity[int(idx)]
                logger.info(f"  {i}. {entity} (similarity: {sim:.4f})")


def main():
    """Run all examples."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python example.py <model_dir> [<train_data>] [<test_data>]")
        print("\nExample:")
        print("  python example.py ./output/conve_model ./data/processed/train.txt ./data/processed/test.txt")
        sys.exit(1)

    model_dir = sys.argv[1]
    train_data_path = sys.argv[2] if len(sys.argv) > 2 else None
    test_data_path = sys.argv[3] if len(sys.argv) > 3 else None

    # Check if model exists
    if not Path(model_dir).exists():
        logger.error(f"Model directory not found: {model_dir}")
        sys.exit(1)

    # Run examples
    logger.info("Running ConvE Examples")
    logger.info("=" * 60)

    # Example 1: Evaluation (requires test data)
    if test_data_path and Path(test_data_path).exists():
        example_evaluation(model_dir, test_data_path)
    else:
        logger.warning("Skipping evaluation example (test data not provided)")

    # Example 2: Prediction
    example_prediction(model_dir)

    # Example 3: TracIn (requires both train and test data)
    if train_data_path and test_data_path and \
       Path(train_data_path).exists() and Path(test_data_path).exists():
        user_input = input("\nRun TracIn example? This is computationally expensive. (y/N): ")
        if user_input.lower() == 'y':
            example_tracin(model_dir, train_data_path, test_data_path)
    else:
        logger.warning("Skipping TracIn example (train/test data not provided)")

    # Example 4: Embeddings
    example_embeddings(model_dir)

    logger.info("\n" + "=" * 60)
    logger.info("Examples completed!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
