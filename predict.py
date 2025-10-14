"""
Prediction script for making inferences with trained ConvE model.

This script allows you to:
1. Make predictions for specific (head, relation) queries
2. Batch predict for multiple queries
3. Export predictions in various formats
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import torch
from pykeen.pipeline import PipelineResult
from pykeen.triples import TriplesFactory

from evaluate import DetailedEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model_and_mappings(model_dir: str):
    """Load trained model and entity/relation mappings.

    Args:
        model_dir: Directory containing trained model

    Returns:
        Tuple of (model, entity_to_id, relation_to_id, id_to_entity, id_to_relation)
    """
    logger.info(f"Loading model from {model_dir}")
    result = PipelineResult.from_directory(model_dir)

    entity_to_id = result.training.entity_to_id
    relation_to_id = result.training.relation_to_id

    # Create reverse mappings
    id_to_entity = {v: k for k, v in entity_to_id.items()}
    id_to_relation = {v: k for k, v in relation_to_id.items()}

    return result.model, entity_to_id, relation_to_id, id_to_entity, id_to_relation


def predict_tails(
    model_dir: str,
    queries: List[Tuple[str, str]],
    top_k: int = 10,
    output_path: Optional[str] = None,
    device: str = 'cpu'
):
    """Predict tail entities for (head, relation) queries.

    Args:
        model_dir: Directory containing trained model
        queries: List of (head_entity, relation) tuples
        top_k: Number of top predictions to return
        output_path: Optional path to save predictions
        device: Device to run prediction on
    """
    # Load model and mappings
    model, entity_to_id, relation_to_id, id_to_entity, id_to_relation = \
        load_model_and_mappings(model_dir)

    # Create evaluator
    evaluator = DetailedEvaluator(
        model=model,
        filter_triples=False,
        device=device
    )

    # Make predictions
    all_predictions = []

    for head_entity, relation in queries:
        # Check if entities/relations exist
        if head_entity not in entity_to_id:
            logger.warning(f"Unknown entity: {head_entity}")
            continue
        if relation not in relation_to_id:
            logger.warning(f"Unknown relation: {relation}")
            continue

        # Get IDs
        head_id = entity_to_id[head_entity]
        relation_id = relation_to_id[relation]

        # Predict
        predictions = evaluator.predict_tails(
            head=head_id,
            relation=relation_id,
            top_k=top_k,
            known_triples=None
        )

        # Convert IDs back to labels
        predictions_with_labels = [
            {
                'rank': i + 1,
                'entity': id_to_entity[entity_id],
                'entity_id': entity_id,
                'score': score
            }
            for i, (entity_id, score) in enumerate(predictions)
        ]

        result = {
            'query': {
                'head': head_entity,
                'head_id': head_id,
                'relation': relation,
                'relation_id': relation_id
            },
            'predictions': predictions_with_labels
        }

        all_predictions.append(result)

        # Print results
        logger.info(f"\nQuery: ({head_entity}, {relation}, ?)")
        logger.info(f"Top {min(5, top_k)} predictions:")
        for pred in predictions_with_labels[:5]:
            logger.info(f"  {pred['rank']}. {pred['entity']} (score: {pred['score']:.4f})")

    # Save results if path provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(all_predictions, f, indent=2)
        logger.info(f"\nSaved predictions to {output_path}")

        # Also save as CSV
        csv_path = output_path.replace('.json', '.csv')
        rows = []
        for result in all_predictions:
            query = result['query']
            for pred in result['predictions']:
                rows.append({
                    'head': query['head'],
                    'relation': query['relation'],
                    'rank': pred['rank'],
                    'predicted_tail': pred['entity'],
                    'score': pred['score']
                })
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved predictions to {csv_path}")

    return all_predictions


def predict_from_file(
    model_dir: str,
    query_file: str,
    output_path: str,
    top_k: int = 10,
    device: str = 'cpu'
):
    """Make predictions for queries from a file.

    Args:
        model_dir: Directory containing trained model
        query_file: Path to query file (TSV: head, relation)
        output_path: Path to save predictions
        top_k: Number of top predictions to return
        device: Device to run prediction on
    """
    # Load queries
    logger.info(f"Loading queries from {query_file}")
    queries = []
    with open(query_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                head, relation = parts[0], parts[1]
                queries.append((head, relation))

    logger.info(f"Loaded {len(queries)} queries")

    # Make predictions
    return predict_tails(
        model_dir=model_dir,
        queries=queries,
        top_k=top_k,
        output_path=output_path,
        device=device
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description='Make predictions with trained ConvE model'
    )

    parser.add_argument(
        '--model-dir', type=str, required=True,
        help='Directory containing trained model'
    )

    # Input options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--query', nargs=2, metavar=('HEAD', 'RELATION'),
        help='Single query: head entity and relation'
    )
    group.add_argument(
        '--query-file', type=str,
        help='File with queries (TSV: head, relation)'
    )

    parser.add_argument(
        '--top-k', type=int, default=10,
        help='Number of top predictions to return'
    )
    parser.add_argument(
        '--output', '-o', type=str,
        help='Output path for predictions'
    )
    parser.add_argument(
        '--device', type=str, default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to run prediction on'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.query:
        # Single query
        head, relation = args.query
        queries = [(head, relation)]

        predictions = predict_tails(
            model_dir=args.model_dir,
            queries=queries,
            top_k=args.top_k,
            output_path=args.output,
            device=args.device
        )

    elif args.query_file:
        # Batch queries from file
        predictions = predict_from_file(
            model_dir=args.model_dir,
            query_file=args.query_file,
            output_path=args.output or 'predictions.json',
            top_k=args.top_k,
            device=args.device
        )


if __name__ == '__main__':
    main()
