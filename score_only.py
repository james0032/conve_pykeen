#!/usr/bin/env python3
"""
Score test triples without computing rankings.

This script gets ConvE scores for test triples WITHOUT computing expensive rankings.
Use this when you only need the model's confidence score for each triple.

Usage:
    python score_only.py --model-dir output/trained_model --test data/test.txt --output test_scores.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

import pandas as pd
import torch
from pykeen.models import ConvE
from pykeen.triples import TriplesFactory

from evaluate import DetailedEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_node_names(node_name_dict_path: str) -> Dict[int, str]:
    """Load entity index to name mapping from node_name_dict.txt.

    Args:
        node_name_dict_path: Path to node_name_dict.txt

    Returns:
        Dictionary mapping entity index to name
    """
    idx_to_name = {}

    if not Path(node_name_dict_path).exists():
        logger.warning(f"Node name dictionary not found: {node_name_dict_path}")
        return idx_to_name

    with open(node_name_dict_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) == 2:
                name, idx = parts
                idx_to_name[int(idx)] = name

    logger.info(f"Loaded {len(idx_to_name)} entity names")
    return idx_to_name


def main():
    parser = argparse.ArgumentParser(
        description='Score test triples without computing rankings (fast!)'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        required=True,
        help='Directory containing trained_model.pkl'
    )
    parser.add_argument(
        '--test',
        type=str,
        required=True,
        help='Path to test.txt'
    )
    parser.add_argument(
        '--entity-to-id',
        type=str,
        help='Path to entity_to_id.tsv (if not in same dir as test.txt)'
    )
    parser.add_argument(
        '--relation-to-id',
        type=str,
        help='Path to relation_to_id.tsv (if not in same dir as test.txt)'
    )
    parser.add_argument(
        '--node-name-dict',
        type=str,
        help='Path to node_name_dict.txt for entity names (if not in same dir as test.txt)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='test_scores.json',
        help='Output path for scores (default: test_scores.json)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu, default: auto-detect)'
    )
    parser.add_argument(
        '--use-sigmoid',
        action='store_true',
        help='Apply sigmoid to convert scores to probabilities [0, 1] (default: False, returns raw logits)'
    )

    args = parser.parse_args()

    # Determine model path
    model_path = Path(args.model_dir)

    # If args.model_dir is a file, use it directly
    if model_path.is_file():
        model_file = model_path
    # If it's a directory, look for best_model.pt or final_model.pt
    elif model_path.is_dir():
        if (model_path / 'best_model.pt').exists():
            model_file = model_path / 'best_model.pt'
            logger.info(f"Found best_model.pt in {args.model_dir}")
        elif (model_path / 'final_model.pt').exists():
            model_file = model_path / 'final_model.pt'
            logger.info(f"Found final_model.pt in {args.model_dir}")
        elif (model_path / 'trained_model.pkl').exists():
            model_file = model_path / 'trained_model.pkl'
            logger.info(f"Found trained_model.pkl in {args.model_dir}")
        else:
            logger.error(f"No model file found in {args.model_dir}")
            logger.error(f"Looking for: best_model.pt, final_model.pt, or trained_model.pkl")
            return
    else:
        logger.error(f"Model path does not exist: {args.model_dir}")
        return

    logger.info(f"Loading model from {model_file}...")

    # Load test triples
    logger.info(f"Loading test triples from {args.test}...")
    test_dir = Path(args.test).parent

    # Determine entity/relation mapping paths
    if args.entity_to_id:
        entity_map_path = args.entity_to_id
    else:
        entity_map_path = test_dir / 'entity_to_id.tsv'

    if args.relation_to_id:
        relation_map_path = args.relation_to_id
    else:
        relation_map_path = test_dir / 'relation_to_id.tsv'

    if args.node_name_dict:
        node_name_dict_path = args.node_name_dict
    else:
        node_name_dict_path = test_dir / 'node_name_dict.txt'

    # Load test triples using the SAME entity/relation mappings from training
    logger.info(f"Loading entity mappings from {entity_map_path}")
    logger.info(f"Loading relation mappings from {relation_map_path}")

    test_triples = TriplesFactory.from_path(
        path=args.test,
        entity_to_id=entity_map_path,
        relation_to_id=relation_map_path
    )
    logger.info(f"Loaded {test_triples.num_triples} test triples")
    logger.info(f"Number of entities: {test_triples.num_entities}")
    logger.info(f"Number of relations: {test_triples.num_relations}")

    # Load entity names
    idx_to_name = load_node_names(str(node_name_dict_path))

    # Load the model checkpoint
    checkpoint = torch.load(model_file, map_location='cpu')

    # Determine if this is a state_dict only or a full checkpoint
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Full checkpoint with metadata
        state_dict = checkpoint['model_state_dict']
        embedding_dim = checkpoint.get('config', {}).get('embedding_dim', 200)
        output_channels = checkpoint.get('config', {}).get('output_channels', 32)
        logger.info(f"Loaded checkpoint with embedding_dim={embedding_dim}")
    elif isinstance(checkpoint, dict):
        # Just a state_dict
        state_dict = checkpoint
        # Try to infer embedding_dim from state_dict
        if 'entity_embeddings.weight' in state_dict:
            embedding_dim = state_dict['entity_embeddings.weight'].shape[1]
        else:
            embedding_dim = 200  # default
        output_channels = 32  # default
        logger.info(f"Loaded state dict, inferred embedding_dim={embedding_dim}")
    else:
        logger.error(f"Unknown checkpoint format")
        return

    # Create model with correct architecture
    logger.info(f"Creating ConvE model...")
    model = ConvE(
        triples_factory=test_triples,
        embedding_dim=embedding_dim,
        output_channels=output_channels
    )

    # Load state dict into model
    model.load_state_dict(state_dict)
    model.eval()
    logger.info(f"Model loaded successfully")

    # Create evaluator
    evaluator = DetailedEvaluator(
        model=model,
        filter_triples=False,  # No filtering needed for score-only
        device=args.device,
        use_sigmoid=args.use_sigmoid
    )

    score_type = "probabilities (0-1)" if args.use_sigmoid else "logits (can be negative)"
    logger.info(f"Scoring mode: {score_type}")

    # Score all test triples WITHOUT computing rankings
    logger.info("Scoring test triples (this will be FAST - no ranking computation)...")
    results = evaluator.score_dataset(
        test_triples=test_triples,
        output_path=None,  # We'll save ourselves with entity names
        include_labels=True  # Get CURIE labels
    )

    # Add entity names to results
    if idx_to_name:
        logger.info("Adding entity names to results...")
        for result in results:
            result['head_name'] = idx_to_name.get(result['head_id'], result.get('head_label', 'UNKNOWN'))
            result['tail_name'] = idx_to_name.get(result['tail_id'], result.get('tail_label', 'UNKNOWN'))
    else:
        # If no node_name_dict, use CURIE as name
        logger.warning("No node_name_dict found - using CURIE IDs as names")
        for result in results:
            result['head_name'] = result.get('head_label', 'UNKNOWN')
            result['tail_name'] = result.get('tail_label', 'UNKNOWN')

    # Save results with entity names
    logger.info(f"Saving results to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    # Save as CSV with all columns
    csv_path = args.output.replace('.json', '.csv')
    df = pd.DataFrame(results)

    # Define column order: index, CURIE, name for head/tail, plus relation and score
    column_order = [
        'head_id', 'head_label', 'head_name',
        'relation_id', 'relation_label',
        'tail_id', 'tail_label', 'tail_name',
        'score'
    ]

    # Only keep columns that exist
    column_order = [col for col in column_order if col in df.columns]
    df = df[column_order]
    df.to_csv(csv_path, index=False)

    logger.info(f"✓ Done! Scored {len(results)} triples")
    logger.info(f"✓ Results saved to {args.output}")
    logger.info(f"✓ CSV saved to {csv_path}")

    # Show some example scores
    logger.info("\nExample scores (first 5 triples):")
    for i, result in enumerate(results[:5]):
        if 'head_name' in result:
            logger.info(f"  {i+1}. {result['head_name']} --[{result['relation_label']}]--> {result['tail_name']}")
            logger.info(f"      Score: {result['score']:.4f}")
            logger.info(f"      IDs: h={result['head_id']} ({result['head_label']}), r={result['relation_id']}, t={result['tail_id']} ({result['tail_label']})")
        elif 'head_label' in result:
            logger.info(f"  {i+1}. {result['head_label']} --[{result['relation_label']}]--> {result['tail_label']}")
            logger.info(f"      Score: {result['score']:.4f} (IDs: h={result['head_id']}, r={result['relation_id']}, t={result['tail_id']})")
        else:
            logger.info(f"  {i+1}. (h={result['head_id']}, r={result['relation_id']}, t={result['tail_id']}) → score={result['score']:.4f}")


if __name__ == '__main__':
    main()
