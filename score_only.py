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
import pickle

import pandas as pd
import torch
from pykeen.triples import TriplesFactory
from pykeen.pipeline import PipelineResult
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

    args = parser.parse_args()

    # Load model
    logger.info(f"Loading model from {args.model_dir}...")
    model_path = Path(args.model_dir) / 'trained_model.pkl'
    #model = torch.load(args.model_dir)
    model = PipelineResult.from_directory(args.model_dir)
    logger.info(f"Model loaded: {model}")

    # Load test triples
    logger.info(f"Loading test triples from {args.test}...")
    test_dir = Path(args.test).parent

    # Determine entity/relation mapping paths
    if args.entity_to_id:
        entity_map_path = args.entity_to_id
    else:
        entity_map_path = test_dir / 'entity_to_id.tsv'
    entity_to_id = {}
    with open(entity_map_path, "r", encoding="utf-8") as f:
        for line in f:
            key, value = line.strip().split("\t")
            entity_to_id[key] = value
    
    if args.relation_to_id:
        relation_map_path = args.relation_to_id
    else:
        relation_map_path = test_dir / 'relation_to_id.tsv'
    relation_to_id = {}
    with open(relation_map_path, "r", encoding="utf-8") as f:
        for line in f:
            key, value = line.strip().split("\t")
            relation_to_id[key] = value


    if args.node_name_dict:
        node_name_dict_path = args.node_name_dict
    else:
        node_name_dict_path = test_dir / 'node_name_dict.txt'

    test_triples = TriplesFactory.from_path(
        path=args.test,
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id
    )
    logger.info(f"Loaded {(test_triples.num_triples)} test triples")

    # Load entity names
    idx_to_name = load_node_names(str(node_name_dict_path))

    # Create evaluator
    evaluator = DetailedEvaluator(
        model=model.model,
        filter_triples=False,  # No filtering needed for score-only
        device=args.device
    )

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
