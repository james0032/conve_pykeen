#!/usr/bin/env python3
"""
Split train candidates into training and validation sets.

This script takes the train_candidates.txt file (output from make_test.py)
and splits it into train.txt (90%) and valid.txt (10%) sets.

Input:
- train_candidates.txt: Edges remaining after test extraction

Output:
- train.txt: Training edges (90% of candidates)
- valid.txt: Validation edges (10% of candidates)
- split_statistics.json: Statistics about the split
"""

import argparse
import json
import logging
import os
import random
from typing import List, Tuple

# Configure logger
logger = logging.getLogger(__name__)


def setup_logging(log_level):
    """Configure logging with the specified level."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_edges(edges_path: str) -> List[Tuple[str, str, str]]:
    """Load edges from file.

    Args:
        edges_path: Path to edges file

    Returns:
        List of (subject, predicate, object) tuples
    """
    logger.info(f"Loading edges from {edges_path}")

    edges = []
    with open(edges_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t')
            if len(parts) != 3:
                logger.warning(f"Line {line_num}: Invalid format (expected 3 columns, got {len(parts)})")
                continue

            subject, predicate, obj = parts
            edges.append((subject, predicate, obj))

            if line_num % 100000 == 0:
                logger.debug(f"Loaded {line_num} edges...")

    logger.info(f"Loaded {len(edges)} edges")
    return edges


def split_edges(
    edges: List[Tuple[str, str, str]],
    train_ratio: float = 0.9,
    seed: int = 42
) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
    """Split edges into training and validation sets.

    Args:
        edges: List of edges to split
        train_ratio: Ratio of edges for training (default: 0.9 = 90%)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_edges, valid_edges)
    """
    random.seed(seed)
    logger.info(f"Splitting {len(edges)} edges with {train_ratio*100}% train / {(1-train_ratio)*100}% valid ratio")

    # Shuffle edges
    shuffled_edges = edges.copy()
    random.shuffle(shuffled_edges)

    # Split
    split_idx = int(len(shuffled_edges) * train_ratio)
    train_edges = shuffled_edges[:split_idx]
    valid_edges = shuffled_edges[split_idx:]

    logger.info(f"Split complete:")
    logger.info(f"  Training edges: {len(train_edges)} ({len(train_edges)/len(edges)*100:.2f}%)")
    logger.info(f"  Validation edges: {len(valid_edges)} ({len(valid_edges)/len(edges)*100:.2f}%)")

    return train_edges, valid_edges


def write_edges(edges: List[Tuple[str, str, str]], output_path: str):
    """Write edges to file.

    Args:
        edges: List of (subject, predicate, object) tuples
        output_path: Path to output file
    """
    logger.info(f"Writing {len(edges)} edges to {output_path}")
    with open(output_path, 'w') as f:
        for subject, predicate, obj in edges:
            f.write(f"{subject}\t{predicate}\t{obj}\n")


def analyze_edge_statistics(
    edges: List[Tuple[str, str, str]],
    name: str
) -> dict:
    """Analyze and return statistics about edges.

    Args:
        edges: List of edges
        name: Name of the edge set (for logging)

    Returns:
        Dictionary of statistics
    """
    if not edges:
        return {
            'num_edges': 0,
            'num_entities': 0,
            'num_relations': 0
        }

    entities = set()
    relations = set()

    for subject, predicate, obj in edges:
        entities.add(subject)
        entities.add(obj)
        relations.add(predicate)

    stats = {
        'num_edges': len(edges),
        'num_entities': len(entities),
        'num_relations': len(relations)
    }

    logger.debug(f"{name} statistics:")
    logger.debug(f"  Edges: {stats['num_edges']}")
    logger.debug(f"  Unique entities: {stats['num_entities']}")
    logger.debug(f"  Unique relations: {stats['num_relations']}")

    return stats


def save_statistics(stats: dict, output_path: str):
    """Save statistics to JSON file.

    Args:
        stats: Dictionary of statistics
        output_path: Path to output JSON file
    """
    logger.info(f"Saving statistics to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Split train candidates into training and validation sets.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script takes the train_candidates.txt file (remaining edges after test
extraction) and splits it into train.txt (90%) and valid.txt (10%) for
training and validation.

Examples:
  # Basic usage
  python train_valid_split.py --input-dir robokop/CGGD_alltreat

  # Custom split ratio (80/20)
  python train_valid_split.py --input-dir robokop/CGGD_alltreat --train-ratio 0.8

  # With custom input file and seed
  python train_valid_split.py --input train_candidates.txt --output-dir output --seed 123
        """
    )

    parser.add_argument(
        '--input-dir',
        type=str,
        default=None,
        help='Input directory containing train_candidates.txt'
    )

    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Path to train candidates file (overrides --input-dir, default: train_candidates.txt in input-dir)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for train.txt and valid.txt (default: same as input-dir)'
    )

    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.9,
        help='Ratio of edges for training set (default: 0.9 = 90%%)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Logging level (default: INFO)'
    )

    return parser.parse_args()


def main():
    """Main entry point for the script."""
    args = parse_arguments()

    # Setup logging
    setup_logging(getattr(logging, args.log_level))

    logger.info("Starting train/validation split")
    logger.info(f"Working directory: {os.getcwd()}")

    try:
        # Determine input path
        if args.input:
            input_path = args.input
            input_dir = os.path.dirname(input_path)
        elif args.input_dir:
            input_dir = args.input_dir
            input_path = os.path.join(input_dir, 'train_candidates.txt')
        else:
            logger.error("Must specify either --input or --input-dir")
            return 1

        # Determine output directory
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = input_dir if input_dir else '.'

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Validate train ratio
        if not 0 < args.train_ratio < 1:
            logger.error(f"Train ratio must be between 0 and 1, got {args.train_ratio}")
            return 1

        logger.info("=" * 80)
        logger.info("Configuration:")
        logger.info(f"  Input file: {input_path}")
        logger.info(f"  Output directory: {output_dir}")
        logger.info(f"  Train ratio: {args.train_ratio*100}%")
        logger.info(f"  Valid ratio: {(1-args.train_ratio)*100}%")
        logger.info(f"  Random seed: {args.seed}")
        logger.info("=" * 80)

        # Validate input file
        if not os.path.exists(input_path):
            logger.error(f"Input file not found: {input_path}")
            return 1

        # Load edges
        edges = load_edges(input_path)

        if not edges:
            logger.error("No edges found in input file")
            return 1

        # Split edges
        train_edges, valid_edges = split_edges(edges, args.train_ratio, args.seed)

        # Write output files
        train_output = os.path.join(output_dir, 'train.txt')
        valid_output = os.path.join(output_dir, 'valid.txt')

        write_edges(train_edges, train_output)
        write_edges(valid_edges, valid_output)

        # Analyze statistics
        logger.info("\nAnalyzing dataset statistics...")
        train_stats = analyze_edge_statistics(train_edges, "Training set")
        valid_stats = analyze_edge_statistics(valid_edges, "Validation set")

        # Combined statistics
        all_entities = set()
        all_relations = set()
        for subject, predicate, obj in edges:
            all_entities.add(subject)
            all_entities.add(obj)
            all_relations.add(predicate)

        # Save statistics
        stats = {
            'total_edges': len(edges),
            'total_entities': len(all_entities),
            'total_relations': len(all_relations),
            'train': train_stats,
            'valid': valid_stats,
            'split_config': {
                'train_ratio': args.train_ratio,
                'seed': args.seed
            }
        }

        stats_output = os.path.join(output_dir, 'split_statistics.json')
        save_statistics(stats, stats_output)

        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("Dataset Statistics:")
        logger.info("=" * 80)
        logger.info(f"Total edges: {len(edges)}")
        logger.info(f"Total unique entities: {len(all_entities)}")
        logger.info(f"Total unique relations: {len(all_relations)}")
        logger.info("")
        logger.info(f"Training set:")
        logger.info(f"  Edges: {train_stats['num_edges']} ({train_stats['num_edges']/len(edges)*100:.2f}%)")
        logger.info(f"  Unique entities: {train_stats['num_entities']}")
        logger.info(f"  Unique relations: {train_stats['num_relations']}")
        logger.info("")
        logger.info(f"Validation set:")
        logger.info(f"  Edges: {valid_stats['num_edges']} ({valid_stats['num_edges']/len(edges)*100:.2f}%)")
        logger.info(f"  Unique entities: {valid_stats['num_entities']}")
        logger.info(f"  Unique relations: {valid_stats['num_relations']}")
        logger.info("=" * 80)

        logger.info("\n" + "=" * 80)
        logger.info("Train/validation split complete!")
        logger.info(f"  Training set: {train_output}")
        logger.info(f"  Validation set: {valid_output}")
        logger.info(f"  Statistics: {stats_output}")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.exception(f"Error occurred during processing: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
