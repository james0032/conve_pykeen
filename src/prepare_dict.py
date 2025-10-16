#!/usr/bin/env python3
"""
Prepare dictionary files (node_dict and rel_dict) from ROBOKOP subgraph output.

This script takes the output from create_robokop_subgraph.py (rotorobo.txt and edge_map.json)
and generates the input files needed for preprocess.py:
- node_dict: entity to index mapping (TSV format: entity\tindex)
- rel_dict: relation to index mapping (TSV format: relation\tindex)

Input:
- rotorobo.txt: Tab-separated triples (subject\tpredicate\tobject)
- edge_map.json: JSON mapping of predicate details to predicate IDs

Output:
- node_dict.txt: Entity to index mapping
- rel_dict.txt: Relation to index mapping
- Statistics about the graph
"""

import argparse
import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Dict, Set, Tuple

# Configure logger
logger = logging.getLogger(__name__)


def setup_logging(log_level):
    """Configure logging with the specified level."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_edge_map(edge_map_path: str) -> Dict[str, str]:
    """Load edge map from JSON file.

    Args:
        edge_map_path: Path to edge_map.json file

    Returns:
        Dictionary mapping predicate details to predicate IDs
    """
    logger.info(f"Loading edge map from {edge_map_path}")
    with open(edge_map_path, 'r') as f:
        edge_map = json.load(f)
    logger.info(f"Loaded {len(edge_map)} predicate mappings")
    return edge_map


def extract_entities_and_relations(
    triples_path: str
) -> Tuple[Set[str], Set[str], int]:
    """Extract unique entities and relations from triples file.

    Args:
        triples_path: Path to rotorobo.txt file

    Returns:
        Tuple of (entities_set, relations_set, triple_count)
    """
    logger.info(f"Reading triples from {triples_path}")

    entities = set()
    relations = set()
    triple_count = 0

    with open(triples_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t')
            if len(parts) != 3:
                logger.warning(f"Line {line_num}: Invalid format (expected 3 columns, got {len(parts)})")
                continue

            subject, predicate, obj = parts
            entities.add(subject)
            entities.add(obj)
            relations.add(predicate)
            triple_count += 1

            if triple_count % 100000 == 0:
                logger.info(f"Processed {triple_count} triples...")

    logger.info(f"Extraction complete:")
    logger.info(f"  Total triples: {triple_count}")
    logger.info(f"  Unique entities: {len(entities)}")
    logger.info(f"  Unique relations: {len(relations)}")

    return entities, relations, triple_count


def create_node_dict(entities: Set[str], output_path: str) -> Dict[str, int]:
    """Create node dictionary mapping entities to indices.

    Args:
        entities: Set of unique entity IDs
        output_path: Path to save node_dict.txt

    Returns:
        Dictionary mapping entity to index
    """
    logger.info("Creating node dictionary...")

    # Sort entities for consistent ordering
    sorted_entities = sorted(entities)

    node_dict = {entity: idx for idx, entity in enumerate(sorted_entities)}

    logger.info(f"Writing node dictionary to {output_path}")
    with open(output_path, 'w') as f:
        for entity, idx in node_dict.items():
            f.write(f"{entity}\t{idx}\n")

    logger.info(f"Node dictionary created with {len(node_dict)} entities")
    return node_dict


def create_rel_dict(relations: Set[str], output_path: str) -> Dict[str, int]:
    """Create relation dictionary mapping relations to indices.

    Args:
        relations: Set of unique relation IDs
        output_path: Path to save rel_dict.txt

    Returns:
        Dictionary mapping relation to index
    """
    logger.info("Creating relation dictionary...")

    # Sort relations for consistent ordering
    sorted_relations = sorted(relations)

    rel_dict = {relation: idx for idx, relation in enumerate(sorted_relations)}

    logger.info(f"Writing relation dictionary to {output_path}")
    with open(output_path, 'w') as f:
        for relation, idx in rel_dict.items():
            f.write(f"{relation}\t{idx}\n")

    logger.info(f"Relation dictionary created with {len(rel_dict)} relations")
    return rel_dict


def analyze_graph_statistics(
    triples_path: str,
    node_dict: Dict[str, int],
    rel_dict: Dict[str, int],
    edge_map: Dict[str, str] = None
):
    """Analyze and log graph statistics.

    Args:
        triples_path: Path to triples file
        node_dict: Entity to index mapping
        rel_dict: Relation to index mapping
        edge_map: Optional edge map
    """
    logger.info("Analyzing graph statistics...")

    # Count occurrences
    subject_counter = Counter()
    object_counter = Counter()
    relation_counter = Counter()

    with open(triples_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) != 3:
                continue
            subject, predicate, obj = parts
            subject_counter[subject] += 1
            object_counter[obj] += 1
            relation_counter[predicate] += 1

    # Calculate statistics
    total_triples = sum(relation_counter.values())
    avg_degree = total_triples / len(node_dict) if node_dict else 0

    logger.info("=" * 80)
    logger.info("Graph Statistics:")
    logger.info("=" * 80)
    logger.info(f"Total entities: {len(node_dict)}")
    logger.info(f"Total relations: {len(rel_dict)}")
    logger.info(f"Total triples: {total_triples}")
    logger.info(f"Average degree per entity: {avg_degree:.2f}")

    # Entity statistics
    if subject_counter:
        max_out_entity = subject_counter.most_common(1)[0]
        logger.info(f"Max out-degree entity: {max_out_entity[0]} ({max_out_entity[1]} edges)")

    if object_counter:
        max_in_entity = object_counter.most_common(1)[0]
        logger.info(f"Max in-degree entity: {max_in_entity[0]} ({max_in_entity[1]} edges)")

    # Relation statistics
    logger.info("\nTop 10 most frequent relations:")
    for relation, count in relation_counter.most_common(10):
        percentage = (count / total_triples) * 100
        logger.info(f"  {relation}: {count} ({percentage:.2f}%)")

    # Edge map statistics
    if edge_map:
        logger.info(f"\nEdge map contains {len(edge_map)} predicate mappings")
        # Show sample of edge map
        logger.info("Sample edge mappings:")
        for i, (detailed_pred, simple_pred) in enumerate(edge_map.items()):
            if i >= 5:
                break
            logger.info(f"  {simple_pred} <- {detailed_pred}")

    logger.info("=" * 80)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Prepare dictionary files from ROBOKOP subgraph output.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script processes the output from create_robokop_subgraph.py and generates
the input files needed for preprocess.py:

Input files:
  - rotorobo.txt: Tab-separated triples (subject\tpredicate\tobject)
  - edge_map.json: JSON mapping of predicate details to predicate IDs

Output files:
  - node_dict.txt: Entity to index mapping (entity\tindex)
  - rel_dict.txt: Relation to index mapping (relation\tindex)

Examples:
  # Basic usage with default input directory
  python prepare_dict.py --input-dir robokop/CGGD_alltreat

  # Specify custom input and output paths
  python prepare_dict.py --triples-file data/rotorobo.txt --output-dir output/dicts

  # With edge map and debug logging
  python prepare_dict.py --input-dir robokop/CGGD_alltreat --edge-map robokop/CGGD_alltreat/edge_map.json --log-level DEBUG
        """
    )

    parser.add_argument(
        '--input-dir',
        type=str,
        default=None,
        help='Input directory containing rotorobo.txt and edge_map.json (default: robokop/CGGD_alltreat)'
    )

    parser.add_argument(
        '--triples-file',
        type=str,
        default=None,
        help='Path to rotorobo.txt file (overrides --input-dir)'
    )

    parser.add_argument(
        '--edge-map',
        type=str,
        default=None,
        help='Path to edge_map.json file (optional)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for dictionary files (default: same as input-dir)'
    )

    parser.add_argument(
        '--node-dict-name',
        type=str,
        default='node_dict.txt',
        help='Name of node dictionary file (default: node_dict.txt)'
    )

    parser.add_argument(
        '--rel-dict-name',
        type=str,
        default='rel_dict.txt',
        help='Name of relation dictionary file (default: rel_dict.txt)'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Logging level (default: INFO)'
    )

    parser.add_argument(
        '--skip-stats',
        action='store_true',
        help='Skip detailed statistics analysis'
    )

    return parser.parse_args()


def main():
    """Main entry point for the script."""
    args = parse_arguments()

    # Setup logging
    setup_logging(getattr(logging, args.log_level))

    logger.info("Starting dictionary preparation")
    logger.info(f"Working directory: {os.getcwd()}")

    try:
        # Determine input directory and files
        if args.input_dir:
            input_dir = args.input_dir
        elif args.triples_file:
            input_dir = os.path.dirname(args.triples_file)
        else:
            input_dir = 'robokop/CGGD_alltreat'
            logger.info(f"No input directory specified, using default: {input_dir}")

        # Determine triples file path
        if args.triples_file:
            triples_file = args.triples_file
        else:
            triples_file = os.path.join(input_dir, 'rotorobo.txt')

        # Determine edge map path
        if args.edge_map:
            edge_map_file = args.edge_map
        else:
            edge_map_file = os.path.join(input_dir, 'edge_map.json')

        # Determine output directory
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = input_dir

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        logger.info("=" * 80)
        logger.info("Configuration:")
        logger.info(f"  Triples file: {triples_file}")
        logger.info(f"  Edge map file: {edge_map_file}")
        logger.info(f"  Output directory: {output_dir}")
        logger.info("=" * 80)

        # Validate input files
        if not os.path.exists(triples_file):
            logger.error(f"Triples file not found: {triples_file}")
            return 1

        # Load edge map if it exists
        edge_map = None
        if os.path.exists(edge_map_file):
            edge_map = load_edge_map(edge_map_file)
        else:
            logger.warning(f"Edge map file not found: {edge_map_file} (continuing without it)")

        # Extract entities and relations
        entities, relations, triple_count = extract_entities_and_relations(triples_file)

        if len(entities) == 0 or len(relations) == 0:
            logger.error("No entities or relations found in triples file")
            return 1

        # Create dictionaries
        node_dict_path = os.path.join(output_dir, args.node_dict_name)
        rel_dict_path = os.path.join(output_dir, args.rel_dict_name)

        node_dict = create_node_dict(entities, node_dict_path)
        rel_dict = create_rel_dict(relations, rel_dict_path)

        # Analyze statistics
        if not args.skip_stats:
            analyze_graph_statistics(triples_file, node_dict, rel_dict, edge_map)

        logger.info("\n" + "=" * 80)
        logger.info("Dictionary preparation complete!")
        logger.info(f"  Node dictionary: {node_dict_path}")
        logger.info(f"  Relation dictionary: {rel_dict_path}")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.exception(f"Error occurred during processing: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
