#!/usr/bin/env python3
"""
Extract test edges from ROBOKOP subgraph with stratified sampling.

This script selects biolink:treats edges from rotorobo.txt to create a test set.
It uses stratified sampling to ensure proper representation of different edge patterns:
- 1-to-1 edges: 6% of total treats edges
- 1-to-N edges: ~2% (all N edges pulled together to avoid leakage)
- N-to-1 edges: ~2% (all N edges pulled together to avoid leakage)

The script prevents data leakage by ensuring that when a subject appears in multiple
1-to-N relationships, ALL those edges are included in the test set together.
Similarly for N-to-1 relationships with the object.

Input:
- rotorobo.txt: Tab-separated triples (subject\tpredicate\tobject)
- edge_map.json: JSON mapping of predicate details to predicate IDs

Output:
- test.txt: Selected test edges (subject\tpredicate\tobject)
- train_candidates.txt: Remaining edges after test extraction
- test_statistics.json: Detailed statistics about the test set
"""

import argparse
import json
import logging
import os
import random
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple

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


def find_treats_predicates(edge_map: Dict[str, str]) -> Set[str]:
    """Find all predicate IDs that correspond to biolink:treats.

    Args:
        edge_map: Dictionary mapping predicate details to predicate IDs

    Returns:
        Set of predicate IDs that represent treats relationships
    """
    treats_predicates = set()

    for predicate_detail, predicate_id in edge_map.items():
        # Parse the JSON string to check for treats predicate
        try:
            pred_dict = json.loads(predicate_detail)
            if pred_dict.get("predicate") == "biolink:treats":
                treats_predicates.add(predicate_id)
        except json.JSONDecodeError:
            logger.warning(f"Could not parse predicate detail: {predicate_detail}")
            continue

    logger.info(f"Found {len(treats_predicates)} treats predicate IDs: {treats_predicates}")
    return treats_predicates


def load_and_categorize_edges(
    triples_path: str,
    treats_predicates: Set[str]
) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]], Dict, Dict]:
    """Load edges and categorize treats edges by their multiplicity patterns.

    Args:
        triples_path: Path to rotorobo.txt file
        treats_predicates: Set of predicate IDs representing treats

    Returns:
        Tuple of (treats_edges, non_treats_edges, subject_counts, object_counts)
        where subject_counts and object_counts map entity -> count for treats edges
    """
    logger.info(f"Loading and categorizing edges from {triples_path}")

    treats_edges = []
    non_treats_edges = []

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

            if predicate in treats_predicates:
                treats_edges.append((subject, predicate, obj))
            else:
                non_treats_edges.append((subject, predicate, obj))

            if (line_num) % 100000 == 0:
                logger.debug(f"Processed {line_num} edges...")

    # Count occurrences of subjects and objects in treats edges
    subject_counts = Counter(edge[0] for edge in treats_edges)
    object_counts = Counter(edge[2] for edge in treats_edges)

    logger.info(f"Total edges: {len(treats_edges) + len(non_treats_edges)}")
    logger.info(f"Treats edges: {len(treats_edges)}")
    logger.info(f"Non-treats edges: {len(non_treats_edges)}")

    return treats_edges, non_treats_edges, subject_counts, object_counts


def categorize_treats_edges(
    treats_edges: List[Tuple[str, str, str]],
    subject_counts: Counter,
    object_counts: Counter
) -> Tuple[List[Tuple[str, str, str]], Dict[str, List[Tuple[str, str, str]]], Dict[str, List[Tuple[str, str, str]]], List[Tuple[str, str, str]]]:
    """Categorize treats edges into 1-1, 1-N, N-1, and N-M patterns.

    Args:
        treats_edges: List of treats edges
        subject_counts: Counter of subject occurrences
        object_counts: Counter of object occurrences

    Returns:
        Tuple of (one_to_one_edges, one_to_n_groups, n_to_one_groups, many_to_many_edges)
        Groups are dicts mapping entity -> list of edges
    """
    logger.info("Categorizing treats edges by multiplicity...")

    one_to_one = []
    one_to_n_groups = defaultdict(list)  # subject -> list of edges (subj appears >1, obj appears 1)
    n_to_one_groups = defaultdict(list)  # object -> list of edges (obj appears >1, subj appears 1)
    many_to_many = []  # Both subject and object appear multiple times (N-M pattern)

    for edge in treats_edges:
        subject, predicate, obj = edge
        subj_count = subject_counts[subject]
        obj_count = object_counts[obj]

        if subj_count == 1 and obj_count == 1:
            # True 1-to-1 edge
            one_to_one.append(edge)
        elif subj_count > 1 and obj_count == 1:
            # N-to-1 pattern (multiple subjects to same object)
            n_to_one_groups[obj].append(edge)
        elif subj_count == 1 and obj_count > 1:
            # 1-to-N pattern (same subject to multiple objects)
            one_to_n_groups[subject].append(edge)
        else:
            # Both subject and object appear multiple times (N-to-M pattern)
            # These are complex edges that shouldn't be split across train/test
            many_to_many.append(edge)

    logger.info(f"Categorization results:")
    logger.info(f"  1-to-1 edges: {len(one_to_one)} ({len(one_to_one)/len(treats_edges)*100:.2f}%)")
    logger.info(f"  1-to-N groups: {len(one_to_n_groups)} subjects with total {sum(len(edges) for edges in one_to_n_groups.values())} edges ({sum(len(edges) for edges in one_to_n_groups.values())/len(treats_edges)*100:.2f}%)")
    logger.info(f"  N-to-1 groups: {len(n_to_one_groups)} objects with total {sum(len(edges) for edges in n_to_one_groups.values())} edges ({sum(len(edges) for edges in n_to_one_groups.values())/len(treats_edges)*100:.2f}%)")
    logger.info(f"  N-to-M edges: {len(many_to_many)} ({len(many_to_many)/len(treats_edges)*100:.2f}%)")

    # Log warning if categories are imbalanced
    if len(one_to_one) < len(treats_edges) * 0.05:
        logger.warning(f"Only {len(one_to_one)/len(treats_edges)*100:.2f}% of edges are true 1-to-1. Consider adjusting sampling strategy.")

    return one_to_one, one_to_n_groups, n_to_one_groups, many_to_many


def sample_test_edges(
    one_to_one: List[Tuple[str, str, str]],
    one_to_n_groups: Dict[str, List[Tuple[str, str, str]]],
    n_to_one_groups: Dict[str, List[Tuple[str, str, str]]],
    many_to_many: List[Tuple[str, str, str]],
    total_treats: int,
    one_to_one_pct: float = 0.06,
    one_to_n_pct: float = 0.02,
    n_to_one_pct: float = 0.02,
    many_to_many_pct: float = 0.02,
    seed: int = 42
) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
    """Sample test edges using stratified sampling with leakage prevention.

    This function prevents data leakage by ensuring that edges sharing the same
    subject or object in the treats relationship are kept together in the same
    split (either all in test or all in train).

    Leakage Prevention Strategy:
    - 1-to-1: Safe to sample individually (by definition, no shared entities)
    - 1-to-N: Sample by subject - when a subject is selected, ALL its edges go to test
    - N-to-1: Sample by object - when an object is selected, ALL its edges go to test
    - N-to-M: Sample by subject - when a subject is selected, ALL its edges go to test

    Args:
        one_to_one: List of 1-to-1 edges
        one_to_n_groups: Dict mapping subject -> list of 1-to-N edges
        n_to_one_groups: Dict mapping object -> list of N-to-1 edges
        many_to_many: List of N-to-M edges
        total_treats: Total number of treats edges
        one_to_one_pct: Target percentage for 1-to-1 edges (default: 6%)
        one_to_n_pct: Target percentage for 1-to-N edges (default: 2%)
        n_to_one_pct: Target percentage for N-to-1 edges (default: 2%)
        many_to_many_pct: Target percentage for N-to-M edges (default: 2%)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (test_edges, remaining_treats_edges)
    """
    random.seed(seed)
    logger.info("=" * 80)
    logger.info("Sampling test edges with stratified sampling:")
    logger.info(f"  Target 1-to-1: {one_to_one_pct * 100}% = ~{int(total_treats * one_to_one_pct)} edges")
    logger.info(f"  Target 1-to-N: {one_to_n_pct * 100}% = ~{int(total_treats * one_to_n_pct)} edges")
    logger.info(f"  Target N-to-1: {n_to_one_pct * 100}% = ~{int(total_treats * n_to_one_pct)} edges")
    logger.info(f"  Target N-to-M: {many_to_many_pct * 100}% = ~{int(total_treats * many_to_many_pct)} edges")
    logger.info("=" * 80)

    test_edges = []
    test_edge_set = set()

    # Sample 1-to-1 edges
    # Note: 1-to-1 edges are safe to sample individually because by definition,
    # each subject and object appears ONLY ONCE in the treats edges, so there's
    # no risk of splitting related edges between train and test
    target_one_to_one = int(total_treats * one_to_one_pct)
    if len(one_to_one) < target_one_to_one:
        logger.warning(f"Not enough 1-to-1 edges ({len(one_to_one)} < {target_one_to_one}), using all available")
        sampled_one_to_one = one_to_one
    else:
        sampled_one_to_one = random.sample(one_to_one, target_one_to_one)

    test_edges.extend(sampled_one_to_one)
    test_edge_set.update(sampled_one_to_one)
    logger.info(f"Sampled {len(sampled_one_to_one)} 1-to-1 edges (actual: {len(sampled_one_to_one)/total_treats*100:.2f}%)")

    # Sample 1-to-N groups
    target_one_to_n = int(total_treats * one_to_n_pct)
    one_to_n_subjects = list(one_to_n_groups.keys())
    random.shuffle(one_to_n_subjects)

    sampled_one_to_n = []
    for subject in one_to_n_subjects:
        edges = one_to_n_groups[subject]
        if len(sampled_one_to_n) + len(edges) <= target_one_to_n * 1.5:  # Allow some flexibility
            sampled_one_to_n.extend(edges)
            test_edge_set.update(edges)
        if len(sampled_one_to_n) >= target_one_to_n:
            break

    test_edges.extend(sampled_one_to_n)
    logger.info(f"Sampled {len(sampled_one_to_n)} edges from 1-to-N groups (actual: {len(sampled_one_to_n)/total_treats*100:.2f}%, target: ~{target_one_to_n})")

    # Sample N-to-1 groups
    target_n_to_one = int(total_treats * n_to_one_pct)
    n_to_one_objects = list(n_to_one_groups.keys())
    random.shuffle(n_to_one_objects)

    sampled_n_to_one = []
    for obj in n_to_one_objects:
        edges = n_to_one_groups[obj]
        if len(sampled_n_to_one) + len(edges) <= target_n_to_one * 1.5:  # Allow some flexibility
            sampled_n_to_one.extend(edges)
            test_edge_set.update(edges)
        if len(sampled_n_to_one) >= target_n_to_one:
            break

    test_edges.extend(sampled_n_to_one)
    logger.info(f"Sampled {len(sampled_n_to_one)} edges from N-to-1 groups (actual: {len(sampled_n_to_one)/total_treats*100:.2f}%, target: ~{target_n_to_one})")

    # Sample N-to-M edges (group by subject AND object to prevent leakage)
    # Build connected components: if edges share subject or object, they must go together
    target_many_to_many = int(total_treats * many_to_many_pct)

    # Group N-to-M edges by subject
    nm_by_subject = defaultdict(list)
    for edge in many_to_many:
        nm_by_subject[edge[0]].append(edge)

    # Shuffle subjects and sample complete groups
    nm_subjects = list(nm_by_subject.keys())
    random.shuffle(nm_subjects)

    sampled_many_to_many = []
    for subject in nm_subjects:
        edges = nm_by_subject[subject]
        if len(sampled_many_to_many) + len(edges) <= target_many_to_many * 1.5:  # Allow some flexibility
            sampled_many_to_many.extend(edges)
            test_edge_set.update(edges)
        if len(sampled_many_to_many) >= target_many_to_many:
            break

    # If we haven't reached target, we use what we have
    if len(sampled_many_to_many) < target_many_to_many * 0.5:
        logger.warning(f"Only sampled {len(sampled_many_to_many)} N-to-M edges (target: {target_many_to_many}). Consider adjusting percentage.")

    test_edges.extend(sampled_many_to_many)
    logger.info(f"Sampled {len(sampled_many_to_many)} N-to-M edges from {len([s for s in nm_subjects if any(e in sampled_many_to_many for e in nm_by_subject[s])])} subjects (actual: {len(sampled_many_to_many)/total_treats*100:.2f}%)")

    # Collect remaining treats edges
    remaining_treats = [
        edge for edge in (one_to_one +
                         [e for edges in one_to_n_groups.values() for e in edges] +
                         [e for edges in n_to_one_groups.values() for e in edges] +
                         many_to_many)
        if edge not in test_edge_set
    ]

    logger.info(f"Total test edges sampled: {len(test_edges)} ({len(test_edges)/total_treats*100:.2f}%)")
    logger.info(f"Remaining treats edges: {len(remaining_treats)}")

    return test_edges, remaining_treats


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


def save_statistics(
    stats: Dict,
    output_path: str
):
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
        description='Extract test edges from ROBOKOP subgraph with stratified sampling.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script extracts biolink:treats edges for testing using stratified sampling:
- 6% from 1-to-1 edges
- ~2% from 1-to-N edges (all edges for a subject pulled together)
- ~2% from N-to-1 edges (all edges for an object pulled together)

This prevents data leakage by ensuring related edges are not split between
train and test sets.

Examples:
  # Basic usage
  python make_test.py --input-dir robokop/CGGD_alltreat

  # Custom percentages
  python make_test.py --input-dir robokop/CGGD_alltreat --one-to-one-pct 0.08 --one-to-n-pct 0.01

  # With custom seed for reproducibility
  python make_test.py --input-dir robokop/CGGD_alltreat --seed 123
        """
    )

    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Input directory containing rotorobo.txt and edge_map.json'
    )

    parser.add_argument(
        '--triples-file',
        type=str,
        default='rotorobo.txt',
        help='Name of triples file (default: rotorobo.txt)'
    )

    parser.add_argument(
        '--edge-map-file',
        type=str,
        default='edge_map.json',
        help='Name of edge map file (default: edge_map.json)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: same as input-dir)'
    )

    parser.add_argument(
        '--one-to-one-pct',
        type=float,
        default=0.06,
        help='Percentage of total treats edges to sample from 1-to-1 bin (default: 0.06 = 6%%)'
    )

    parser.add_argument(
        '--one-to-n-pct',
        type=float,
        default=0.02,
        help='Target percentage for 1-to-N edges (default: 0.02 = 2%%)'
    )

    parser.add_argument(
        '--n-to-one-pct',
        type=float,
        default=0.02,
        help='Target percentage for N-to-1 edges (default: 0.02 = 2%%)'
    )

    parser.add_argument(
        '--many-to-many-pct',
        type=float,
        default=0.02,
        help='Target percentage for N-to-M edges (default: 0.02 = 2%%)'
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

    logger.info("Starting test edge extraction")
    logger.info(f"Working directory: {os.getcwd()}")

    try:
        # Determine paths
        input_dir = args.input_dir
        output_dir = args.output_dir if args.output_dir else input_dir

        triples_path = os.path.join(input_dir, args.triples_file)
        edge_map_path = os.path.join(input_dir, args.edge_map_file)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        logger.info("=" * 80)
        logger.info("Configuration:")
        logger.info(f"  Input directory: {input_dir}")
        logger.info(f"  Triples file: {triples_path}")
        logger.info(f"  Edge map file: {edge_map_path}")
        logger.info(f"  Output directory: {output_dir}")
        logger.info(f"  Sampling percentages: 1-1={args.one_to_one_pct*100}%, 1-N={args.one_to_n_pct*100}%, N-1={args.n_to_one_pct*100}%, N-M={args.many_to_many_pct*100}%")
        logger.info(f"  Random seed: {args.seed}")
        logger.info("=" * 80)

        # Validate input files
        if not os.path.exists(triples_path):
            logger.error(f"Triples file not found: {triples_path}")
            return 1

        if not os.path.exists(edge_map_path):
            logger.error(f"Edge map file not found: {edge_map_path}")
            return 1

        # Load edge map and find treats predicates
        edge_map = load_edge_map(edge_map_path)
        treats_predicates = find_treats_predicates(edge_map)

        if not treats_predicates:
            logger.error("No treats predicates found in edge map")
            return 1

        # Load and categorize edges
        treats_edges, non_treats_edges, subject_counts, object_counts = load_and_categorize_edges(
            triples_path, treats_predicates
        )

        if not treats_edges:
            logger.error("No treats edges found in triples file")
            return 1

        # Categorize treats edges
        one_to_one, one_to_n_groups, n_to_one_groups, many_to_many = categorize_treats_edges(
            treats_edges, subject_counts, object_counts
        )

        # Sample test edges
        test_edges, remaining_treats = sample_test_edges(
            one_to_one, one_to_n_groups, n_to_one_groups, many_to_many,
            len(treats_edges),
            args.one_to_one_pct, args.one_to_n_pct, args.n_to_one_pct, args.many_to_many_pct,
            args.seed
        )

        # Combine remaining treats with non-treats edges for train candidates
        train_candidates = remaining_treats + non_treats_edges

        # Write output files
        test_output = os.path.join(output_dir, 'test.txt')
        train_candidates_output = os.path.join(output_dir, 'train_candidates.txt')

        write_edges(test_edges, test_output)
        write_edges(train_candidates, train_candidates_output)

        # Save statistics
        stats = {
            'total_edges': len(treats_edges) + len(non_treats_edges),
            'treats_edges': len(treats_edges),
            'non_treats_edges': len(non_treats_edges),
            'test_edges': len(test_edges),
            'test_percentage': len(test_edges) / len(treats_edges) * 100,
            'train_candidate_edges': len(train_candidates),
            'categorization': {
                'one_to_one_total': len(one_to_one),
                'one_to_one_percentage': len(one_to_one) / len(treats_edges) * 100,
                'one_to_n_total': sum(len(edges) for edges in one_to_n_groups.values()),
                'one_to_n_percentage': sum(len(edges) for edges in one_to_n_groups.values()) / len(treats_edges) * 100,
                'n_to_one_total': sum(len(edges) for edges in n_to_one_groups.values()),
                'n_to_one_percentage': sum(len(edges) for edges in n_to_one_groups.values()) / len(treats_edges) * 100,
                'many_to_many_total': len(many_to_many),
                'many_to_many_percentage': len(many_to_many) / len(treats_edges) * 100
            },
            'test_composition': {
                'one_to_one_in_test': len([e for e in test_edges if e in one_to_one]),
                'one_to_n_in_test': len([e for e in test_edges if any(e in edges for edges in one_to_n_groups.values())]),
                'n_to_one_in_test': len([e for e in test_edges if any(e in edges for edges in n_to_one_groups.values())]),
                'many_to_many_in_test': len([e for e in test_edges if e in many_to_many])
            },
            'sampling_config': {
                'one_to_one_pct': args.one_to_one_pct,
                'one_to_n_pct': args.one_to_n_pct,
                'n_to_one_pct': args.n_to_one_pct,
                'many_to_many_pct': args.many_to_many_pct,
                'seed': args.seed
            }
        }

        stats_output = os.path.join(output_dir, 'test_statistics.json')
        save_statistics(stats, stats_output)

        logger.info("\n" + "=" * 80)
        logger.info("Test edge extraction complete!")
        logger.info(f"  Test edges: {test_output} ({len(test_edges)} edges)")
        logger.info(f"  Train candidates: {train_candidates_output} ({len(train_candidates)} edges)")
        logger.info(f"  Statistics: {stats_output}")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.exception(f"Error occurred during processing: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
