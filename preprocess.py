"""
Data preprocessing script to convert custom triple format to PyKEEN format.

Input format:
- Triple dataset: subject\tpredicate\tobject (TSV)
- node_dict: entity\tindex (TSV)
- rel_dict: relation\tindex (TSV)
- edge_map.json: JSON mapping detailed predicates to simplified relations

Output format:
- PyKEEN compatible TSV files with entity and relation labels preserved
- Mapping files for cross-method comparison
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def load_node_dict(node_dict_path: str) -> Dict[str, int]:
    """Load entity to index mapping from node_dict file.

    Args:
        node_dict_path: Path to node_dict file

    Returns:
        Dictionary mapping entity labels to indices
    """
    node_dict = {}
    with open(node_dict_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                entity, idx = parts
                node_dict[entity] = int(idx)
    return node_dict


def load_rel_dict(rel_dict_path: str) -> Dict[str, int]:
    """Load relation to index mapping from rel_dict file.

    Args:
        rel_dict_path: Path to rel_dict file

    Returns:
        Dictionary mapping relation labels to indices
    """
    rel_dict = {}
    with open(rel_dict_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                relation, idx = parts
                rel_dict[relation] = int(idx)
    return rel_dict


def load_edge_map(edge_map_path: str) -> Dict[str, str]:
    """Load edge mapping from detailed predicates to simplified relations.

    Args:
        edge_map_path: Path to edge_map.json file

    Returns:
        Dictionary mapping detailed predicate descriptions to relation labels
    """
    with open(edge_map_path, 'r') as f:
        return json.load(f)


def load_triples(triple_path: str) -> pd.DataFrame:
    """Load triples from TSV file.

    Args:
        triple_path: Path to triple file (TSV format)

    Returns:
        DataFrame with columns: subject, relation, object
    """
    triples = pd.read_csv(triple_path, sep='\t', header=None,
                          names=['subject', 'relation', 'object'])
    return triples


def preprocess_data(
    triple_path: str,
    node_dict_path: str,
    rel_dict_path: str,
    output_path: str,
    edge_map_path: str = None,
    validate: bool = True,
    save_mappings: bool = True
) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, int]]:
    """Preprocess triple data for PyKEEN.

    Args:
        triple_path: Path to input triple file
        node_dict_path: Path to node_dict file with entity indices
        rel_dict_path: Path to rel_dict file with relation indices
        output_path: Path to save processed triples
        edge_map_path: Optional path to edge_map.json
        validate: Whether to validate indices match the dictionaries
        save_mappings: Whether to save entity/relation mappings (default: True)

    Returns:
        Tuple of (processed_triples, node_dict, rel_dict)
    """
    # Load dictionaries
    print(f"Loading node dictionary from {node_dict_path}")
    node_dict = load_node_dict(node_dict_path)
    print(f"  Found {len(node_dict)} entities")

    print(f"Loading relation dictionary from {rel_dict_path}")
    rel_dict = load_rel_dict(rel_dict_path)
    print(f"  Found {len(rel_dict)} relations")

    # Load edge map if provided
    edge_map = None
    if edge_map_path and os.path.exists(edge_map_path):
        print(f"Loading edge map from {edge_map_path}")
        edge_map = load_edge_map(edge_map_path)

    # Load triples
    print(f"Loading triples from {triple_path}")
    triples = load_triples(triple_path)
    print(f"  Loaded {len(triples)} triples")

    # Validate entities and relations exist in dictionaries
    if validate:
        print("Validating triples...")
        unknown_entities = set()
        unknown_relations = set()

        for _, row in triples.iterrows():
            if row['subject'] not in node_dict:
                unknown_entities.add(row['subject'])
            if row['object'] not in node_dict:
                unknown_entities.add(row['object'])
            if row['relation'] not in rel_dict:
                unknown_relations.add(row['relation'])

        if unknown_entities:
            print(f"  WARNING: Found {len(unknown_entities)} unknown entities")
            print(f"    First few: {list(unknown_entities)[:5]}")

        if unknown_relations:
            print(f"  WARNING: Found {len(unknown_relations)} unknown relations")
            print(f"    Relations: {list(unknown_relations)}")

    # Filter triples to only include known entities and relations
    mask = (triples['subject'].isin(node_dict.keys()) &
            triples['object'].isin(node_dict.keys()) &
            triples['relation'].isin(rel_dict.keys()))

    filtered_triples = triples[mask].copy()
    print(f"  Kept {len(filtered_triples)} valid triples (filtered {len(triples) - len(filtered_triples)})")

    # Save processed triples in PyKEEN format (TSV: head, relation, tail)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    filtered_triples.to_csv(output_path, sep='\t', header=False, index=False)
    print(f"Saved processed triples to {output_path}")

    # Save entity and relation mappings for PyKEEN (only if requested)
    if save_mappings:
        entity_map_path = output_path.replace('.txt', '_entity_to_id.tsv')
        relation_map_path = output_path.replace('.txt', '_relation_to_id.tsv')

        # Create sorted mappings by index
        entities_sorted = sorted(node_dict.items(), key=lambda x: x[1])
        relations_sorted = sorted(rel_dict.items(), key=lambda x: x[1])

        with open(entity_map_path, 'w') as f:
            for entity, idx in entities_sorted:
                f.write(f"{entity}\t{idx}\n")
        print(f"Saved entity mapping to {entity_map_path}")

        with open(relation_map_path, 'w') as f:
            for relation, idx in relations_sorted:
                f.write(f"{relation}\t{idx}\n")
        print(f"Saved relation mapping to {relation_map_path}")

    # Print statistics
    print("\nDataset Statistics:")
    print(f"  Entities: {len(node_dict)}")
    print(f"  Relations: {len(rel_dict)}")
    print(f"  Triples: {len(filtered_triples)}")
    print(f"  Avg triples per entity: {len(filtered_triples) / len(node_dict):.2f}")

    return filtered_triples, node_dict, rel_dict


def split_data(
    train_triple_path: str,
    valid_triple_path: str,
    test_triple_path: str,
    node_dict_path: str,
    rel_dict_path: str,
    output_dir: str,
    edge_map_path: str = None,
    validate: bool = True
):
    """Process and split data into train/valid/test sets for PyKEEN.

    Args:
        train_triple_path: Path to training triples
        valid_triple_path: Path to validation triples
        test_triple_path: Path to test triples
        node_dict_path: Path to node_dict file
        rel_dict_path: Path to rel_dict file
        output_dir: Directory to save processed files
        edge_map_path: Optional path to edge_map.json
        validate: Whether to validate entities and relations (default: True)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load the shared node and relation dictionaries
    print("Loading shared entity and relation mappings...")
    node_dict = load_node_dict(node_dict_path)
    rel_dict = load_rel_dict(rel_dict_path)
    print(f"  Loaded {len(node_dict)} entities")
    print(f"  Loaded {len(rel_dict)} relations")

    # Save entity and relation mappings ONCE for all splits
    entity_map_path = os.path.join(output_dir, 'entity_to_id.tsv')
    relation_map_path = os.path.join(output_dir, 'relation_to_id.tsv')

    entities_sorted = sorted(node_dict.items(), key=lambda x: x[1])
    relations_sorted = sorted(rel_dict.items(), key=lambda x: x[1])

    with open(entity_map_path, 'w') as f:
        for entity, idx in entities_sorted:
            f.write(f"{entity}\t{idx}\n")
    print(f"Saved shared entity mapping to {entity_map_path}")

    with open(relation_map_path, 'w') as f:
        for relation, idx in relations_sorted:
            f.write(f"{relation}\t{idx}\n")
    print(f"Saved shared relation mapping to {relation_map_path}")

    print("\n" + "=" * 60)
    print("Processing Training Data")
    print("=" * 60)
    train_out = os.path.join(output_dir, 'train.txt')
    train_triples, _, _ = preprocess_data(
        train_triple_path, node_dict_path, rel_dict_path,
        train_out, edge_map_path, validate, save_mappings=False
    )

    print("\n" + "=" * 60)
    print("Processing Validation Data")
    print("=" * 60)
    valid_out = os.path.join(output_dir, 'valid.txt')
    valid_triples, _, _ = preprocess_data(
        valid_triple_path, node_dict_path, rel_dict_path,
        valid_out, edge_map_path, validate, save_mappings=False
    )

    print("\n" + "=" * 60)
    print("Processing Test Data")
    print("=" * 60)
    test_out = os.path.join(output_dir, 'test.txt')
    test_triples, _, _ = preprocess_data(
        test_triple_path, node_dict_path, rel_dict_path,
        test_out, edge_map_path, validate, save_mappings=False
    )

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Train triples: {len(train_triples)}")
    print(f"Valid triples: {len(valid_triples)}")
    print(f"Test triples: {len(test_triples)}")
    print(f"Total: {len(train_triples) + len(valid_triples) + len(test_triples)}")
    print(f"\nOutput directory: {output_dir}")
    print(f"\nShared mapping files:")
    print(f"  Entity mapping: {entity_map_path}")
    print(f"  Relation mapping: {relation_map_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Preprocess knowledge graph data for PyKEEN ConvE'
    )

    parser.add_argument(
        '--train', type=str, required=True,
        help='Path to training triple file (TSV format)'
    )
    parser.add_argument(
        '--valid', type=str, required=True,
        help='Path to validation triple file (TSV format)'
    )
    parser.add_argument(
        '--test', type=str, required=True,
        help='Path to test triple file (TSV format)'
    )
    parser.add_argument(
        '--node-dict', type=str, required=True,
        help='Path to node_dict file (entity to index mapping)'
    )
    parser.add_argument(
        '--rel-dict', type=str, required=True,
        help='Path to rel_dict file (relation to index mapping)'
    )
    parser.add_argument(
        '--edge-map', type=str, default=None,
        help='Path to edge_map.json file (optional)'
    )
    parser.add_argument(
        '--output-dir', '-o', type=str, required=True,
        help='Output directory for processed files'
    )
    parser.add_argument(
        '--no-validate', action='store_true',
        help='Skip validation of entities and relations (default: validate)'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    split_data(
        train_triple_path=args.train,
        valid_triple_path=args.valid,
        test_triple_path=args.test,
        node_dict_path=args.node_dict,
        rel_dict_path=args.rel_dict,
        output_dir=args.output_dir,
        edge_map_path=args.edge_map,
        validate=not args.no_validate  # Invert no_validate to get validate
    )


if __name__ == '__main__':
    main()
