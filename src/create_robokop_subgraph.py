# Given a base edges.jsonl from robokop, filter it and format the results to make the correct input for nn-geometric
# The input format is a tab-delimited file of subject\tpredicate\object\n.
import os
import argparse
import logging
from pathlib import Path

import jsonlines
import json

# Configure logger
logger = logging.getLogger(__name__)

def remove_subclass_and_cid(edge, typemap):
    if edge["predicate"] == "biolink:subclass_of":
        return True
    if edge["subject"].startswith("CAID"):
        return True
    return False

def check_accepted(edge, typemap, accepted):
    subj = edge["subject"]
    obj = edge["object"]
    subj_types = typemap.get(subj, set())
    obj_types = typemap.get(obj, set())
    for acc in accepted:
        if acc[0] in subj_types and acc[1] in obj_types:
            return False
        if acc[1] in subj_types and acc[0] in obj_types:
            return False
    return True

def check_remove(edge, typemap, remove):
    subj = edge["subject"]
    obj = edge["object"]
    subj_types = typemap.get(subj, set())
    obj_types = typemap.get(obj, set())
    for acc in remove:
        if acc[0] in subj_types and acc[1] in obj_types:
            return True
        if acc[1] in subj_types and acc[0] in obj_types:
            return True
    return False

def remove_CD(edge, typemap):
    remove = [ ("biolink:ChemicalEntity", "biolink:DiseaseOrPhenotypicFeature"),
               ("biolink:DiseaseOrPhenotypicFeature", "biolink:ChemicalEntity")
              ]
    return check_remove(edge, typemap, remove)

def dont_remove(edge, typemap):
    return False

def keep_CD(edge, typemap):
    # return True if you want to filter this edge out
    # We want to keep edges between chemicals and genes, between genes and disease, and between chemicals and diseases
    # Unfortunately this means that we need a type map... Dangit
    if edge["predicate"] == "biolink:subclass_of":
        return True
    accepted = [ ("biolink:ChemicalEntity", "biolink:DiseaseOrPhenotypicFeature") ]
    return check_accepted(edge, typemap, accepted)

def keep_CCGGDD(edge, typemap):
    # return True if you want to filter this edge out
    # We want to keep edges between chemicals and genes, between genes and disease, and between chemicals and diseases
    # Unfortunately this means that we need a type map... Dangit
    if edge["predicate"] == "biolink:subclass_of":
        return True
    accepted = [ ("biolink:ChemicalEntity", "biolink:DiseaseOrPhenotypicFeature"),
                ("biolink:ChemicalEntity", "biolink:ChemicalEntity"),
                ("biolink:ChemicalEntity", "biolink:Gene"),
                ("biolink:Gene", "biolink:Gene"),
                ("biolink:DiseaseOrPhenotypicFeature", "biolink:DiseaseOrPhenotypicFeature"),
                ("biolink:Gene", "biolink:DiseaseOrPhenotypicFeature")]
    return check_accepted(edge, typemap, accepted)

def keep_CGGD(edge, typemap):
    # return True if you want to filter this edge out
    # We want to keep edges between chemicals and genes, between genes and disease, and between chemicals and diseases
    # Unfortunately this means that we need a type map... Dangit
    if edge["predicate"] == "biolink:subclass_of":
        return True
    accepted = [ ("biolink:ChemicalEntity", "biolink:DiseaseOrPhenotypicFeature"),
                ("biolink:ChemicalEntity", "biolink:Gene"),
                ("biolink:Gene", "biolink:Gene"),
                ("biolink:Gene", "biolink:DiseaseOrPhenotypicFeature")]
    return check_accepted(edge, typemap, accepted)

def keep_CGGD_alltreat(edge, typemap):
    # return True if you want to filter this edge out
    # We want to keep edges between chemicals and genes, between genes and disease, and between chemicals and diseases
    # Unfortunately this means that we need a type map... Dangit
    if edge["predicate"] == "biolink:treats":
        return False
    accepted = [ ("biolink:ChemicalEntity", "biolink:DiseaseOrPhenotypicFeature"),
                ("biolink:ChemicalEntity", "biolink:GeneOrGeneProduct"),
                ("biolink:GeneOrGeneProduct", "biolink:GeneOrGeneProduct"),
                ("biolink:GeneOrGeneProduct", "biolink:DiseaseOrPhenotypicFeature")]
    return check_accepted(edge, typemap, accepted)

def pred_trans(edge, edge_map):
    edge_key = {"predicate": edge["predicate"]}
    edge_key["subject_aspect_qualifier"] = edge.get("subject_aspect_qualifier", "")
    edge_key["object_aspect_qualifier"] = edge.get("object_aspect_qualifier", "")
    edge_key["subject_direction_qualifier"] = edge.get("subject_direction_qualifier", "")
    edge_key["object_direction_qualifier"] = edge.get("object_direction_qualifier", "")
    edge_key_string = json.dumps(edge_key, sort_keys=True)
    if edge_key_string not in edge_map:
        edge_map[edge_key_string] = f"predicate:{len(edge_map)}"
    return edge_map[edge_key_string]


def dump_edge_map(edge_map, outdir):
    output_file=f"{outdir}/edge_map.json"
    logger.info(f"Writing edge map to {output_file}")
    with open(output_file, "w") as writer:
        json.dump(edge_map, writer, indent=2)
    logger.info(f"Edge map written with {len(edge_map)} unique predicates")

def create_robokop_input(node_file="robokop/nodes.jsonl", edges_file="robokop/edges.jsonl", style="original", outdir=None):
    # Determine output directory
    if outdir is None:
        outdir = f"robokop/{style}"

    output_file = f"{outdir}/rotorobo.txt"

    logger.info("=" * 80)
    logger.info("Configuration:")
    logger.info(f"  Node file: {node_file}")
    logger.info(f"  Edges file: {edges_file}")
    logger.info(f"  Style: {style}")
    logger.info(f"  Output directory: {outdir}")
    logger.info(f"  Output file: {output_file}")
    logger.info("=" * 80)

    # Select filtering strategy based on style
    if style == "original":
        # This filters the edges by
        # 1) removing all subclass_of and
        # 2) removing all edges with a subject that starts with "CAID"
        remove_edge = remove_subclass_and_cid
        logger.info("Using 'original' style: removing subclass_of and CAID edges")
    elif style == "CD":
        # No subclasses
        # only chemical/disease edges
        remove_edge = keep_CD
        logger.info("Using 'CD' style: keeping only Chemical-Disease edges")
    elif style == "CCGGDD":
        # No subclasses
        # only chemical/disease edges and disease/disease edges
        remove_edge = keep_CCGGDD
        logger.info("Using 'CCGGDD' style: keeping Chemical-Chemical, Gene-Gene, Disease-Disease edges")
    elif style == "CGGD":
        # No subclasses
        # only chemical/disease edges and disease/disease edges
        remove_edge = keep_CGGD
        logger.info("Using 'CGGD' style: keeping Chemical-Gene-Disease edges")
    elif style == "rCD":
        remove_edge = remove_CD
        logger.info("Using 'rCD' style: removing Chemical-Disease edges")
    elif style == "keepall":
        remove_edge = dont_remove
        logger.info("Using 'keepall' style: keeping all edges")
    elif style == "CGGD_alltreat":
        remove_edge = keep_CGGD_alltreat
        logger.info("Using 'CGGD_alltreat' style: keeping CGGD edges plus all 'treats' relationships")
    else:
        logger.error(f"Unknown style: {style}")
        logger.error("Valid styles: original, CD, CCGGDD, CGGD, rCD, keepall, CGGD_alltreat")
        raise ValueError(f"Unknown style: {style}")

    # Create output directory if it doesn't exist
    if not os.path.exists(outdir):
        logger.info(f"Creating output directory: {outdir}")
        os.makedirs(outdir)

    # Build type map from nodes file
    logger.info(f"Reading nodes from {node_file}...")
    type_map = {}
    node_count = 0
    with jsonlines.open(node_file) as reader:
        for node in reader:
            type_map[node["id"]] = set(node["category"])
            node_count += 1
            if node_count % 10000 == 0:
                logger.debug(f"Processed {node_count} nodes...")

    logger.info(f"Type map created with {len(type_map)} nodes")

    # Process edges
    logger.info(f"Processing edges from {edges_file}...")
    edge_map = {}
    total_edges = 0
    filtered_edges = 0
    kept_edges = 0

    with jsonlines.open(edges_file) as reader:
        with open(output_file, "w") as writer:
            for edge in reader:
                total_edges += 1
                if total_edges % 100000 == 0:
                    logger.info(f"Processed {total_edges} edges, kept {kept_edges}, filtered {filtered_edges}")

                if remove_edge(edge, type_map):
                    filtered_edges += 1
                    continue

                writer.write(f"{edge['subject']}\t{pred_trans(edge,edge_map)}\t{edge['object']}\n")
                kept_edges += 1

    logger.info(f"Edge processing complete:")
    logger.info(f"  Total edges processed: {total_edges}")
    logger.info(f"  Edges kept: {kept_edges}")
    logger.info(f"  Edges filtered: {filtered_edges}")
    logger.info(f"  Filter rate: {(filtered_edges/total_edges*100):.2f}%")

    dump_edge_map(edge_map, outdir)
    logger.info(f"Subgraph creation complete! Output written to {output_file}")

def setup_logging(log_level):
    """Configure logging with the specified level."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Create ROBOKOP subgraph by filtering edges based on node types and predicates.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available styles:
  original       - Remove subclass_of and CAID edges
  CD             - Keep only Chemical-Disease edges
  CCGGDD         - Keep Chemical-Chemical, Gene-Gene, Disease-Disease edges
  CGGD           - Keep Chemical-Gene-Disease edges
  rCD            - Remove Chemical-Disease edges
  keepall        - Keep all edges (no filtering)
  CGGD_alltreat  - Keep CGGD edges plus all 'treats' relationships

Examples:
  python create_robokop_subgraph.py --style CGGD_alltreat
  python create_robokop_subgraph.py --node-file data/nodes.jsonl --edges-file data/edges.jsonl --style CD --outdir output/cd_graph
  python create_robokop_subgraph.py --style keepall --log-level DEBUG
        """
    )

    parser.add_argument(
        '--style',
        type=str,
        default='CGGD_alltreat',
        choices=['original', 'CD', 'CCGGDD', 'CGGD', 'rCD', 'keepall', 'CGGD_alltreat'],
        help='Filtering style to apply (default: CGGD_alltreat)'
    )

    parser.add_argument(
        '--node-file',
        type=str,
        default='robokop/nodes.jsonl',
        help='Path to the nodes JSONL file (default: robokop/nodes.jsonl)'
    )

    parser.add_argument(
        '--edges-file',
        type=str,
        default='robokop/edges.jsonl',
        help='Path to the edges JSONL file (default: robokop/edges.jsonl)'
    )

    parser.add_argument(
        '--outdir',
        type=str,
        default=None,
        help='Output directory (default: robokop/{style})'
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

    logger.info("Starting ROBOKOP subgraph creation")
    logger.info(f"Python executable: {os.sys.executable}")
    logger.info(f"Working directory: {os.getcwd()}")

    try:
        # Validate input files exist
        if not os.path.exists(args.node_file):
            logger.error(f"Node file not found: {args.node_file}")
            return 1

        if not os.path.exists(args.edges_file):
            logger.error(f"Edges file not found: {args.edges_file}")
            return 1

        # Create subgraph
        create_robokop_input(
            node_file=args.node_file,
            edges_file=args.edges_file,
            style=args.style,
            outdir=args.outdir
        )

        logger.info("All processing complete!")
        return 0

    except Exception as e:
        logger.exception(f"Error occurred during processing: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
