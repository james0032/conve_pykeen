"""
Custom evaluation module for ConvE with detailed predictions.

This module provides comprehensive evaluation metrics and per-triple predictions
for link prediction tasks.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from pykeen.evaluation import RankBasedEvaluator
from pykeen.models import Model
from pykeen.triples import CoreTriplesFactory
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DetailedEvaluator:
    """Evaluator that provides per-triple predictions and rankings.

    This evaluator extends PyKEEN's evaluation to provide:
    1. Per-triple predictions with scores
    2. Rankings for each test triple
    3. Top-k predictions for each query
    4. Filtered evaluation (removes training triples)
    """

    def __init__(
        self,
        model: Model,
        batch_size: int = 32,
        filter_triples: bool = True,
        device: Optional[str] = None,
        use_sigmoid: bool = False
    ):
        """Initialize evaluator.

        Args:
            model: Trained PyKEEN model
            batch_size: Batch size for evaluation
            filter_triples: Whether to filter known triples during ranking
            device: Device to run evaluation on
            use_sigmoid: If True, apply sigmoid to convert logits to probabilities [0, 1]
        """
        self.model = model
        self.batch_size = batch_size
        self.filter_triples = filter_triples
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_sigmoid = use_sigmoid
        self.model.to(self.device)

    def score_triple(
        self,
        head: int,
        relation: int,
        tail: int
    ) -> float:
        """Get the ConvE score for a specific triple WITHOUT computing rankings.

        This is much faster than evaluate_triple() when you only need the score
        and don't need rank, MRR, or hits@k metrics.

        Args:
            head: Head entity index
            relation: Relation index
            tail: Tail entity index

        Returns:
            Model score for the triple. If use_sigmoid=True, returns probability [0, 1].
            Otherwise returns raw logit (can be negative).
        """
        self.model.eval()

        with torch.no_grad():
            # Create batch with single triple
            hr_batch = torch.LongTensor([[head, relation]]).to(self.device)

            # Score all possible tails - returns LOGITS (raw scores, can be negative)
            all_scores = self.model.score_t(hr_batch).squeeze(0)  # Shape: (num_entities,)

            # Get the score for the true tail
            true_score = all_scores[tail]

            # Apply sigmoid to convert logit to probability if requested
            if self.use_sigmoid:
                true_score = torch.sigmoid(true_score)

            true_score = true_score.item()

        return true_score

    def evaluate_triple(
        self,
        head: int,
        relation: int,
        tail: int,
        known_triples: Optional[set] = None
    ) -> Dict[str, any]:
        """Evaluate a single triple and return detailed results INCLUDING rankings.

        This computes the full ranking which is expensive. If you only need the score,
        use score_triple() instead.

        Args:
            head: Head entity index
            relation: Relation index
            tail: Tail entity index
            known_triples: Set of known (h, r, t) triples to filter

        Returns:
            Dictionary with evaluation results including:
                - score: Model score for the triple
                - rank: Rank of the tail entity
                - reciprocal_rank: 1/rank
                - hits@k: Whether triple is in top k
                - top_predictions: Top-k predicted tails with scores
        """
        self.model.eval()

        with torch.no_grad():
            # Create batch with single triple
            hr_batch = torch.LongTensor([[head, relation]]).to(self.device)

            # Score all possible tails
            all_scores = self.model.score_t(hr_batch).squeeze(0)  # Shape: (num_entities,)

            # Get the score for the true tail - THIS IS THE CONVE SCORE
            true_score = all_scores[tail].item()

            # === RANKING COMPUTATION STARTS HERE ===
            # Everything below this line is for computing rank/hits@k metrics
            # If you only need the score, use score_triple() instead

            # Filter known triples if requested
            if self.filter_triples and known_triples is not None:
                # Set scores of known triples (except the true tail) to -inf
                for h, r, t in known_triples:
                    if h == head and r == relation and t != tail:
                        all_scores[t] = float('-inf')

            # Get ranking - EXPENSIVE: sorts all entity scores
            sorted_indices = torch.argsort(all_scores, descending=True)
            rank = (sorted_indices == tail).nonzero(as_tuple=True)[0].item() + 1

            # Get top-k predictions
            top_k = min(10, len(all_scores))
            top_scores, top_indices = torch.topk(all_scores, k=top_k)

            top_predictions = [
                {'entity': int(top_indices[i]), 'score': float(top_scores[i])}
                for i in range(top_k)
            ]

            results = {
                'head': head,
                'relation': relation,
                'tail': tail,
                'score': true_score,
                'rank': rank,
                'reciprocal_rank': 1.0 / rank,
                'hits@1': int(rank == 1),
                'hits@3': int(rank <= 3),
                'hits@5': int(rank <= 5),
                'hits@10': int(rank <= 10),
                'top_predictions': top_predictions
            }

        return results

    def score_dataset(
        self,
        test_triples: CoreTriplesFactory,
        output_path: Optional[str] = None,
        include_labels: bool = True
    ) -> List[Dict[str, any]]:
        """Get ConvE scores for all test triples WITHOUT computing rankings.

        This is MUCH faster than evaluate_dataset() when you only need scores.
        Use this when you don't need rank, MRR, or hits@k metrics.

        Args:
            test_triples: Test triples factory
            output_path: Optional path to save scores (JSON and CSV)
            include_labels: If True, include entity/relation names in addition to IDs

        Returns:
            List of dictionaries with head, relation, tail, and score
        """
        logger.info(f"Scoring {test_triples.num_triples} test triples (no ranking)...")

        # Get ID-to-label mappings if labels requested
        if include_labels:
            # Create reverse mappings: id -> label
            id_to_entity = {v: k for k, v in test_triples.entity_to_id.items()}
            id_to_relation = {v: k for k, v in test_triples.relation_to_id.items()}

        # Score each triple
        results = []
        for h, r, t in tqdm(test_triples.mapped_triples, desc="Scoring"):
            h, r, t = int(h), int(r), int(t)
            score = self.score_triple(h, r, t)

            result = {
                'head_id': h,
                'relation_id': r,
                'tail_id': t,
                'score': score
            }

            # Add labels if requested
            if include_labels:
                result['head_label'] = id_to_entity.get(h, f'UNKNOWN_{h}')
                result['relation_label'] = id_to_relation.get(r, f'UNKNOWN_{r}')
                result['tail_label'] = id_to_entity.get(t, f'UNKNOWN_{t}')

            results.append(result)

        logger.info(f"Scored {len(results)} triples")

        # Save results if path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved scores to {output_path}")

            # Also save as CSV
            csv_path = output_path.replace('.json', '.csv')
            df = pd.DataFrame(results)
            # Reorder columns for better readability
            if include_labels:
                column_order = ['head_id', 'head_label', 'relation_id', 'relation_label',
                               'tail_id', 'tail_label', 'score']
            else:
                column_order = ['head_id', 'relation_id', 'tail_id', 'score']
            df = df[column_order]
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved CSV scores to {csv_path}")

        return results

    def evaluate_dataset(
        self,
        test_triples: CoreTriplesFactory,
        training_triples: Optional[CoreTriplesFactory] = None,
        validation_triples: Optional[CoreTriplesFactory] = None,
        output_path: Optional[str] = None
    ) -> Dict[str, any]:
        """Evaluate all triples in test set WITH full ranking metrics.

        This computes ranks which is expensive. If you only need scores,
        use score_dataset() instead for much faster execution.

        Args:
            test_triples: Test triples factory
            training_triples: Training triples factory (for filtering)
            validation_triples: Validation triples factory (for filtering)
            output_path: Optional path to save detailed results

        Returns:
            Dictionary with aggregated metrics and per-triple results
        """
        logger.info(f"Evaluating {test_triples.num_triples} test triples...")

        # Build set of known triples for filtering
        known_triples = set()
        if self.filter_triples:
            if training_triples is not None:
                for h, r, t in training_triples.mapped_triples:
                    known_triples.add((int(h), int(r), int(t)))
            if validation_triples is not None:
                for h, r, t in validation_triples.mapped_triples:
                    known_triples.add((int(h), int(r), int(t)))

        # Evaluate each triple
        results = []
        for h, r, t in tqdm(test_triples.mapped_triples, desc="Evaluating"):
            h, r, t = int(h), int(r), int(t)
            result = self.evaluate_triple(h, r, t, known_triples)
            results.append(result)

        # Calculate aggregate metrics
        ranks = [r['rank'] for r in results]
        reciprocal_ranks = [r['reciprocal_rank'] for r in results]

        metrics = {
            'num_triples': len(results),
            'mean_rank': float(np.mean(ranks)),
            'median_rank': float(np.median(ranks)),
            'mean_reciprocal_rank': float(np.mean(reciprocal_ranks)),
            'hits@1': float(np.mean([r['hits@1'] for r in results])),
            'hits@3': float(np.mean([r['hits@3'] for r in results])),
            'hits@5': float(np.mean([r['hits@5'] for r in results])),
            'hits@10': float(np.mean([r['hits@10'] for r in results])),
        }

        logger.info(f"Evaluation Results:")
        logger.info(f"  Mean Rank: {metrics['mean_rank']:.2f}")
        logger.info(f"  Mean Reciprocal Rank: {metrics['mean_reciprocal_rank']:.4f}")
        logger.info(f"  Hits@1: {metrics['hits@1']:.4f}")
        logger.info(f"  Hits@3: {metrics['hits@3']:.4f}")
        logger.info(f"  Hits@5: {metrics['hits@5']:.4f}")
        logger.info(f"  Hits@10: {metrics['hits@10']:.4f}")

        # Save detailed results if path provided
        if output_path:
            output_data = {
                'metrics': metrics,
                'per_triple_results': results
            }

            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)

            logger.info(f"Saved detailed results to {output_path}")

            # Also save as CSV for easier analysis
            csv_path = output_path.replace('.json', '.csv')
            df = pd.DataFrame(results)
            # Don't include top_predictions in CSV (too verbose)
            df_clean = df.drop(columns=['top_predictions'])
            df_clean.to_csv(csv_path, index=False)
            logger.info(f"Saved CSV results to {csv_path}")

        return {
            'metrics': metrics,
            'per_triple_results': results
        }

    def predict_tails(
        self,
        head: int,
        relation: int,
        top_k: int = 10,
        known_triples: Optional[set] = None
    ) -> List[Tuple[int, float]]:
        """Predict top-k tail entities for a given (head, relation) pair.

        Args:
            head: Head entity index
            relation: Relation index
            top_k: Number of top predictions to return
            known_triples: Set of known triples to filter

        Returns:
            List of (entity_id, score) tuples
        """
        self.model.eval()

        with torch.no_grad():
            hr_batch = torch.LongTensor([[head, relation]]).to(self.device)
            all_scores = self.model.score_t(hr_batch).squeeze(0)

            # Filter known triples if requested
            if self.filter_triples and known_triples is not None:
                for h, r, t in known_triples:
                    if h == head and r == relation:
                        all_scores[t] = float('-inf')

            # Get top-k
            top_scores, top_indices = torch.topk(all_scores, k=min(top_k, len(all_scores)))

            predictions = [
                (int(top_indices[i]), float(top_scores[i]))
                for i in range(len(top_indices))
            ]

        return predictions

    def get_entity_embeddings(self) -> torch.Tensor:
        """Get entity embeddings from the model.

        Returns:
            Entity embedding matrix, shape (num_entities, embedding_dim)
        """
        return self.model.entity_representations[0]().detach().cpu()

    def get_relation_embeddings(self) -> torch.Tensor:
        """Get relation embeddings from the model.

        Returns:
            Relation embedding matrix, shape (num_relations, embedding_dim)
        """
        return self.model.relation_representations[0]().detach().cpu()


def evaluate_model(
    model: Model,
    test_triples: CoreTriplesFactory,
    training_triples: Optional[CoreTriplesFactory] = None,
    validation_triples: Optional[CoreTriplesFactory] = None,
    batch_size: int = 32,
    filter_triples: bool = True,
    output_path: Optional[str] = None,
    device: Optional[str] = None
) -> Dict[str, any]:
    """Convenience function to evaluate a model.

    Args:
        model: Trained PyKEEN model
        test_triples: Test triples factory
        training_triples: Training triples factory (for filtering)
        validation_triples: Validation triples factory (for filtering)
        batch_size: Batch size for evaluation
        filter_triples: Whether to filter known triples
        output_path: Optional path to save results
        device: Device to run evaluation on

    Returns:
        Dictionary with metrics and per-triple results
    """
    evaluator = DetailedEvaluator(
        model=model,
        batch_size=batch_size,
        filter_triples=filter_triples,
        device=device
    )

    return evaluator.evaluate_dataset(
        test_triples=test_triples,
        training_triples=training_triples,
        validation_triples=validation_triples,
        output_path=output_path
    )
