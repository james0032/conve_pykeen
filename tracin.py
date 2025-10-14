"""
TracIn (Tracing Influence) implementation for ConvE model.

TracIn computes the influence of training examples on test predictions by
approximating the influence through gradients. This helps understand which
training triples are most important for specific predictions.

Reference: Pruthi et al. "Estimating Training Data Influence by Tracing Gradient Descent"
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from pykeen.models import Model
from pykeen.triples import CoreTriplesFactory
from tqdm import tqdm

logger = logging.getLogger(__name__)


class TracInAnalyzer:
    """TracIn analyzer for computing training data influence.

    This class computes the influence of training examples on test predictions
    using gradient-based approximations.
    """

    def __init__(
        self,
        model: Model,
        loss_fn: str = 'bce',
        device: Optional[str] = None
    ):
        """Initialize TracIn analyzer.

        Args:
            model: Trained PyKEEN model
            loss_fn: Loss function to use ('bce' for binary cross-entropy)
            device: Device to run computation on
        """
        self.model = model
        self.loss_fn = loss_fn
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def compute_gradient(
        self,
        head: int,
        relation: int,
        tail: int,
        label: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """Compute gradient of loss with respect to model parameters for a single triple.

        Args:
            head: Head entity index
            relation: Relation index
            tail: Tail entity index
            label: Label for the triple (1.0 for positive, 0.0 for negative)

        Returns:
            Dictionary mapping parameter names to gradient tensors
        """
        self.model.train()
        self.model.zero_grad()

        # Forward pass
        hr_batch = torch.LongTensor([[head, relation]]).to(self.device)
        scores = self.model.score_t(hr_batch)  # Shape: (1, num_entities)

        # Get score for the specific tail
        score = scores[0, tail]

        # Compute loss
        if self.loss_fn == 'bce':
            # Binary cross-entropy loss
            target = torch.tensor([label], dtype=torch.float32, device=self.device)
            loss = F.binary_cross_entropy_with_logits(score.unsqueeze(0), target)
        else:
            raise ValueError(f"Unknown loss function: {self.loss_fn}")

        # Backward pass
        loss.backward()

        # Collect gradients
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone()

        return gradients

    def compute_influence(
        self,
        train_triple: Tuple[int, int, int],
        test_triple: Tuple[int, int, int],
        learning_rate: float = 1e-3
    ) -> float:
        """Compute influence of a training triple on a test triple.

        The influence is computed as:
            influence = learning_rate * dot(grad_train, grad_test)

        A positive influence means the training example pushes the model
        towards the correct prediction for the test example.

        Args:
            train_triple: (head, relation, tail) training triple
            test_triple: (head, relation, tail) test triple
            learning_rate: Learning rate used during training

        Returns:
            Influence score (scalar)
        """
        # Compute gradients for training triple (positive example)
        train_h, train_r, train_t = train_triple
        grad_train = self.compute_gradient(train_h, train_r, train_t, label=1.0)

        # Compute gradients for test triple (positive example)
        test_h, test_r, test_t = test_triple
        grad_test = self.compute_gradient(test_h, test_r, test_t, label=1.0)

        # Compute dot product of gradients
        influence = 0.0
        for name in grad_train:
            if name in grad_test:
                # Flatten and compute dot product
                grad_train_flat = grad_train[name].flatten()
                grad_test_flat = grad_test[name].flatten()
                influence += torch.dot(grad_train_flat, grad_test_flat).item()

        # Scale by learning rate
        influence *= learning_rate

        return influence

    def compute_influences_for_test_triple(
        self,
        test_triple: Tuple[int, int, int],
        training_triples: CoreTriplesFactory,
        learning_rate: float = 1e-3,
        top_k: Optional[int] = None,
        batch_size: int = 1
    ) -> List[Dict[str, any]]:
        """Compute influences of all training triples on a single test triple.

        Args:
            test_triple: (head, relation, tail) test triple
            training_triples: Training triples factory
            learning_rate: Learning rate used during training
            top_k: If specified, return only top-k most influential triples
            batch_size: Batch size for processing (currently must be 1)

        Returns:
            List of dictionaries with training triple info and influence score
        """
        logger.info(f"Computing influences for test triple: {test_triple}")

        influences = []

        # Compute gradient for test triple once
        test_h, test_r, test_t = test_triple
        grad_test = self.compute_gradient(test_h, test_r, test_t, label=1.0)

        # Iterate through training triples
        for h, r, t in tqdm(training_triples.mapped_triples, desc="Computing influences"):
            train_triple = (int(h), int(r), int(t))

            # Compute gradient for training triple
            grad_train = self.compute_gradient(*train_triple, label=1.0)

            # Compute dot product
            influence = 0.0
            for name in grad_train:
                if name in grad_test:
                    grad_train_flat = grad_train[name].flatten()
                    grad_test_flat = grad_test[name].flatten()
                    influence += torch.dot(grad_train_flat, grad_test_flat).item()

            influence *= learning_rate

            influences.append({
                'train_head': train_triple[0],
                'train_relation': train_triple[1],
                'train_tail': train_triple[2],
                'influence': influence
            })

        # Sort by influence (descending)
        influences.sort(key=lambda x: abs(x['influence']), reverse=True)

        # Return top-k if specified
        if top_k is not None:
            influences = influences[:top_k]

        return influences

    def analyze_test_set(
        self,
        test_triples: CoreTriplesFactory,
        training_triples: CoreTriplesFactory,
        learning_rate: float = 1e-3,
        top_k: int = 10,
        max_test_triples: Optional[int] = None,
        output_path: Optional[str] = None
    ) -> Dict[str, any]:
        """Analyze influence of training data on test predictions.

        Args:
            test_triples: Test triples factory
            training_triples: Training triples factory
            learning_rate: Learning rate used during training
            top_k: Number of top influential training triples to return per test triple
            max_test_triples: Maximum number of test triples to analyze (None for all)
            output_path: Optional path to save results

        Returns:
            Dictionary with influence analysis results
        """
        logger.info(f"Analyzing influences for test set...")

        results = []
        test_triple_list = [(int(h), int(r), int(t)) for h, r, t in test_triples.mapped_triples]

        if max_test_triples is not None:
            test_triple_list = test_triple_list[:max_test_triples]

        for test_triple in tqdm(test_triple_list, desc="Analyzing test triples"):
            influences = self.compute_influences_for_test_triple(
                test_triple=test_triple,
                training_triples=training_triples,
                learning_rate=learning_rate,
                top_k=top_k
            )

            results.append({
                'test_head': test_triple[0],
                'test_relation': test_triple[1],
                'test_tail': test_triple[2],
                'top_influences': influences
            })

        analysis = {
            'num_test_triples': len(results),
            'num_training_triples': len(training_triples),
            'top_k': top_k,
            'learning_rate': learning_rate,
            'results': results
        }

        # Save results if path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(analysis, f, indent=2)
            logger.info(f"Saved TracIn analysis to {output_path}")

        return analysis

    def compute_self_influence(
        self,
        training_triples: CoreTriplesFactory,
        learning_rate: float = 1e-3,
        output_path: Optional[str] = None
    ) -> List[Dict[str, any]]:
        """Compute self-influence for each training triple.

        Self-influence measures how much a training example influences itself,
        which can indicate the importance or difficulty of that example.

        Args:
            training_triples: Training triples factory
            learning_rate: Learning rate used during training
            output_path: Optional path to save results

        Returns:
            List of dictionaries with triple info and self-influence score
        """
        logger.info("Computing self-influences for training set...")

        influences = []

        for h, r, t in tqdm(training_triples.mapped_triples, desc="Computing self-influences"):
            triple = (int(h), int(r), int(t))

            # Compute influence on itself
            grad = self.compute_gradient(*triple, label=1.0)

            # Compute squared L2 norm of gradient
            self_influence = 0.0
            for name in grad:
                grad_flat = grad[name].flatten()
                self_influence += torch.dot(grad_flat, grad_flat).item()

            self_influence *= learning_rate

            influences.append({
                'head': triple[0],
                'relation': triple[1],
                'tail': triple[2],
                'self_influence': self_influence
            })

        # Sort by self-influence (descending)
        influences.sort(key=lambda x: x['self_influence'], reverse=True)

        # Save results if path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(influences, f, indent=2)
            logger.info(f"Saved self-influence analysis to {output_path}")

        return influences


def compute_tracin_influence(
    model: Model,
    test_triple: Tuple[int, int, int],
    training_triples: CoreTriplesFactory,
    learning_rate: float = 1e-3,
    top_k: int = 10,
    device: Optional[str] = None
) -> List[Dict[str, any]]:
    """Convenience function to compute TracIn influences.

    Args:
        model: Trained PyKEEN model
        test_triple: (head, relation, tail) test triple
        training_triples: Training triples factory
        learning_rate: Learning rate used during training
        top_k: Number of top influential training triples to return
        device: Device to run computation on

    Returns:
        List of dictionaries with training triple info and influence score
    """
    analyzer = TracInAnalyzer(model=model, device=device)
    return analyzer.compute_influences_for_test_triple(
        test_triple=test_triple,
        training_triples=training_triples,
        learning_rate=learning_rate,
        top_k=top_k
    )
