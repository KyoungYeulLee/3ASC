"""Top-k recall: the metric the model is selected and reported on."""

from typing import Iterable

import numpy as np


class AverageMeter:
    """Running average of a scalar."""

    def __init__(self):
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """Add an observation.

        Args:
            val (float): Observed value.
            n (int): Weight of the observation.
        """
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def aggregate_by_level_numpy(
    levels: np.ndarray,
    probs: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce instances to one entry per level, keeping the best-scoring one.

    A causal variant can be reported more than once - one row per associated
    disease - so the ranking is done over levels (CPRAs), not raw rows.

    Args:
        levels (np.ndarray): Level identifier of each instance.
        probs (np.ndarray): Predicted probability of each instance.
        labels (np.ndarray): Label of each instance.

    Returns:
        tuple[np.ndarray, np.ndarray]: Probability and label of the highest
            scoring instance of every unique level.
    """
    unique_levels, inverse_indices = np.unique(levels, return_inverse=True)
    n_unique = len(unique_levels)

    aggregated_probs = np.zeros(n_unique)
    aggregated_labels = np.zeros(n_unique)

    for i in range(n_unique):
        mask = inverse_indices == i
        level_probs = probs[mask]
        level_labels = labels[mask]
        max_idx = np.argmax(level_probs)
        aggregated_probs[i] = level_probs[max_idx]
        aggregated_labels[i] = level_labels[max_idx]

    return aggregated_probs, aggregated_labels


def calculate_topk_recalls_for_ks(
    sorted_labels: np.ndarray,
    n_causal: int,
    max_k: int,
) -> dict[int, float | None]:
    """Top-k recall for every k from pre-sorted labels.

    Args:
        sorted_labels (np.ndarray): Labels ordered by descending probability.
        n_causal (int): Number of causal instances.
        max_k (int): Largest k to compute.

    Returns:
        dict[int, float | None]: Recall per k. None where k is smaller than the
            number of causal variants, which makes the recall unattainable.
    """
    recalls = {}
    cumsum = np.cumsum(sorted_labels)

    for k in range(1, max_k + 1):
        if k < n_causal:
            recalls[k] = None
        else:
            k_idx = min(k, len(sorted_labels)) - 1
            recalls[k] = cumsum[k_idx] / n_causal

    return recalls


def calculate_multi_k_topk_recall(
    instance_prob: np.ndarray,
    instance_label: np.ndarray,
    max_k: int,
) -> dict[int, float | None]:
    """Top-k recall of one patient, over raw instances.

    Args:
        instance_prob (np.ndarray): Predicted probability of each instance.
        instance_label (np.ndarray): Label of each instance.
        max_k (int): Largest k to compute.

    Returns:
        dict[int, float | None]: Recall per k, all None if the patient has no
            causal variant.
    """
    n_causal = int(np.sum(instance_label))
    if n_causal == 0:
        return {k: None for k in range(1, max_k + 1)}

    sorted_labels = instance_label[np.argsort(instance_prob)[::-1]]

    return calculate_topk_recalls_for_ks(sorted_labels, n_causal, max_k)


def calculate_multi_k_topk_recall_by_level(
    instance_level: np.ndarray,
    instance_prob: np.ndarray,
    instance_label: np.ndarray,
    max_k: int,
) -> dict[int, float | None]:
    """Top-k recall of one patient, over levels.

    Args:
        instance_level (np.ndarray): Level identifier of each instance.
        instance_prob (np.ndarray): Predicted probability of each instance.
        instance_label (np.ndarray): Label of each instance.
        max_k (int): Largest k to compute.

    Returns:
        dict[int, float | None]: Recall per k, all None if the patient has no
            causal variant.
    """
    aggregated_probs, aggregated_labels = aggregate_by_level_numpy(
        np.asarray(instance_level),
        np.asarray(instance_prob),
        np.asarray(instance_label),
    )

    n_causal = int(np.sum(aggregated_labels))
    if n_causal == 0:
        return {k: None for k in range(1, max_k + 1)}

    sorted_labels = aggregated_labels[np.argsort(aggregated_probs)[::-1]]

    return calculate_topk_recalls_for_ks(sorted_labels, n_causal, max_k)


def build_per_patient_recall_matrix(
    instance_probs: Iterable[np.ndarray],
    instance_labels: Iterable[np.ndarray],
    instance_levels: Iterable[list[str]],
    max_k: int,
) -> np.ndarray:
    """Top-k recall of every patient, for every k.

    Args:
        instance_probs (Iterable[np.ndarray]): Instance probabilities per patient.
        instance_labels (Iterable[np.ndarray]): Instance labels per patient.
        instance_levels (Iterable[list[str]]): Instance levels per patient.
        max_k (int): Largest k to compute.

    Returns:
        np.ndarray: Matrix of shape (n_patients, max_k), np.nan where the recall
            is undefined.
    """
    rows = [
        calculate_multi_k_topk_recall_by_level(
            np.asarray(instance_level),
            np.asarray(instance_prob),
            np.asarray(instance_label),
            max_k,
        )
        for instance_level, instance_prob, instance_label in zip(
            instance_levels, instance_probs, instance_labels
        )
    ]

    recall_matrix = np.full((len(rows), max_k), np.nan, dtype=float)
    for row_idx, patient_recalls in enumerate(rows):
        for k in range(1, max_k + 1):
            recall = patient_recalls[k]
            if recall is not None:
                recall_matrix[row_idx, k - 1] = recall

    return recall_matrix


def mean_recalls_from_matrix(recall_matrix: np.ndarray) -> dict[int, float | None]:
    """Average a per-patient recall matrix over patients, for every k.

    Args:
        recall_matrix (np.ndarray): Output of `build_per_patient_recall_matrix`.

    Returns:
        dict[int, float | None]: Mean recall per k over the patients it is
            defined for, None where no patient has a defined value.
    """
    result = {}
    for k in range(1, recall_matrix.shape[1] + 1):
        column = recall_matrix[:, k - 1]
        defined = column[~np.isnan(column)]
        result[k] = float(defined.mean()) if defined.size else None

    return result
