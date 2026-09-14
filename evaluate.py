"""Evaluate a trained 3ASC 3.0 checkpoint on a benchmark cohort.

    python evaluate.py [--config conf/config.yaml]

Reports, for the whole benchmark and for the patients whose causal variant is
of each variant type: top-k recall over CPRAs, the pooled AUROC and PR-AUC over
every variant of every patient, and the mean per-patient AUROC and PR-AUC.
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.data_model import PatientData, PatientDataSet
from core.datasets import VariantDataset
from core.metric import (
    aggregate_by_level_numpy,
    build_per_patient_recall_matrix,
    mean_recalls_from_matrix,
)
from core.utils import load_pickle
from model.networks import build_model

logger = logging.getLogger("3asc.evaluate")


def load_checkpoint(
    predictor_path: Path, device: str
) -> tuple[torch.nn.Module, DictConfig, dict]:
    """Rebuild the trained model, its configuration and its scalers.

    The architecture and the feature set are read from the config saved next to
    the weights, so an evaluation cannot silently disagree with how the model
    was trained.

    Args:
        predictor_path (Path): Checkpoint file to evaluate.
        device (str): Device to run inference on.

    Returns:
        tuple: The model in eval mode, the training configuration, and the
            fitted scaler of every variant type.
    """
    checkpoint_dir = predictor_path.parent
    cfg = OmegaConf.load(checkpoint_dir / "config.yaml")

    model = build_model(cfg)
    model.load_state_dict(
        torch.load(predictor_path, weights_only=True, map_location="cpu")
    )
    model = model.to(device).eval()

    scalers = {
        variant_type: joblib.load(checkpoint_dir / f"{variant_type}_scaler.joblib")
        for variant_type in cfg.model.variant_type
    }

    return model, cfg, scalers


def predict(
    model: torch.nn.Module, patient_data_x: dict, variant_types: list[str], device: str
) -> tuple[float, np.ndarray]:
    """Score one patient.

    Args:
        model (torch.nn.Module): Model to run.
        patient_data_x (dict): Features of one patient, from `VariantDataset`.
        variant_types (list[str]): Variant types to score, in the order they
            were embedded during training.
        device (str): Device to run inference on.

    Returns:
        tuple[float, np.ndarray]: Bag probability, and the probability of every
            variant of the patient, ordered by variant type.
    """
    variant_x = {
        v_type: patient_data_x[v_type].to(device)
        for v_type in variant_types
        if v_type in patient_data_x
    }

    with torch.no_grad():
        bag_logit, instance_logits = model(
            x=variant_x,
            v_embeddings=patient_data_x["v_embeddings"].to(device),
            p_embeddings=patient_data_x["p_embeddings"].to(device),
            gfe_known_mask=patient_data_x["gfe_known_mask"].to(device),
        )

    bag_prob = torch.sigmoid(bag_logit.cpu()).item()
    instance_probs = torch.sigmoid(instance_logits.cpu()).numpy().squeeze(axis=0)

    return bag_prob, instance_probs


def collect_patient_predictions(
    patient_data: PatientData,
    instance_probs: np.ndarray,
    variant_types: list[str],
    groups: dict[str, dict[str, list]],
) -> bool:
    """File one patient's predictions under "all" and its causal variant types.

    A patient is listed under a variant type when its causal variant is of that
    type; the scores filed are always those of its whole bag, so the per-type
    numbers answer "how well is this kind of case solved", not "how well is this
    kind of variant scored".

    Args:
        patient_data (PatientData): The patient that was scored.
        instance_probs (np.ndarray): Predicted probability of every variant.
        variant_types (list[str]): Variant types that were scored.
        groups (dict[str, dict[str, list]]): Collected scores, labels and levels
            per group, updated in place.

    Returns:
        bool: Whether the predictions were filed. False when the scores, labels
            and CPRAs do not line up - the caller keeps such a patient out of
            every metric, the bag-level ones included.
    """
    labels = []
    levels = []
    causal_variant_types = []
    for variant_type in variant_types:
        variant_data = getattr(patient_data, f"{variant_type}_data")
        if len(variant_data.x) == 0:
            continue
        labels.extend(variant_data.y)
        # Recall is counted over CPRAs: one variant can appear once per disease.
        levels.extend(variant.cpra for variant in variant_data.variants)
        if np.sum(variant_data.y) > 0:
            causal_variant_types.append(variant_type)

    if not len(instance_probs) == len(labels) == len(levels):
        logger.warning(
            f"Skipping {patient_data.sample_id}: {len(instance_probs)} scores, "
            f"{len(labels)} labels and {len(levels)} CPRAs do not line up"
        )
        return False

    for group in ["all"] + causal_variant_types:
        groups[group]["scores"].append(np.asarray(instance_probs))
        groups[group]["labels"].append(np.asarray(labels))
        groups[group]["levels"].append(levels)

    return True


def compute_metrics(
    groups: dict[str, dict[str, list]],
    bag_labels: list[int],
    bag_scores: list[float],
    variant_types: list[str],
    top_k: int,
) -> dict[str, float | None]:
    """Turn the collected predictions into the reported metrics.

    Args:
        groups (dict[str, dict[str, list]]): Scores, labels and levels per group.
        bag_labels (list[int]): Whether each patient has a causal variant.
        bag_scores (list[float]): Bag probability of each patient.
        variant_types (list[str]): Variant types that were scored.
        top_k (int): Largest k to report top-k recall for.

    Returns:
        dict[str, float | None]: Every reported metric, keyed by name. A metric
            the benchmark leaves undefined is None, so it serialises to null
            instead of being averaged in as a number.
    """
    metrics = {
        "bag_auroc": (
            roc_auc_score(bag_labels, bag_scores)
            if len(np.unique(bag_labels)) > 1
            else None
        )
    }

    for group, predictions in groups.items():
        recall_matrix = build_per_patient_recall_matrix(
            predictions["scores"],
            predictions["labels"],
            predictions["levels"],
            top_k,
        )
        k_recalls = mean_recalls_from_matrix(recall_matrix)

        max_recall = None
        first_k_at_max = None
        for k in range(1, top_k + 1):
            metrics[f"top-{k}_recall_{group}"] = k_recalls[k]
            if k_recalls[k] is not None and (
                max_recall is None or k_recalls[k] > max_recall
            ):
                max_recall = k_recalls[k]
                first_k_at_max = k
        metrics[f"first_k_at_max_recall_{group}"] = first_k_at_max

    # Pooled over every variant of every patient.
    all_scores = np.concatenate(groups["all"]["scores"])
    all_labels = np.concatenate(groups["all"]["labels"])
    metrics["auroc"] = roc_auc_score(all_labels, all_scores)
    metrics["auprc"] = average_precision_score(all_labels, all_scores)
    # PR-AUC has no fixed no-skill baseline: it equals the positive rate.
    metrics["positive_rate"] = float(all_labels.mean())
    metrics["n_positive_variants"] = int(all_labels.sum())

    for variant_type in variant_types:
        per_patient_aurocs = []
        per_patient_auprcs = []
        predictions = groups.get(
            variant_type, {"scores": [], "labels": [], "levels": []}
        )
        for scores, labels, levels in zip(
            predictions["scores"], predictions["labels"], predictions["levels"]
        ):
            grouped_scores, grouped_labels = aggregate_by_level_numpy(
                np.asarray(levels), np.asarray(scores), np.asarray(labels)
            )
            # AUROC is undefined without both classes, PR-AUC without a positive.
            if np.unique(grouped_labels).size > 1:
                per_patient_aurocs.append(
                    float(roc_auc_score(grouped_labels, grouped_scores))
                )
            if grouped_labels.sum() > 0:
                per_patient_auprcs.append(
                    float(average_precision_score(grouped_labels, grouped_scores))
                )

        metrics[f"mean_auroc_{variant_type}"] = (
            float(np.mean(per_patient_aurocs)) if per_patient_aurocs else None
        )
        metrics[f"mean_auprc_{variant_type}"] = (
            float(np.mean(per_patient_auprcs)) if per_patient_auprcs else None
        )

    return metrics


def evaluate(cfg: DictConfig) -> dict[str, float | None]:
    """Score every patient of the benchmark and report the metrics.

    Args:
        cfg (DictConfig): Full configuration; `evaluate` selects the checkpoint,
            the benchmark and the disease map to score with.

    Returns:
        dict[str, float | None]: Every reported metric, keyed by name.
    """
    eval_cfg = cfg.evaluate
    device = eval_cfg.device
    if "cuda" in str(device):
        # Inference-only speedups; they do not change the ranking of a bag.
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model, train_cfg, scalers = load_checkpoint(Path(eval_cfg.predictor_path), device)
    variant_types = list(train_cfg.model.variant_type)
    logger.info(f"Evaluating {eval_cfg.predictor_path} on {variant_types}")

    benchmark_dataset: PatientDataSet = load_pickle(eval_cfg.benchmark_data_path)
    data_loader = DataLoader(
        VariantDataset(
            patient_dataset=benchmark_dataset,
            features=train_cfg.features,
            scalers=scalers,
            variant_types=variant_types,
            disease_embeddings_path=eval_cfg.disease_embeddings_path,
            patient_symptom_embeddings_path=eval_cfg.patient_symptom_embeddings_path,
            disease_cpra_map_path=eval_cfg.disease_cpra_map_path,
        )
    )

    groups = defaultdict(lambda: defaultdict(list))
    bag_labels = []
    bag_scores = []
    for patient_data, batch in zip(benchmark_dataset, tqdm(data_loader), strict=True):
        bag_label, patient_data_x, _ = batch
        if not any(v_type in patient_data_x for v_type in variant_types):
            logger.warning(f"Skipping {patient_data.sample_id}: no variant to score")
            continue

        bag_prob, instance_probs = predict(model, patient_data_x, variant_types, device)
        if not collect_patient_predictions(
            patient_data, instance_probs, variant_types, groups
        ):
            continue
        # The stored bag label, which is also the target the training BCE uses.
        bag_labels.append(int(bag_label.item()))
        bag_scores.append(bag_prob)

    metrics = compute_metrics(
        groups, bag_labels, bag_scores, variant_types, eval_cfg.top_k
    )
    logger.info(json.dumps(metrics, indent=4))

    return metrics


def main() -> None:
    """Parse the command line and evaluate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("conf/config.yaml"), help="config file"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="{asctime}\t{name}\t{levelname}\t{message}",
        style="{",
    )
    evaluate(OmegaConf.load(args.config))

    return


if __name__ == "__main__":
    main()
