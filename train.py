"""Train the 3ASC 3.0 model.

python train.py [--config conf/config.yaml]
"""

import argparse
import logging
import math
from pathlib import Path

import joblib
import numpy as np
import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from core.data_model import PatientDataSet
from core.datasets import VariantDataset, load_training_dataset, to_float_features
from core.trainer import MILGFETrainer
from core.utils import set_seed
from model.networks import build_model

logger = logging.getLogger("3asc.train")


def split_train_val(
    patient_dataset: PatientDataSet, cfg: DictConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Split patients into a training and a validation fold.

    The split is stratified on which variant types carry the causal variant, so
    the rarer types (STR, BND, INS) are represented in both folds.

    Args:
        patient_dataset (PatientDataSet): Patients to split.
        cfg (DictConfig): Full configuration.

    Returns:
        tuple[np.ndarray, np.ndarray]: Training and validation indices.
    """
    multi_labels = np.array(
        [
            [
                1 if getattr(patient_data, f"{variant_type}_data").y.sum() > 0 else 0
                for variant_type in cfg.features.keys()
            ]
            for patient_data in patient_dataset
        ],
        dtype=int,
    )
    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        random_state=cfg.model.random_state,
        test_size=cfg.model.validation.split_ratio,
    )

    return next(splitter.split(np.arange(len(patient_dataset)), multi_labels))


def build_dataloader(
    patient_dataset: PatientDataSet,
    cfg: DictConfig,
    scalers: dict[str, StandardScaler],
    shuffle: bool,
) -> DataLoader:
    """Wrap a fold into a dataloader serving one patient per batch.

    Args:
        patient_dataset (PatientDataSet): Patients of the fold.
        cfg (DictConfig): Full configuration.
        scalers (dict[str, StandardScaler]): Scaler per variant type, shared
            with the other fold and fitted on the training fold only.
        shuffle (bool): Whether to reshuffle the patients every epoch.

    Returns:
        DataLoader: Dataloader over a `VariantDataset` of the fold.
    """
    dataset = VariantDataset(
        patient_dataset=patient_dataset,
        features=cfg.features,
        scalers=scalers,
        variant_types=cfg.model.variant_type,
        disease_embeddings_path=cfg.model.disease_embeddings_path,
        patient_symptom_embeddings_path=cfg.model.patient_symptom_embeddings_path,
        disease_cpra_map_path=cfg.model.disease_cpra_map_path,
    )
    sampler = DistributedSampler(
        dataset=dataset, num_replicas=1, rank=0, shuffle=shuffle
    )
    # A private generator keeps the global RNG stream - and with it every
    # dropout mask - identical to the published run. The epoch order comes
    # from the sampler's own seed, not from here.
    generator = torch.Generator()
    generator.manual_seed(cfg.model.random_state)

    return DataLoader(
        dataset=dataset, sampler=sampler, pin_memory=True, generator=generator
    )


def build_optimizer_and_scheduler(
    model: torch.nn.Module, cfg: DictConfig, steps_per_epoch: int
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """Create the optimizer and its cosine schedule with linear warmup.

    Args:
        model (torch.nn.Module): Model to optimize.
        cfg (DictConfig): Full configuration.
        steps_per_epoch (int): Number of bags in one training epoch.

    Returns:
        tuple: The AdamW optimizer and its LambdaLR schedule. The schedule is
            counted in optimizer steps, so gradient accumulation shortens it.
    """
    parameters = cfg.model.parameters
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=parameters.learning_rate,
        weight_decay=parameters.weight_decay,
    )

    num_training_steps = int(
        cfg.model.train.n_epochs
        * steps_per_epoch
        // cfg.trainer.gradient_accumulation_steps
    )
    num_warmup_steps = int(num_training_steps * parameters.warmup_proportion)

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    logger.info(
        f"{num_training_steps} optimizer steps, {num_warmup_steps} of them warmup"
    )

    return optimizer, scheduler


def log_epoch_metrics(
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    cfg: DictConfig,
) -> None:
    """Log the losses and top-k recalls of one epoch.

    Args:
        epoch (int): Epoch number.
        train_metrics (dict[str, float]): Metrics of the training phase.
        val_metrics (dict[str, float]): Metrics of the validation phase.
        cfg (DictConfig): Full configuration.
    """
    top_k = cfg.model.validation.top_k
    logger.info(
        f"Epoch {epoch} | "
        f"Train Loss: {train_metrics['train_total_loss_avg']:.4f}, "
        f"Recall@{top_k}: {train_metrics[f'train_top_{top_k}_recall_avg']:.4f} | "
        f"Val Loss: {val_metrics['val_total_loss_avg']:.4f}, "
        f"Recall@{top_k}: {val_metrics[f'val_top_{top_k}_recall_avg']:.4f}"
    )
    for phase, metrics in (("Train", train_metrics), ("Val", val_metrics)):
        by_type = ", ".join(
            f"{variant_type.upper()}: "
            f"{metrics[f'{phase.lower()}_top_{top_k}_recall_{variant_type}_avg']:.4f}"
            for variant_type in cfg.model.variant_type
        )
        logger.info(f"  {phase} Recall@{top_k} by type: {by_type}")

    return


def train(cfg: DictConfig) -> None:
    """Train the model and write one checkpoint per epoch.

    Args:
        cfg (DictConfig): Full configuration.
    """
    set_seed(cfg.model.random_state)
    logger.info(f"Random seed set to {cfg.model.random_state}")

    checkpoint_dir = Path(cfg.checkpoint_dir_path) / cfg.model.experiment_name
    if checkpoint_dir.exists():
        raise FileExistsError(
            f"{checkpoint_dir} already exists. Checkpoints are never "
            "overwritten - pick a new model.experiment_name or move the "
            "previous run away."
        )
    checkpoint_dir.mkdir(parents=True)

    patient_dataset = load_training_dataset(
        train_data_path=cfg.model.train.train_data_path,
        exclude_from_train_data_path=cfg.model.train.exclude_from_train_data_path,
    )
    train_indices, val_indices = split_train_val(patient_dataset, cfg)
    train_patient_dataset = patient_dataset[train_indices]
    val_patient_dataset = patient_dataset[val_indices]
    logger.info(
        f"{len(train_patient_dataset)} training and "
        f"{len(val_patient_dataset)} validation patients"
    )

    scalers = {
        variant_type: StandardScaler() for variant_type in cfg.model.variant_type
    }
    train_data_loader = build_dataloader(
        train_patient_dataset, cfg, scalers, shuffle=True
    )
    val_data_loader = build_dataloader(val_patient_dataset, cfg, scalers, shuffle=False)
    for variant_type in cfg.model.variant_type:
        feature_indices = train_data_loader.dataset.feature_indices[variant_type]
        scalers[variant_type].fit(
            to_float_features(
                train_patient_dataset.feature_matrix(variant_type)[:, feature_indices]
            )
        )

    model = build_model(cfg)
    model = model.to(cfg.model.device)
    optimizer, scheduler = build_optimizer_and_scheduler(
        model, cfg, steps_per_epoch=len(train_data_loader)
    )
    trainer = MILGFETrainer(
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=cfg.model.device,
    )

    # Everything an evaluation run needs to rebuild this model.
    OmegaConf.save(cfg, checkpoint_dir / "config.yaml")
    for variant_type, scaler in scalers.items():
        joblib.dump(scaler, checkpoint_dir / f"{variant_type}_scaler.joblib")

    best_metric = math.inf
    patience = 0
    for epoch in range(1, cfg.model.train.n_epochs + 1):
        train_data_loader.sampler.set_epoch(epoch)
        train_metrics = trainer.run_epoch(
            phase="train", epoch=epoch, dataloader=train_data_loader
        )
        val_data_loader.sampler.set_epoch(epoch)
        val_metrics = trainer.run_epoch(
            phase="val", epoch=epoch, dataloader=val_data_loader
        )
        log_epoch_metrics(epoch, train_metrics, val_metrics, cfg)

        torch.save(model.state_dict(), checkpoint_dir / f"epoch_{epoch}_model.pth")

        if val_metrics["val_total_loss_avg"] < best_metric:
            best_metric = val_metrics["val_total_loss_avg"]
            patience = 0
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pth")
            logger.info(f"Best model updated at epoch {epoch}")
        else:
            patience += 1
            if patience == cfg.model.parameters.n_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    return


def main() -> None:
    """Parse the command line and train."""
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
    train(OmegaConf.load(args.config))

    return


if __name__ == "__main__":
    main()
