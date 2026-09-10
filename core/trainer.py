"""Training and validation loop of the 3ASC 3.0 (MIL-GFE) model."""

from collections import defaultdict
from typing import Literal

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from tqdm import tqdm

from core.losses import focal_loss, pointwise_ranknet_loss
from core.metric import AverageMeter, calculate_multi_k_topk_recall


class MILGFETrainer:
    """Runs one epoch at a time over a bag-per-batch dataloader.

    The objective sums three terms: a bag-level BCE on the CLS logit, a focal
    loss on the instance logits, and a pointwise RankNet loss that pushes the
    causal variant above the rest of the bag.

    Args:
        cfg (DictConfig): Full configuration.
        model (torch.nn.Module): Model to train.
        optimizer (torch.optim.Optimizer): Optimizer.
        scheduler (torch.optim.lr_scheduler.LRScheduler): Learning rate schedule,
            stepped once per optimizer step.
        device (str | torch.device): Device the batches are moved to.
    """

    def __init__(
        self,
        cfg: DictConfig,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        device: str | torch.device,
    ):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.variant_types = list(cfg.model.variant_type)
        self.top_k = cfg.model.validation.top_k
        self.accumulation_steps = cfg.trainer.gradient_accumulation_steps
        self.metric_names = [
            "total_loss",
            "bag_loss",
            "instance_loss",
            "ranknet_loss",
            f"top_{self.top_k}_recall",
            *[f"top_{self.top_k}_recall_{v_type}" for v_type in self.variant_types],
        ]

    def step(self, step: int, batch: tuple) -> None:
        """Run one bag through the model, and back through it when training.

        Args:
            step (int): 1-based index of the bag within the epoch. Gradients are
                applied every `gradient_accumulation_steps` bags and at the end
                of the epoch.
            batch (tuple): One patient, as returned by `VariantDataset`.
        """
        bag_label, patient_data_x, patient_data_y = batch
        bag_label = bag_label.to(self.device)
        if len(bag_label.shape) == 1:
            bag_label = bag_label.unsqueeze(dim=0)

        # Variant types the patient has no variant of are simply absent.
        variant_x = {
            v_type: patient_data_x[v_type].to(self.device)
            for v_type in self.variant_types
            if v_type in patient_data_x
        }
        if not variant_x:
            return

        instance_labels_by_type = {
            v_type: patient_data_y[v_type].to(self.device) for v_type in variant_x
        }
        # [batch_size, n_variants]
        instance_labels = torch.cat(list(instance_labels_by_type.values()), dim=1)

        bag_logit, instance_logits = self.model(
            x=variant_x,
            v_embeddings=patient_data_x["v_embeddings"].to(self.device),
            p_embeddings=patient_data_x["p_embeddings"].to(self.device),
            gfe_known_mask=patient_data_x["gfe_known_mask"].to(self.device),
        )
        instance_probs = torch.sigmoid(instance_logits)

        loss_cfg = self.cfg.trainer
        bag_loss = F.binary_cross_entropy_with_logits(bag_logit, bag_label)
        bag_loss = bag_loss * loss_cfg.bag_loss_weight
        loss = bag_loss
        self.loss_n_metrics["bag_loss"].update(bag_loss.item(), 1)

        instance_loss = focal_loss(
            instance_probs,
            instance_labels,
            gamma=loss_cfg.focal_loss.gamma,
            alpha=loss_cfg.focal_loss.alpha,
        )
        instance_loss = instance_loss * loss_cfg.instance_loss_weight
        loss = loss + instance_loss
        self.loss_n_metrics["instance_loss"].update(instance_loss.item(), 1)

        # Ranking only has a target on patients with a causal variant.
        causal_variant_types = [
            v_type
            for v_type, labels in instance_labels_by_type.items()
            if bool(torch.any(labels))
        ]
        if causal_variant_types:
            ranknet_loss = pointwise_ranknet_loss(
                instance_probs,
                instance_labels,
                sigma=loss_cfg.ranknet_loss.sigma,
            )
            ranknet_loss = ranknet_loss * loss_cfg.ranknet_loss.loss_weight
            loss = loss + ranknet_loss
            self.loss_n_metrics["ranknet_loss"].update(ranknet_loss.item(), 1)

        if self.phase == "train":
            scaled_loss = loss / self.accumulation_steps
            scaled_loss.backward()

            # Off-by-one, known and deliberately not fixed: `step` is 1-based,
            # so the first window closes after 15 bags instead of 16 and every
            # later boundary is shifted with it. Left exactly as the published
            # model was trained - correcting it changes which bags share an
            # update, and with it the whole trajectory.
            if (step + 1) % self.accumulation_steps == 0 or step == self.total_steps:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()

        self.loss_n_metrics["total_loss"].update(loss.item(), 1)

        if not causal_variant_types:
            return

        # Over raw instances, while the benchmark of `evaluate.py` collapses the
        # rows sharing a CPRA first (`calculate_multi_k_topk_recall_by_level`).
        # The two are not the same number: this one is for watching training.
        recalls = calculate_multi_k_topk_recall(
            instance_probs.detach().cpu().numpy().squeeze(axis=0),
            instance_labels.detach().cpu().numpy().squeeze(axis=0),
            max_k=self.top_k,
        )
        recall = recalls.get(self.top_k)
        if recall is None:
            return

        self.loss_n_metrics[f"top_{self.top_k}_recall"].update(recall)
        for v_type in causal_variant_types:
            self.loss_n_metrics[f"top_{self.top_k}_recall_{v_type}"].update(recall)

        return

    def run_epoch(
        self,
        phase: Literal["train", "val"],
        epoch: int,
        dataloader: torch.utils.data.DataLoader,
    ) -> dict[str, float]:
        """Run one epoch.

        Args:
            phase (Literal["train", "val"]): Whether to update the weights.
            epoch (int): Epoch number, shown on the progress bar.
            dataloader (torch.utils.data.DataLoader): One patient per batch.

        Returns:
            dict[str, float]: Average of every loss term and of top-k recall,
                overall and per variant type.
        """
        self.phase = phase
        self.total_steps = len(dataloader)
        self.loss_n_metrics = defaultdict(AverageMeter)

        self.model.train() if phase == "train" else self.model.eval()
        self.optimizer.zero_grad()

        pbar = tqdm(
            enumerate(dataloader, start=1),
            total=self.total_steps,
            desc=f"{phase} {epoch}",
        )
        with torch.set_grad_enabled(phase == "train"):
            for step, batch in pbar:
                self.step(step=step, batch=batch)
                pbar.set_postfix_str(
                    " | ".join(
                        f"{name}: {self.loss_n_metrics[name].avg:.5f}"
                        for name in self.metric_names
                    )
                )
        pbar.close()

        return {
            f"{phase}_{name}_avg": self.loss_n_metrics[name].avg
            for name in self.metric_names
        }
