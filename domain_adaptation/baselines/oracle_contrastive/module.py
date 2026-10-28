"""Lightning module for the fully supervised CREMI-A contrastive oracle."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from connectomics.training.lightning import ConnectomicsModule

from contrastive import PixelInstanceDiscriminationLoss


class ContrastiveOracleModule(ConnectomicsModule):
    """Add area-balanced instance discrimination to a MedNeXt decoder feature."""

    def __init__(self, cfg: Any, contrastive_cfg: Any) -> None:
        super().__init__(cfg)
        self.contrastive_weight = float(contrastive_cfg.weight)
        self.max_contrastive_batch_samples = int(contrastive_cfg.max_batch_samples)
        if self.max_contrastive_batch_samples <= 0:
            raise ValueError("max_batch_samples must be positive")
        self._contrastive_feature: torch.Tensor | None = None

        in_channels = int(contrastive_cfg.feature_channels)
        hidden_channels = int(contrastive_cfg.projection_hidden_channels)
        out_channels = int(contrastive_cfg.projection_channels)
        groups = min(8, hidden_channels)
        while hidden_channels % groups:
            groups -= 1
        self.contrastive_projector = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.GELU(),
            nn.Conv1d(hidden_channels, out_channels, kernel_size=1),
        )
        self.contrastive_loss = PixelInstanceDiscriminationLoss(
            sample_temperature=float(contrastive_cfg.sample_temperature),
            sample_k=int(contrastive_cfg.sample_k),
            discrimination_temperature=float(
                contrastive_cfg.instance_discrimination_temperature
            ),
        )

        modules = dict(self.model.named_modules())
        feature_layer = str(contrastive_cfg.feature_layer)
        if feature_layer not in modules:
            raise ValueError(
                f"Unknown contrastive feature layer {feature_layer!r}; "
                "inspect model.named_modules()"
            )
        self._feature_hook = modules[feature_layer].register_forward_hook(
            self._capture_contrastive_feature
        )

    def _capture_contrastive_feature(self, _module, _inputs, output) -> None:
        if not isinstance(output, torch.Tensor):
            raise TypeError("Contrastive feature hook must produce a tensor")
        self._contrastive_feature = output

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        images = batch["image"]
        labels = batch["label"]
        gt_seg = batch.get("gt_seg")
        if gt_seg is None:
            raise RuntimeError(
                "Contrastive training requires batch['gt_seg']; set "
                "data.label_transform.emit_gt_seg: true"
            )
        raw_mask = batch.get("mask")
        mask = (raw_mask > 0).float() if raw_mask is not None else None
        target_mask = batch.get("label_mask")

        self._contrastive_feature = None
        outputs = self(images)
        if self._contrastive_feature is None:
            raise RuntimeError("Contrastive feature hook did not run")

        supervised_loss, loss_dict = self._compute_loss(
            outputs,
            labels,
            stage="train",
            mask=mask,
            target_mask=target_mask,
            gt_seg=gt_seg,
        )
        contrastive_feature = self._contrastive_feature
        contrastive_gt_seg = gt_seg
        if contrastive_feature.shape[0] > self.max_contrastive_batch_samples:
            sample_indices = torch.randperm(
                contrastive_feature.shape[0], device=contrastive_feature.device
            )[: self.max_contrastive_batch_samples]
            contrastive_feature = contrastive_feature.index_select(0, sample_indices)
            contrastive_gt_seg = contrastive_gt_seg.index_select(0, sample_indices)

        sampled_feature, sampled_labels = self.contrastive_loss.sample_voxels(
            contrastive_feature, contrastive_gt_seg
        )
        if sampled_feature.shape[2] <= 1:
            contrastive_loss = contrastive_feature.sum() * 0.0
            contrastive_loss = contrastive_loss + sum(
                parameter.sum() * 0.0
                for parameter in self.contrastive_projector.parameters()
            )
        else:
            projected_feature = self.contrastive_projector(sampled_feature)
            contrastive_loss = self.contrastive_loss.loss_from_samples(
                projected_feature, sampled_labels
            )
        weighted_contrastive = self.contrastive_weight * contrastive_loss
        total_loss = supervised_loss + weighted_contrastive

        loss_dict["train_loss_supervised"] = supervised_loss.detach()
        loss_dict["train_loss_contrastive"] = contrastive_loss.detach()
        loss_dict["train_loss_contrastive_weighted"] = weighted_contrastive.detach()
        loss_dict["train_loss_total"] = total_loss.detach()
        self.log_dict(
            loss_dict,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=False,
        )
        return total_loss


__all__ = ["ContrastiveOracleModule"]
