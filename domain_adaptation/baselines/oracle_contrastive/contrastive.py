"""Memory-bounded voxel-level instance contrastive loss."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _gumbel_topk_sample(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Sample without replacement using the Gumbel-Top-k trick."""
    uniform = torch.rand_like(logits).clamp_min(torch.finfo(logits.dtype).tiny)
    noise = -torch.log(-torch.log(uniform))
    return torch.topk(logits + noise, k=k, dim=1).indices


class PixelInstanceDiscriminationLoss(nn.Module):
    """Contrast sampled voxel features using integer instance IDs."""

    def __init__(
        self,
        *,
        sample_temperature: float = 0.6,
        sample_k: int = 512,
        discrimination_temperature: float = 0.3,
    ) -> None:
        super().__init__()
        if sample_k <= 1:
            raise ValueError("sample_k must be greater than one")
        if discrimination_temperature <= 0:
            raise ValueError("discrimination_temperature must be positive")
        self.sample_temperature = float(sample_temperature)
        self.sample_k = int(sample_k)
        self.discrimination_temperature = float(discrimination_temperature)

    def sample_voxels(
        self, feature: torch.Tensor, instance_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Area-balanced sample features as BxCxK and instance IDs as BxK."""
        if feature.ndim != 5:
            raise ValueError(f"feature must be BxCxDxHxW, got {tuple(feature.shape)}")
        if instance_ids.ndim == 5:
            if instance_ids.shape[1] != 1:
                raise ValueError("5-D instance_ids must have one channel")
            instance_ids = instance_ids[:, 0]
        if instance_ids.ndim != 4:
            raise ValueError(
                "instance_ids must be BxDxHxW or Bx1xDxHxW, "
                f"got {tuple(instance_ids.shape)}"
            )

        labels = F.interpolate(
            instance_ids[:, None].float(),
            size=feature.shape[-3:],
            mode="nearest",
        )[:, 0].long()
        flat_labels = labels.flatten(1)
        flat_feature = feature.flatten(2)
        valid_counts = (flat_labels > 0).sum(dim=1)
        k = min(self.sample_k, int(valid_counts.min().item()))
        if k <= 1:
            return flat_feature[:, :, :0], flat_labels[:, :0]

        sampling_logits = torch.full_like(flat_labels, -99999.0, dtype=torch.float32)
        for batch_index in range(flat_labels.shape[0]):
            valid = flat_labels[batch_index] > 0
            _, inverse, counts = torch.unique(
                flat_labels[batch_index, valid],
                return_inverse=True,
                return_counts=True,
            )
            inverse_area = valid.sum().float() / counts[inverse].float().clamp_min(1.0)
            sampling_logits[batch_index, valid] = (
                inverse_area.log() * self.sample_temperature
            )

        sample_indices = _gumbel_topk_sample(sampling_logits, k)
        sampled_labels = torch.gather(flat_labels, 1, sample_indices)
        sampled_feature = torch.gather(
            flat_feature,
            2,
            sample_indices[:, None].expand(-1, flat_feature.shape[1], -1),
        )
        return sampled_feature, sampled_labels

    def loss_from_samples(
        self, sampled_feature: torch.Tensor, sampled_labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute soft-target cross entropy from sampled voxel pairs."""
        if sampled_feature.ndim != 3 or sampled_labels.ndim != 2:
            raise ValueError("Expected sampled_feature BxCxK and sampled_labels BxK")
        if (
            sampled_feature.shape[0] != sampled_labels.shape[0]
            or sampled_feature.shape[2] != sampled_labels.shape[1]
        ):
            raise ValueError("Sampled feature and label shapes do not align")
        if sampled_feature.shape[2] <= 1:
            return sampled_feature.sum() * 0.0

        sampled_feature = F.normalize(
            sampled_feature.float(), p=2, dim=1, eps=1.0e-6
        )
        gt_similarity = (
            sampled_labels[:, :, None] == sampled_labels[:, None, :]
        ).float()
        gt_distribution = gt_similarity / gt_similarity.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        predicted_similarity = torch.einsum(
            "bck,bcj->bkj", sampled_feature, sampled_feature
        ) / self.discrimination_temperature
        return F.cross_entropy(predicted_similarity, gt_distribution, reduction="mean")


__all__ = ["PixelInstanceDiscriminationLoss"]
