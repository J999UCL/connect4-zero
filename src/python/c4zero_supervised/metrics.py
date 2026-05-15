from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch.nn import functional as F

from c4zero_supervised.data import SupervisedBatch
from c4zero_train.model import AlphaZeroNet


@dataclass(slots=True)
class MetricTotals:
    samples: int = 0
    policy_loss_sum: float = 0.0
    top1_correct: int = 0
    top3_correct: int = 0
    target_prob_sum: float = 0.0
    illegal_prob_sum: float = 0.0
    category_totals: dict[str, "MetricTotals"] = field(default_factory=dict)

    def update(self, other: "MetricTotals") -> None:
        self.samples += other.samples
        self.policy_loss_sum += other.policy_loss_sum
        self.top1_correct += other.top1_correct
        self.top3_correct += other.top3_correct
        self.target_prob_sum += other.target_prob_sum
        self.illegal_prob_sum += other.illegal_prob_sum

    def as_dict(self) -> dict[str, float | int | dict]:
        if self.samples == 0:
            return {
                "samples": 0,
                "policy_loss": 0.0,
                "top1_target_accuracy": 0.0,
                "top3_target_accuracy": 0.0,
                "mean_target_probability": 0.0,
                "mean_illegal_probability": 0.0,
            }
        payload: dict[str, float | int | dict] = {
            "samples": self.samples,
            "policy_loss": self.policy_loss_sum / self.samples,
            "top1_target_accuracy": self.top1_correct / self.samples,
            "top3_target_accuracy": self.top3_correct / self.samples,
            "mean_target_probability": self.target_prob_sum / self.samples,
            "mean_illegal_probability": self.illegal_prob_sum / self.samples,
        }
        if self.category_totals:
            payload["by_category"] = {
                name: totals.as_dict()
                for name, totals in sorted(self.category_totals.items())
            }
        return payload


def forward_policy_logits(model: AlphaZeroNet, inputs: torch.Tensor) -> torch.Tensor:
    x = torch.relu(model.stem_bn(model.stem_conv(inputs)))
    x = model.tower(x)
    policy = torch.relu(model.policy_bn(model.policy_conv(x)))
    policy = torch.flatten(policy, start_dim=1)
    return model.policy_fc(policy)


def policy_loss(policy_logits: torch.Tensor, target_policy: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(policy_logits, dim=1)
    return -(target_policy * log_probs).sum(dim=1).mean()


def batch_metrics(
    policy_logits: torch.Tensor,
    batch: SupervisedBatch,
    category_names: tuple[str, ...],
) -> MetricTotals:
    target_mask = batch.target_policy > 0.0
    batch_size = int(batch.target_policy.shape[0])
    per_sample_loss = -(batch.target_policy * F.log_softmax(policy_logits, dim=1)).sum(dim=1)
    probs = torch.softmax(policy_logits, dim=1)

    topk = torch.topk(policy_logits, k=3, dim=1).indices
    top1 = topk[:, 0]
    top1_correct = target_mask.gather(1, top1[:, None]).squeeze(1)
    top3_correct = target_mask.gather(1, topk).any(dim=1)
    target_prob = (probs * target_mask).sum(dim=1)

    action_ids = torch.arange(16, device=batch.legal_mask.device, dtype=torch.int64)
    legal_mask = ((batch.legal_mask[:, None] >> action_ids[None, :]) & 1).bool()
    illegal_prob = (probs * (~legal_mask)).sum(dim=1)

    totals = MetricTotals(
        samples=batch_size,
        policy_loss_sum=float(per_sample_loss.detach().sum().cpu()),
        top1_correct=int(top1_correct.detach().sum().cpu()),
        top3_correct=int(top3_correct.detach().sum().cpu()),
        target_prob_sum=float(target_prob.detach().sum().cpu()),
        illegal_prob_sum=float(illegal_prob.detach().sum().cpu()),
    )

    for index, name in enumerate(category_names):
        mask = batch.category_id == index
        count = int(mask.sum().detach().cpu())
        if count == 0:
            continue
        totals.category_totals[name] = MetricTotals(
            samples=count,
            policy_loss_sum=float(per_sample_loss[mask].detach().sum().cpu()),
            top1_correct=int(top1_correct[mask].detach().sum().cpu()),
            top3_correct=int(top3_correct[mask].detach().sum().cpu()),
            target_prob_sum=float(target_prob[mask].detach().sum().cpu()),
            illegal_prob_sum=float(illegal_prob[mask].detach().sum().cpu()),
        )
    return totals
