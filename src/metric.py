# модель и функция потерь работают с таргетом типа float, а для метрики нужен int
import torch
import torchmetrics
from torch import nn


class AUROC(nn.Module):
    def __init__(self, num_labels: int) -> None:
        super().__init__()
        self.metric = torchmetrics.AUROC(task='multilabel', num_labels=num_labels)

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.metric(preds, target.int())  # we are here for this line

    def compute(self) -> torch.Tensor:
        return self.metric.compute()

    def reset(self) -> None:
        self.metric.reset()
