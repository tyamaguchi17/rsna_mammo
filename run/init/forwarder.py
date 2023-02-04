from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor



class Forwarder(nn.Module):
    def __init__(self, cfg: DictConfig, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.cfg = cfg

    def loss(
        self,
        logits,
        labels,
    ) -> Tensor:

        loss = F.binary_cross_entropy_with_logits(
            logits.view(-1, 1), labels.view(-1, 1), reduction="none"
        )  # (B, C)

        if mean:
            return torch.mean(loss)
        else:
            return loss

    def forward(
        self, batch: Dict[str, Tensor], phase: str, epoch=None
    ) -> Tuple[Tensor, Tensor]:

        # inputs: Input tensor.
        inputs = batch["image"]

        # labels: Target labels of shape (B, C) where C is the number of classes.
        labels = batch["label"]

        if phase == "train":
            with torch.set_grad_enabled(True):
                embed_features = self.model.forward_features(inputs)
                logits = self.model.head(embed_features)
            loss = self.loss(
                logits=logits,
                labels=labels,
            )
        else:
            embed_features = self.model.forward_features(inputs)
            logits = self.model.head(embed_features)

            loss = self.loss(
                logits=logits,
                labels=labels,
            )

        return logits, loss, embed_features
