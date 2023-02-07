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

    def loss_bce(
        self,
        logits,
        labels,
        mean=True,
    ) -> Tensor:

        loss = F.binary_cross_entropy_with_logits(
            logits.view(-1, 1), labels.view(-1, 1), reduction="none"
        )  # (B, 1)

        if mean:
            return torch.mean(loss)
        else:
            return loss

    def loss_ce(
        self,
        logits,
        labels,
        mean=True,
    ) -> Tensor:

        loss = F.cross_entropy(logits, labels, reduction="none")  # (B, C)

        if mean:
            return torch.mean(loss)
        else:
            return loss

    def loss(
        self,
        logits,
        logits_biopsy,
        logits_invasive,
        logits_age,
        logits_machine_id,
        logits_site_id,
        labels,
        labels_biospy,
        labels_invasive,
        labels_age,
        labels_machine_id,
        labels_site_id,
    ):
        cfg = self.cfg.loss
        loss = self.loss_bce(logits, labels) * cfg.cancer_weight
        loss += self.loss_bce(logits_biopsy, labels_biospy) * cfg.biopsy_weight
        loss += self.loss_bce(logits_invasive, labels_invasive) * cfg.invasive_weight
        loss += self.loss_ce(logits_age, labels_age) * cfg.age_weight
        loss += (
            self.loss_ce(logits_machine_id, labels_machine_id) * cfg.machine_id_weight
        )
        loss += self.loss_bce(logits_site_id, labels_site_id) * cfg.site_id_weight
        return loss

    def forward(
        self, batch: Dict[str, Tensor], phase: str, epoch=None
    ) -> Tuple[Tensor, Tensor]:

        # inputs: Input tensor.
        inputs = batch["image"]

        # labels
        labels = batch["label"]
        labels_biospy = batch["biopsy"]
        labels_invasive = batch["invasive"]
        labels_age = batch["age_3"]
        labels_machine_id = batch["machine_id_enc"]
        labels_site_id = batch["site_id"] - 1

        # LR model labels
        # labels_2 = batch["label_2"]
        # labels_biospy_2 = batch["biopsy_2"]
        # labels_invasive_2 = batch["invasive_2"]

        if phase == "train":
            with torch.set_grad_enabled(True):
                embed_features = self.model.forward_features(inputs)
                logits = self.model.head.head(embed_features)
                logits_biopsy = self.model.head.head_biopsy(embed_features)
                logits_invasive = self.model.head.head_invasive(embed_features)
                logits_age = self.model.head.head_age(embed_features)
                logits_machine_id = self.model.head.head_machine_id(embed_features)
                logits_site_id = self.model.head.head_site_id(embed_features)
                # logits_2 = self.model.head.head_2(embed_features)
                # logits_biopsy_2 = self.model.head.head_biopsy_2(embed_features)
                # logits_invasive_2 = self.model.head.head_invasive_2(embed_features)
            loss = self.loss(
                logits=logits,
                logits_biopsy=logits_biopsy,
                logits_invasive=logits_invasive,
                logits_age=logits_age,
                logits_machine_id=logits_machine_id,
                logits_site_id=logits_site_id,
                labels=labels,
                labels_biospy=labels_biospy,
                labels_invasive=labels_invasive,
                labels_age=labels_age,
                labels_machine_id=labels_machine_id,
                labels_site_id=labels_site_id,
            )
        else:
            embed_features = self.model.forward_features(inputs)
            logits = self.model.head.head(embed_features)
            logits_biopsy = self.model.head.head_biopsy(embed_features)
            logits_invasive = self.model.head.head_invasive(embed_features)
            logits_age = self.model.head.head_age(embed_features)
            logits_machine_id = self.model.head.head_machine_id(embed_features)
            logits_site_id = self.model.head.head_site_id(embed_features)
            # logits_2 = self.model.head.head_2(embed_features)
            # logits_biopsy_2 = self.model.head.head_biopsy_2(embed_features)
            # logits_invasive_2 = self.model.head.head_invasive_2(embed_features)

            loss = self.loss(
                logits=logits,
                logits_biopsy=logits_biopsy,
                logits_invasive=logits_invasive,
                logits_age=logits_age,
                logits_machine_id=logits_machine_id,
                logits_site_id=logits_site_id,
                labels=labels,
                labels_biospy=labels_biospy,
                labels_invasive=labels_invasive,
                labels_age=labels_age,
                labels_machine_id=labels_machine_id,
                labels_site_id=labels_site_id,
            )

        return (
            logits,
            loss,
            embed_features,
            logits_biopsy,
            logits_invasive,
            logits_age,
            logits_machine_id,
            logits_site_id,
        )
