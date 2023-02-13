from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor
from torch_ema import ExponentialMovingAverage


class Forwarder(nn.Module):
    def __init__(self, cfg: DictConfig, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        # workaround for device inconsistency of ExponentialMovingAverage
        self.ema = None
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
        logits_birads,
        logits_age,
        logits_machine_id,
        logits_site_id,
        labels,
        labels_biospy,
        labels_invasive,
        labels_birads,
        labels_age,
        labels_machine_id,
        labels_site_id,
    ):
        cfg = self.cfg.loss
        loss = self.loss_bce(logits, labels) * cfg.cancer_weight
        loss += self.loss_bce(logits_biopsy, labels_biospy) * cfg.biopsy_weight
        loss += self.loss_bce(logits_invasive, labels_invasive) * cfg.invasive_weight
        loss += self.loss_bce(logits_age, labels_age) * cfg.age_weight
        loss += (
            self.loss_ce(logits_machine_id, labels_machine_id) * cfg.machine_id_weight
        )
        loss += self.loss_bce(logits_site_id, labels_site_id) * cfg.site_id_weight
        loss += self.loss_bce(logits_birads, labels_birads) * cfg.birads_weight
        return loss

    def forward(
        self, batch: Dict[str, Tensor], phase: str, epoch=None
    ) -> Tuple[Tensor, Tensor]:

        # workaround for device inconsistency of ExponentialMovingAverage
        if self.ema is None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=0.999)

        use_multi_view = self.cfg.use_multi_view
        use_multi_lat = self.cfg.use_multi_lat

        # inputs: Input tensor.
        inputs = batch["image"]

        if use_multi_view or use_multi_lat:
            bs, ch, h, w = inputs.shape
            inputs = inputs.view(bs * ch, 1, h, w)

        # labels
        labels = batch["label"].to(torch.float16)
        labels_biospy = batch["biopsy"].to(torch.float16)
        labels_invasive = batch["invasive"].to(torch.float16)
        labels_age = batch["age_scaled"].to(torch.float16)
        labels_machine_id = batch["machine_id_enc"]
        labels_site_id = (batch["site_id"] - 1).to(torch.float16)
        labels_birads = batch["BIRADS_scaled"].to(torch.float16)

        if use_multi_lat:
            # LR model labels
            labels_2 = batch["label_2"].to(torch.float16)
            labels_biospy_2 = batch["biopsy_2"].to(torch.float16)
            labels_invasive_2 = batch["invasive_2"].to(torch.float16)
            labels_birads_2 = batch["BIRADS_scaled_2"].to(torch.float16)

        if phase == "train":
            with torch.set_grad_enabled(True):
                embed_features = self.model.forward_features(inputs)
                if use_multi_view or use_multi_lat:
                    embed_features = embed_features.view(bs, -1)
                logits = self.model.head.head(embed_features)
                logits_biopsy = self.model.head.head_biopsy(embed_features)
                logits_invasive = self.model.head.head_invasive(embed_features)
                logits_birads = self.model.head.head_birads(embed_features)
                logits_age = self.model.head.head_age(embed_features)
                logits_machine_id = self.model.head.head_machine_id(embed_features)
                logits_site_id = self.model.head.head_site_id(embed_features)
            loss = self.loss(
                logits=logits,
                logits_biopsy=logits_biopsy,
                logits_invasive=logits_invasive,
                logits_birads=logits_birads,
                logits_age=logits_age,
                logits_machine_id=logits_machine_id,
                logits_site_id=logits_site_id,
                labels=labels,
                labels_biospy=labels_biospy,
                labels_invasive=labels_invasive,
                labels_birads=labels_birads,
                labels_age=labels_age,
                labels_machine_id=labels_machine_id,
                labels_site_id=labels_site_id,
            )
            if use_multi_lat:
                logits_2 = self.model.head.head_2(embed_features)
                logits_biopsy_2 = self.model.head.head_biopsy_2(embed_features)
                logits_invasive_2 = self.model.head.head_invasive_2(embed_features)
                logits_birads_2 = self.model.head.head_birads_2(embed_features)
                loss += self.loss(
                    logits=logits_2,
                    logits_biopsy=logits_biopsy_2,
                    logits_invasive=logits_invasive_2,
                    logits_birads=logits_birads_2,
                    logits_age=logits_age,
                    logits_machine_id=logits_machine_id,
                    logits_site_id=logits_site_id,
                    labels=labels_2,
                    labels_biospy=labels_biospy_2,
                    labels_invasive=labels_invasive_2,
                    labels_birads=labels_birads_2,
                    labels_age=labels_age,
                    labels_machine_id=labels_machine_id,
                    labels_site_id=labels_site_id,
                )
        else:
            if phase == "test":
                with self.ema.average_parameters():
                    embed_features = self.model.forward_features(inputs)
                    if use_multi_view or use_multi_lat:
                        embed_features = embed_features.view(bs, -1)
                    logits = self.model.head.head(embed_features)
                    logits_biopsy = self.model.head.head_biopsy(embed_features)
                    logits_invasive = self.model.head.head_invasive(embed_features)
                    logits_birads = self.model.head.head_birads(embed_features)
                    logits_age = self.model.head.head_age(embed_features)
                    logits_machine_id = self.model.head.head_machine_id(embed_features)
                    logits_site_id = self.model.head.head_site_id(embed_features)
                    if use_multi_lat:
                        logits_2 = self.model.head.head_2(embed_features)
                        logits_biopsy_2 = self.model.head.head_biopsy_2(embed_features)
                        logits_invasive_2 = self.model.head.head_invasive_2(
                            embed_features
                        )
                        logits_birads_2 = self.model.head.head_birads_2(embed_features)
            elif phase == "val":
                embed_features = self.model.forward_features(inputs)
                if use_multi_view:
                    embed_features = embed_features.view(bs, -1)
                logits = self.model.head.head(embed_features)
                logits_biopsy = self.model.head.head_biopsy(embed_features)
                logits_invasive = self.model.head.head_invasive(embed_features)
                logits_birads = self.model.head.head_birads(embed_features)
                logits_age = self.model.head.head_age(embed_features)
                logits_machine_id = self.model.head.head_machine_id(embed_features)
                logits_site_id = self.model.head.head_site_id(embed_features)
                if use_multi_lat:
                    logits_2 = self.model.head.head_2(embed_features)
                    logits_biopsy_2 = self.model.head.head_biopsy_2(embed_features)
                    logits_invasive_2 = self.model.head.head_invasive_2(embed_features)
                    logits_birads_2 = self.model.head.head_birads_2(embed_features)

            loss = self.loss(
                logits=logits,
                logits_biopsy=logits_biopsy,
                logits_invasive=logits_invasive,
                logits_birads=logits_birads,
                logits_age=logits_age,
                logits_machine_id=logits_machine_id,
                logits_site_id=logits_site_id,
                labels=labels,
                labels_biospy=labels_biospy,
                labels_invasive=labels_invasive,
                labels_birads=labels_birads,
                labels_age=labels_age,
                labels_machine_id=labels_machine_id,
                labels_site_id=labels_site_id,
            )
            if use_multi_lat:
                loss += self.loss(
                    logits=logits_2,
                    logits_biopsy=logits_biopsy_2,
                    logits_invasive=logits_invasive_2,
                    logits_birads=logits_birads_2,
                    logits_age=logits_age,
                    logits_machine_id=logits_machine_id,
                    logits_site_id=logits_site_id,
                    labels=labels_2,
                    labels_biospy=labels_biospy_2,
                    labels_invasive=labels_invasive_2,
                    labels_birads=labels_birads_2,
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
