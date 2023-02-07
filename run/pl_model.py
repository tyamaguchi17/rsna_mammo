from logging import getLogger
from pathlib import Path
from typing import Dict, List, Literal

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from sklearn.metrics import roc_auc_score
from torch import Tensor
from torch.utils.data import DataLoader

from run.init.dataset import init_datasets_from_config
from run.init.forwarder import Forwarder
from run.init.model import init_model_from_config
from run.init.optimizer import init_optimizer_from_config
from run.init.preprocessing import Preprocessing
from run.init.scheduler import init_scheduler_from_config
from src.datasets.wrapper import WrapperDataset

logger = getLogger(__name__)


def pf_score(labels, predictions, percentile=0, bin=False):
    beta = 1
    y_true_count = 0
    ctp = 0
    cfp = 0

    th = np.percentile(predictions, percentile)
    predictions[np.where(predictions < th)] = 0
    if bin:
        predictions[np.where(predictions >= th)] = 1

    for idx in range(len(labels)):
        prediction = min(max(predictions[idx], 0), 1)
        if labels[idx]:
            y_true_count += 1
            ctp += prediction
        else:
            cfp += prediction

    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if c_precision > 0 and c_recall > 0:
        result = (
            (1 + beta_squared)
            * (c_precision * c_recall)
            / (beta_squared * c_precision + c_recall)
        )
        return result
    else:
        return 0


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class PLModel(LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg.copy()
        self.save_embed = self.cfg.training.save_embed

        pretrained = False if cfg.training.debug else True
        model = init_model_from_config(cfg.model, pretrained=pretrained)
        self.forwarder = Forwarder(cfg.forwarder, model)

        raw_datasets = init_datasets_from_config(cfg.dataset)

        preprocessing = Preprocessing(cfg.augmentation, **cfg.preprocessing)
        self.datasets = {}
        transforms = {
            "train": preprocessing.get_train_transform(),
            "val": preprocessing.get_val_transform(),
            "test": preprocessing.get_test_transform(),
        }
        for phase in ["train", "val", "test"]:
            self.datasets[phase] = WrapperDataset(
                raw_datasets[phase], transforms[phase], phase
            )
            logger.info(f"{phase}: {len(self.datasets[phase])}")

        logger.info(
            f"training steps per epoch: {len(self.datasets['train'])/cfg.training.batch_size}"
        )
        self.cfg.scheduler.num_steps_per_epoch = (
            len(self.datasets["train"]) / cfg.training.batch_size
        )

    def on_train_epoch_start(self):
        pass

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int):
        additional_info = {}
        _, loss, _, _, _, _, _, _ = self.forwarder.forward(
            batch, phase="train", epoch=self.current_epoch, **additional_info
        )

        self.log(
            "train_loss",
            loss.detach().item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        sch = self.lr_schedulers()
        sch.step()
        self.log(
            "lr",
            sch.get_last_lr()[0],
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=1,
        )

        return loss

    def _end_process(self, outputs: List[Dict[str, Tensor]], phase: str):
        # Aggregate results
        epoch_results: Dict[str, np.ndarray] = {}
        outputs = self.all_gather(outputs)

        for key in [
            "original_index",
            "image_id",
            "image_id_2",
            "patient_id",
            "laterality",
            "label",
            "pred",
            "embed_features",
        ]:
            if key == "embed_features":
                if not self.save_embed:
                    continue
            if isinstance(outputs[0][key], Tensor):
                result = torch.cat([torch.atleast_1d(x[key]) for x in outputs], dim=1)
                result = torch.flatten(result, end_dim=1)
                epoch_results[key] = result.detach().cpu().numpy()
            else:
                result = np.concatenate([x[key] for x in outputs])
                epoch_results[key] = result

        df = pd.DataFrame(
            data={
                "original_index": epoch_results["original_index"].reshape(-1),
                "image_id": epoch_results["image_id"].reshape(-1),
                "image_id_2": epoch_results["image_id_2"].reshape(-1),
                "patient_id": epoch_results["patient_id"].reshape(-1),
                "laterality": epoch_results["laterality"].reshape(-1),
            }
        )
        df["pred"] = sigmoid(epoch_results["pred"][:, 0].reshape(-1))
        df["label"] = epoch_results["label"]
        df["pred_biopsy"] = sigmoid(epoch_results["pred_biopsy"][:, 0].reshape(-1))
        df["pred_invasive"] = sigmoid(epoch_results["pred_invasive"][:, 0].reshape(-1))
        df["pred_age"] = epoch_results["pred_invasive"].argmax(axis=1).reshape(-1) * 3
        df["pred_machine_id"] = (
            epoch_results["pred_machine_id"].argmax(axis=1).reshape(-1)
        )
        df["pred_site_id"] = (epoch_results["pred_site_id"][:, 0] > 0).reshape(-1) + 1
        df = (
            df.drop_duplicates()
            .groupby(by="original_index")
            .mean()
            .reset_index()
            .sort_values(by="original_index")
        )
        if phase == "test" and self.trainer.global_rank == 0:
            # Save test results ".npz" format
            test_results_filepath = Path(self.cfg.out_dir) / "test_results"
            if not test_results_filepath.exists():
                test_results_filepath.mkdir(exist_ok=True)
            np.savez_compressed(
                str(test_results_filepath / "test_results.npz"),
                **epoch_results,
            )
            df.to_csv(test_results_filepath / "test_results.csv", index=False)

        loss = (
            torch.cat([torch.atleast_1d(x["loss"]) for x in outputs])
            .detach()
            .cpu()
            .numpy()
        )
        mean_loss = np.mean(loss)

        df = df[["patient_id", "laterality", "label", "pred"]]
        df = df.groupby(by=["patient_id", "laterality"]).mean().reset_index()
        if phase != "test" and self.trainer.global_rank == 0:
            test_results_filepath = Path(self.cfg.out_dir) / "test_results"
            if not test_results_filepath.exists():
                test_results_filepath.mkdir(exist_ok=True)
            df.to_csv(
                test_results_filepath / f"epoch_{self.current_epoch}_results.csv",
                index=False,
            )

        pred = df["pred"].values
        label = df["label"].values
        pf_score_000 = pf_score(label, pred)
        pf_score_985 = pf_score(label, pred, percentile=98.5)
        pf_score_983 = pf_score(label, pred, percentile=98.3)
        pf_score_980 = pf_score(label, pred, percentile=98.0)
        f1_score_983 = pf_score(label, pred, percentile=98.3, bin=True)
        try:
            auc_score = roc_auc_score(label.reshape(-1), pred.reshape(-1))
        except Exception:
            auc_score = 0

        # Log items
        self.log(f"{phase}/loss", mean_loss, prog_bar=True)
        self.log(f"{phase}/pf_score", pf_score_000, prog_bar=True)
        self.log(f"{phase}/pf_score_985", pf_score_985, prog_bar=True)
        self.log(f"{phase}/pf_score_983", pf_score_983, prog_bar=True)
        self.log(f"{phase}/pf_score_980", pf_score_980, prog_bar=True)
        self.log(f"{phase}/f1_score_983", f1_score_983, prog_bar=True)
        self.log(f"{phase}/auc", auc_score, prog_bar=True)

    def _evaluation_step(self, batch: Dict[str, Tensor], phase: Literal["val", "test"]):
        (
            preds,
            loss,
            embed_features,
            preds_biopsy,
            preds_invasive,
            preds_age,
            preds_machine_id,
            preds_site_id,
        ) = self.forwarder.forward(batch, phase=phase, epoch=self.current_epoch)

        output = {
            "loss": loss,
            "label": batch["label"],
            "original_index": batch["original_index"],
            "patient_id": batch["patient_id"],
            "image_id": batch["image_id"],
            "image_id_2": batch["image_id_2"],
            "laterality": batch["laterality"],
            "pred": preds.detach(),
            "pred_biopsy": preds_biopsy.detach(),
            "pred_invasive": preds_invasive.detach(),
            "pred_age": preds_age.detach(),
            "pred_machine_id": preds_machine_id.detach(),
            "pred_site_id": preds_site_id.detach(),
            "embed_features": embed_features.detach(),
        }
        return output

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int):
        return self._evaluation_step(batch, phase="val")

    def validation_epoch_end(self, outputs: List[Dict[str, Tensor]]) -> None:
        self._end_process(outputs, "val")

    def test_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        return self._evaluation_step(batch, phase="test")

    def test_epoch_end(self, outputs: List[Dict[str, Tensor]]) -> None:
        self._end_process(outputs, "test")

    def configure_optimizers(self):
        model = self.forwarder.model
        opt_cls, kwargs = init_optimizer_from_config(
            self.cfg.optimizer, model.forward_features.parameters()
        )

        self.cfg.optimizer.lr = self.cfg.optimizer.lr_head
        kwargs_head = init_optimizer_from_config(
            self.cfg.optimizer, model.head.parameters(), return_cls=False
        )

        optimizer = opt_cls([kwargs, kwargs_head])
        scheduler = init_scheduler_from_config(self.cfg.scheduler, optimizer)

        if scheduler is None:
            return [optimizer]
        return [optimizer], [scheduler]

    def _dataloader(self, phase: str) -> DataLoader:
        logger.info(f"{phase} data loader called")
        dataset = self.datasets[phase]

        batch_size = self.cfg.training.batch_size
        num_workers = self.cfg.training.num_workers

        num_gpus = self.cfg.training.num_gpus
        if phase != "train":
            batch_size = self.cfg.training.batch_size_test
        batch_size //= num_gpus
        num_workers //= num_gpus

        drop_last = True if self.cfg.training.drop_last and phase == "train" else False
        shuffle = phase == "train"

        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
        )
        return loader

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(phase="train")

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(phase="val")

    def test_dataloader(self) -> DataLoader:
        return self._dataloader(phase="test")
