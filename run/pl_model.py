from logging import getLogger
from pathlib import Path
from typing import Dict, List, Literal

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader

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

    predictions = predictions.copy()
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
    if y_true_count:
        c_recall = ctp / y_true_count
    else:
        c_recall = 0
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
            if phase == "train":
                train_dataset = WrapperDataset(
                    raw_datasets["train"],
                    transforms["train"],
                    "train",
                    view=cfg.dataset.view,
                )
                pos_cnt = train_dataset.base.df["cancer"].sum() * (
                    cfg.dataset.positive_aug_num + 1
                )
                if cfg.dataset.positive_aug_num > 0:
                    train_positive_dataset = WrapperDataset(
                        raw_datasets["train_positive"],
                        transforms["train"],
                        "train",
                        view=cfg.dataset.view,
                    )
                    train_dataset = [train_dataset] + [
                        train_positive_dataset
                        for _ in range(cfg.dataset.positive_aug_num)
                    ]
                    train_dataset = ConcatDataset(train_dataset)
                self.datasets["train"] = train_dataset
                logger.info(f"{phase}: {len(self.datasets[phase])}")
                logger.info(f"{phase} positive records: {pos_cnt}")
            else:
                self.datasets[phase] = WrapperDataset(
                    raw_datasets[phase], transforms[phase], phase
                )
                logger.info(f"{phase}: {len(self.datasets[phase])}")
                logger.info(f"{phase} positive records: {pos_cnt}")

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
        # self.log(
        #     "lr_head",
        #     sch.get_last_lr()[1],
        #     on_step=True,
        #     on_epoch=False,
        #     prog_bar=True,
        #     logger=True,
        #     sync_dist=True,
        #     batch_size=1,
        # )

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
            "pred_biopsy",
            "pred_invasive",
            "pred_age",
            "pred_machine_id",
            "pred_site_id",
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
                "original_index": epoch_results["original_index"]
                .reshape(-1)
                .astype(int),
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
        df["pred_age"] = (
            sigmoid(epoch_results["pred_age"].reshape(-1)) * 90
        )
        df["pred_machine_id"] = (
            epoch_results["pred_machine_id"].argmax(axis=1).reshape(-1)
        )
        df["pred_site_id"] = (epoch_results["pred_site_id"][:, 0] > 0).reshape(-1) + 1
        df = (
            df.drop_duplicates()
            .groupby(by=["original_index"])
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
            if "patient_id" in df.columns:
                df = df[["patient_id", "laterality", "label", "pred"]]
                df = df.groupby(by=["patient_id", "laterality"]).mean().reset_index()
                df.to_csv(test_results_filepath / "test_results_view.csv", index=False)
            else:
                if self.datasets[phase].base.data_name == "vindr":
                    df_vindr = pd.read_csv("./vindr/vindr_train.csv")
                    df_vindr["cancer"] = df["pred"]
                    df_vindr.to_csv(
                        test_results_filepath / "vinder_pl.csv", index=False
                    )

        loss = (
            torch.cat([torch.atleast_1d(x["loss"]) for x in outputs])
            .detach()
            .cpu()
            .numpy()
        )
        mean_loss = np.mean(loss)

        if phase != "test" and self.trainer.global_rank == 0:
            test_results_filepath = Path(self.cfg.out_dir) / "test_results"
            if not test_results_filepath.exists():
                test_results_filepath.mkdir(exist_ok=True)
            df.to_csv(
                test_results_filepath / f"epoch_{self.current_epoch}_results_view.csv",
                index=False,
            )
            df = df[["patient_id", "laterality", "label", "pred"]]
            df = df.groupby(by=["patient_id", "laterality"]).mean().reset_index()
            df.to_csv(
                test_results_filepath / f"epoch_{self.current_epoch}_results.csv",
                index=False,
            )
            weights_filepath = Path(self.cfg.out_dir) / "weights"
            if not weights_filepath.exists():
                weights_filepath.mkdir(exist_ok=True)
            weights_path = str(
                weights_filepath / f"model_weights_epoch_{self.current_epoch}.pth"
            )
            # logger.info(f"Extracting and saving weights: {weights_path}")
            torch.save(self.forwarder.model.state_dict(), weights_path)

        pred = df["pred"].values
        label = df["label"].values
        pf_score_000 = pf_score(label, pred)
        f1_score_980 = pf_score(label, pred, percentile=98.0, bin=True)
        f1_score_981 = pf_score(label, pred, percentile=98.1, bin=True)
        f1_score_982 = pf_score(label, pred, percentile=98.2, bin=True)
        f1_score_983 = pf_score(label, pred, percentile=98.3, bin=True)
        f1_score_984 = pf_score(label, pred, percentile=98.4, bin=True)
        max_f1_score = max(
            f1_score_980, f1_score_981, f1_score_982, f1_score_983, f1_score_984
        )
        try:
            auc_score = roc_auc_score(label.reshape(-1), pred.reshape(-1))
        except Exception:
            auc_score = 0

        try:
            pr_auc_score = average_precision_score(label.reshape(-1), pred.reshape(-1))
        except Exception:
            auc_score = 0

        mean_auc_score = (auc_score + pr_auc_score) / 2

        # Log items
        self.log(f"{phase}/loss", mean_loss, prog_bar=True)
        self.log(f"{phase}/pf_score", pf_score_000, prog_bar=False)
        self.log(f"{phase}/f1_score_980", f1_score_980, prog_bar=False)
        self.log(f"{phase}/f1_score_981", f1_score_981, prog_bar=False)
        self.log(f"{phase}/f1_score_982", f1_score_982, prog_bar=False)
        self.log(f"{phase}/f1_score_983", f1_score_983, prog_bar=False)
        self.log(f"{phase}/f1_score_984", f1_score_984, prog_bar=False)
        self.log(f"{phase}/f1_score", max_f1_score, prog_bar=True)
        self.log(f"{phase}/pr_auc", pr_auc_score, prog_bar=True)
        self.log(f"{phase}/auc", auc_score, prog_bar=True)
        self.log(f"{phase}/mean_auc", mean_auc_score, prog_bar=True)

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

    def on_before_zero_grad(self, *args, **kwargs):
        self.forwarder.ema.update(self.forwarder.model.parameters())

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
