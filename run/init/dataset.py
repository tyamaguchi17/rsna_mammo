from typing import Dict

from omegaconf import DictConfig

from src.datasets.rsna_mammo import RSNADataset


def init_datasets_from_config(cfg: DictConfig):
    if cfg.type == "rsna_mammo":
        datasets = get_rsna_dataset(
            num_folds=cfg.num_folds,
            test_fold=cfg.test_fold,
            val_fold=cfg.val_fold,
            seed=cfg.seed,
            num_records=cfg.num_records,
            phase=cfg.phase,
            cfg=cfg,
        )
    else:
        raise ValueError(f"Unknown dataset type: {cfg.type}")

    return datasets


def get_rsna_dataset(
    num_folds: int,
    test_fold: int,
    val_fold: int,
    seed: int = 2023,
    num_records: int = 0,
    phase: str = "train",
    cfg=None,
) -> Dict[str, RSNADataset]:

    df = RSNADataset.create_dataframe(
        num_folds,
        seed,
        num_records,
        fold_path=cfg.fold_path,
    )

    if phase == "train":
        train_df = df[(df["fold"] != val_fold) & (df["fold"] != test_fold)]
        val_df = df[df["fold"] == val_fold]
        test_df = df[df["fold"] == test_fold]
        train_positive = train_df[train_df["cancer"]==1]

        train_dataset = RSNADataset(train_df, phase="train", cfg=cfg)
        val_dataset = RSNADataset(val_df, phase="test", cfg=cfg)
        test_dataset = RSNADataset(test_df, phase="test", cfg=cfg)
        train_positive_dataset = RSNADataset(train_positive, phase="train", cfg=cfg)
    elif phase == "valid":
        train_dataset = RSNADataset(train_df, phase="test", cfg=cfg)
        val_dataset = RSNADataset(train_df, phase="test", cfg=cfg)
        test_dataset = RSNADataset(train_df, phase="test", cfg=cfg)
    elif phase == "test":
        train_dataset = RSNADataset(test_df, phase="test", cfg=cfg)
        val_dataset = RSNADataset(test_df, phase="test", cfg=cfg)
        test_dataset = RSNADataset(test_df, phase="test", cfg=cfg)
    elif phase == "vindr":
        df_vindr = RSNADataset.create_dataframe(data_type="vindr")
        train_dataset = RSNADataset(df_vindr, phase="test", cfg=cfg, data_name="vindr")
        val_dataset = RSNADataset(df_vindr, phase="test", cfg=cfg, data_name="vindr")
        test_dataset = RSNADataset(df_vindr, phase="test", cfg=cfg, data_name="vindr")

    datasets = {"train": train_dataset, "val": val_dataset, "test": test_dataset}
    if phase == "train":
        dataset["train_positive"] = train_positive_dataset
    return datasets
