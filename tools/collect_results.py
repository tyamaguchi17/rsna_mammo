import argparse
import glob
import shutil
from pathlib import Path

import pandas as pd


def parse():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--fold", type=int, default=4)
    args = parser.parse_args()
    return args


def main(args):
    exp_name = args.exp_name
    fold = args.fold
    outdir = Path(f"../{exp_name}")
    outdir.mkdir(exist_ok=True)
    oof = pd.DataFrame()
    train_df = pd.read_csv("./fold/train_with_fold.csv")
    train_df["prediction_id"] = train_df["patient_id"].astype(str) + "_" + train_df["laterality"]
    train_df = train_df[["prediction_id", "cancer", "site_id", "age"]]
    train_df = train_df.drop_duplicates().reset_index(drop=True)
    if fold > 0:
        for i in range(fold):
            fold_dir = outdir / f"fold_{i}"
            fold_dir.mkdir(exist_ok=True)
            weights_path = glob.glob(
                f"../results/{exp_name}_fold_{i}/**/model_weights_ema.pth",
                recursive=True,
            )[0]
            shutil.copyfile(weights_path, fold_dir / "model_weights_ema.pth")
            config_path = glob.glob(
                f"../results/{exp_name}_fold_{i}/.hydra/config.yaml",
                recursive=True,
            )[0]
            shutil.copyfile(config_path, fold_dir / "config.yaml")
            oof_path = glob.glob(
                f"../results/{exp_name}_fold_{i}/**/test_results_breast.csv",
                recursive=True,
            )[0]
            df = pd.read_csv(oof_path)
            df["patient_id"] = df["patient_id"].astype(int)
            df["laterality"] = df["laterality"].map({0: "L", 1: "R"})
            df["prediction_id"] = df["patient_id"].astype(str) + "_" + df["laterality"]
            df["fold"] = i
            df["cancer"] = df["label"]
            df = df.drop("label", axis=1)
            oof = pd.concat([oof, df]).reset_index(drop=True)
        oof = oof.merge(train_df[["prediction_id", "site_id"]], on="prediction_id")
        oof.to_csv(outdir / "oof.csv", index=False)
    else:
        weights_path = glob.glob(f"../results/{exp_name}/**/model_weights_ema.pth")[0]
        shutil.copyfile(weights_path, outdir / "model_weights_ema.pth")


if __name__ == "__main__":
    args = parse()
    main(args)
