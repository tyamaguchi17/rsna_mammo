## 前提
nvidia-driverが入っていることとdocker composeが使える必要がある。
repositoryをcloneする。
```
git clone https://github.com/tyamaguchi17/rsna_mammo.git --recursive
```

## Directory構成
```
$ ls
data  rsna_mammo  results
```
のようにinput/output directoryとrepositoryを同じ階層に配置する。dataのdirectory構造は `f"data/{patient_id}/{image_id}.png"`のようにする。コンペデータとvindrが同様の形式で配置されていることを想定しており、https://www.kaggle.com/datasets/analokamus/rsna-mammo-vindr-2048 を落としてきてsymblic linkを貼る等すると良い。


## 実行環境
(tmux上で)docker composeコマンドでcontainerを作る。
```
$ cd rsna_mammo
$ docker compose -f docker/docker-compose.yaml up
```

containerに入る
```
$ docker exec -it (container_id) bash
```

containerの中でrepository内に入り必要なライブラリをinstallする。
```
root@(container_id):/home/working# cd rsna_mammo
root@(container_id):/home/working/rsna_mammo# ls ..
data  rsna_mammo  results
root@(container_id):/home/working/rsna_mammo# pip install -r requirements.txt
```

必要に応じてwandbにloginする。
```
root@(container_id):/home/working/rsna_mammo# wandb login
```

## 学習コマンド
- shファイルを実行する方法
```
root@(container_id):/home/working/rsna_mammo# bash run/conf/exp/exp_xx_yy.bash
```
- pythonコマンドで実行する方法(一例)
```
root@(container_id):/home/working/rsna_mammo# python -m run.train \
    dataset.num_folds=4 \
    dataset.test_fold=0 \
    dataset.use_cache=false \
    training.batch_size=16 \
    training.batch_size_test=32 \
    training.epoch=10 \
    preprocessing.h_resize_to=1024 \
    preprocessing.w_resize_to=512 \
    augmentation.use_light_aug=true \
    model.base_model=convnext_base.fb_in22k_ft_in1k \
    training.use_wandb=true \
    training.num_workers=24 \
    training.num_gpus=1 \
    optimizer.lr=2e-5 \
    scheduler.warmup_steps_ratio=0.0 \
    training.accumulate_grad_batches=2 \
    out_dir=../results/convnext_base_baseline_fold_0
```

複数gpuを使う場合は`training.num_gpus`を変更し、`training.batch_size`と`training.accumulate_grad_batches`の積が一定になるように`training.batch_size`を大きくすると良さそう。

## 学習後
学習が終わると実行時に設定したoutdir内に結果ファイルが格納される。学習終了時にはema weightでscore計算が行われる。outdir内のdirectory構成は
```
$ ls
rsna_mammo  rsna_mammo.log  test_results  wandb  weights
```

`${outdir}/rsna_mammo/${job_id}/checkpoint`内にckptとweightが保存される。`model_weights.pth`がbest epoch (aucとpr_aucの平均をmonitorしている)の重み、`model_weights_ema.pth`がemaされた重み。

`(outdir)/weights`に各epochの重み、`${outdir}/test_results`に各epochの推論結果とema weightで推論した結果のcsvファイルが格納される。ファイル名が`results_breast.csv`で終わっているものがbreast levelの予測結果で `results.csv`で終わっているのがimage levelの予測結果。ema weightで予測した結果のファイル名は `test_results_breast.csv` と `test_results.csv`

次のコマンドでweightとoofを1つのdirectoryにまとめることができる(`outdir=${exp_name}_fold_${FOLD}`のように設定しているとする)。
```
root@(container_id):/home/working/rsna_mammo# python tools/collect_results.py --exp_name ${exp_name}
```

`/home/working`以下に`${exp_name}`というdirectoryができ、ema weight, config file, oof.csvが格納され、kaggle datasetにuploadすることでsubmissionにそのまま使用できる。
