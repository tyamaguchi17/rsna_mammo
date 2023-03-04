## Prerequisites
Using docker compose is recommended.


Clone this repository
```
git clone https://github.com/tyamaguchi17/rsna_mammo.git --recursive
```

## Directory Structures
```
$ ls
data  rsna_mammo  results
```
Place the input/output directory and repository in the same directory as above. The directory structure of data should be like `f "data/{patient_id}/{image_id}.png"`. The image files (.png) are assumed to have been resized to (2048, 2048).


## Execution Environments
Use docker compose command and make a container.
```
$ cd rsna_mammo
$ docker compose -f docker/docker-compose.yaml up
```

Enter into the container.
```
$ docker exec -it (container_id) bash
```

Install required libraries
```
root@(container_id):/home/working# cd rsna_mammo
root@(container_id):/home/working/rsna_mammo# ls ..
data  rsna_mammo  results
root@(container_id):/home/working/rsna_mammo# pip install -r requirements.txt
```

Log in to wandb if you like.
```
root@(container_id):/home/working/rsna_mammo# wandb login
```

## Training Commands
- Using bash command
```
root@(container_id):/home/working/rsna_mammo# bash run/conf/exp/exp_xx_yy.sh
```
The models used in the final submission will be reproduced by
```
root@(container_id):/home/working/rsna_mammo# bash run/conf/exp/exp_final.sh
```

- Using python command (example)
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

## After Training
When training is completed, the result file is stored in the out_dir. At the end of training, score calculation is performed using ema weight.
```
$ ls
rsna_mammo  rsna_mammo.log  test_results  wandb  weights
```

Checkpoints and weights are stored in `${outdir}/rsna_mammo/${job_id}/checkpoint`. `model_weights.pth` is the weights of the best epoch (the average of auc and pr_auc is monitored.) and `model_weights_ema.pth` is the ema weights.

You can collect weights and oof into one directory with the following command (assuming you have set `out_dir=${exp_name}_fold_${FOLD}`. Example:  `exp_name=convnext_small_multi_lat_final`)
```
root@(container_id):/home/working/rsna_mammo# python tools/collect_results.py --exp_name ${exp_name}
```

A `${exp_name}` directory is created under `/home/working`, where ema weight, config file, and oof.csv are stored, and can be directly used for submission by uploading them to the kaggle dataset.

## Links
- For an overview of our key ideas and detailed explanation, please also refer to [6th place solution](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/discussion/390974) in Kaggle discussion.
- RabotniKuma part -> https://github.com/analokmaus/kaggle-rsna-breast-cancer
- YOLOX(ishikei) part -> https://github.com/ishikei14k/RSNA_Screening_Mammography_Breast_Cancer_Detection
