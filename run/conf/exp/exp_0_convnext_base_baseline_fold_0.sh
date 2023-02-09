python -m run.train \
  dataset.num_folds=4 \
  dataset.test_fold=$FOLD \
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
  optimizer.lr=2e-5 \
  scheduler.warmup_steps_ratio=0.0 \
  training.accumulate_grad_batches=2 \
  out_dir=../results/convnext_base_baseline_fold_$FOLD
