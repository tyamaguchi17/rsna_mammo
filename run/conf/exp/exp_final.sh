for FOLD in {0..3}; do
python -m run.train \
  dataset.num_folds=4 \
  dataset.test_fold=$FOLD \
  dataset.augmentation.p_crop_resize=1.0 \
  dataset.augmentation.p_roi_crop=0.5 \
  dataset.use_multi_view=true \
  dataset.use_multi_lat=true \
  training.batch_size=8 \
  training.batch_size_test=8 \
  training.epoch=10 \
  preprocessing.h_resize_to=1024 \
  preprocessing.w_resize_to=512 \
  augmentation.use_light_aug=true \
  model.base_model=convnext_small.fb_in22k_ft_in1k \
  training.num_workers=24 \
  training.num_gpus=1 \
  optimizer.lr=5e-5 \
  optimizer.lr_head=5e-4 \
  scheduler.warmup_steps_ratio=0.0 \
  forwarder.loss.cancer_weight=8.0 \
  forwarder.loss.biopsy_weight=1.0 \
  forwarder.loss.invasive_weight=1.0 \
  forwarder.loss.birads_weight=0.0 \
  forwarder.loss.age_weight=0.5 \
  training.accumulate_grad_batches=8 \
  training.use_wandb=true \
  out_dir=../results/convnext_small_multi_lat_final_fold_$FOLD; done
