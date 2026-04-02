python train_B.py --config config.yaml --train-mode mixed --loss-type dicece --label-setup 4c --checkpoint-suffix mixed_dicece_3to1 --max-val-cases 30


```bash
python train_B.py --config config.yaml --train-mode mixed --loss-type dicefocal --label-setup 4c --checkpoint-suffix mixed_dicefocal_3to1 --max-val-cases 30
```


Running Stage B: multi-epoch baseline training...
Checkpoint directory: /home/faculty1/Brain_Tumor_Segmentation/experiments/baseline_t1ce_multiclass/checkpoints/baseline_t1ce_multiclass_mixed_dicece_3to1
[epoch 1/10 | step 1/1189] loss: 0.931095
Traceback (most recent call last):
  File "/home/faculty1/Brain_Tumor_Segmentation/experiments/baseline_t1ce_multiclass/train_B.py", line 829, in <module>
    main()
    ~~~~^^
  File "/home/faculty1/Brain_Tumor_Segmentation/experiments/baseline_t1ce_multiclass/train_B.py", line 810, in main
    run_stage_b_training(
    ~~~~~~~~~~~~~~~~~~~~^
        loader=train_loader,
        ^^^^^^^^^^^^^^^^^^^^
    ...<10 lines>...
        reset_optimizer=args.reset_optimizer,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/home/faculty1/Brain_Tumor_Segmentation/experiments/baseline_t1ce_multiclass/train_B.py", line 610, in run_stage_b_training
    for step, batch in enumerate(loader, start=1):
                       ~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/faculty1/anaconda3/lib/python3.13/site-packages/torch/utils/data/dataloader.py", line 741, in __next__
    data = self._next_data()
  File "/home/faculty1/anaconda3/lib/python3.13/site-packages/torch/utils/data/dataloader.py", line 1548, in _next_data
    return self._process_data(data, worker_id)
           ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/faculty1/anaconda3/lib/python3.13/site-packages/torch/utils/data/dataloader.py", line 1586, in _process_data
    data.reraise()
    ~~~~~~~~~~~~^^
  File "/home/faculty1/anaconda3/lib/python3.13/site-packages/torch/_utils.py", line 785, in reraise
    raise exception
RuntimeError: Caught RuntimeError in DataLoader worker process 1.
Original Traceback (most recent call last):
  File "/home/faculty1/anaconda3/lib/python3.13/site-packages/monai/transforms/transform.py", line 150, in apply_transform
    return _apply_transform(transform, data, unpack_items, lazy, overrides, log_stats)
  File "/home/faculty1/anaconda3/lib/python3.13/site-packages/monai/transforms/transform.py", line 98, in _apply_transform
    return transform(data, lazy=lazy) if isinstance(transform, LazyTrait) else transform(data)
           ~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/faculty1/anaconda3/lib/python3.13/site-packages/monai/transforms/croppad/dictionary.py", line 997, in __call__
    self.randomize(d.get(self.label_key), fg_indices, bg_indices, d.get(self.image_key))
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/faculty1/anaconda3/lib/python3.13/site-packages/monai/transforms/croppad/dictionary.py", line 979, in randomize
    self.cropper.randomize(label=label, fg_indices=fg_indices, bg_indices=bg_indices, image=image)
    ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/faculty1/anaconda3/lib/python3.13/site-packages/monai/transforms/croppad/array.py", line 1152, in randomize
    self.centers = generate_pos_neg_label_crop_centers(
                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        self.spatial_size,
        ^^^^^^^^^^^^^^^^^^
    ...<6 lines>...
        self.allow_smaller,
        ^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/home/faculty1/anaconda3/lib/python3.13/site-packages/monai/transforms/utils.py", line 690, in generate_pos_neg_label_crop_centers
    centers.append(correct_crop_centers(center, spatial_size, label_spatial_shape, allow_smaller))
                   ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/faculty1/anaconda3/lib/python3.13/site-packages/monai/transforms/utils.py", line 614, in correct_crop_centers
    raise ValueError(
    ...<2 lines>...
    )
ValueError: The size of the proposed random crop ROI is larger than the image size, got ROI size (128, 128, 128) and label image size (256, 256, 108) respectively.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/faculty1/anaconda3/lib/python3.13/site-packages/torch/utils/data/_utils/worker.py", line 358, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
  File "/home/faculty1/anaconda3/lib/python3.13/site-packages/torch/utils/data/_utils/fetch.py", line 54, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
            ~~~~~~~~~~~~^^^^^
  File "/home/faculty1/anaconda3/lib/python3.13/site-packages/monai/data/dataset.py", line 109, in __getitem__
    return self._transform(index)
           ~~~~~~~~~~~~~~~^^^^^^^
  File "/home/faculty1/anaconda3/lib/python3.13/site-packages/monai/data/dataset.py", line 95, in _transform
    return self.transform(data_i)
           ~~~~~~~~~~~~~~^^^^^^^^
  File "/home/faculty1/anaconda3/lib/python3.13/site-packages/monai/transforms/compose.py", line 346, in __call__
    result = execute_compose(
        input_,
    ...<8 lines>...
        log_stats=self.log_stats,
    )
  File "/home/faculty1/anaconda3/lib/python3.13/site-packages/monai/transforms/compose.py", line 116, in execute_compose
    data = apply_transform(
        _transform, data, map_items, unpack_items, lazy=lazy, overrides=overrides, log_stats=log_stats
    )
  File "/home/faculty1/anaconda3/lib/python3.13/site-packages/monai/transforms/transform.py", line 180, in apply_transform
    raise RuntimeError(f"applying transform {transform}") from e
RuntimeError: applying transform <monai.transforms.croppad.dictionary.RandCropByPosNegLabeld object at 0x775af05463c0>

(base) faculty1@labadmin3213:~/Brain_Tumor_Segmentation/experiments/baseline_t1ce_multiclass$ 

