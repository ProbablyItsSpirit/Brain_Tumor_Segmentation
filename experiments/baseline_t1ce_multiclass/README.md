(base) faculty1@labadmin3213:~/Brain_Tumor_Segmentation/experiments/baseline_t1ce_multiclass$ python inference.py --config config.yaml --checkpoint checkpoints/baseline_t1ce_multiclass/stage_b_best.pt
GLI train: 1621 cases loaded (missing/skipped: 0)
PED train: 257 cases loaded (missing/skipped: 0)
MEN train: 500 cases loaded (missing/skipped: 0)

Total test samples for inference: 757
/home/faculty1/Brain_Tumor_Segmentation/experiments/baseline_t1ce_multiclass/inference.py:73: DeprecationWarning: __array__ implementation doesn't accept a copy keyword, so passing copy=False failed. __array__ must implement 'dtype' and 'copy' keyword arguments. To learn more, see the migration guide https://numpy.org/devdocs/numpy_2_0_migration_guide.html#adapting-to-changes-in-the-copy-keyword
  remapped = np.array(label, copy=True)
/home/faculty1/Brain_Tumor_Segmentation/experiments/baseline_t1ce_multiclass/inference.py:73: DeprecationWarning: __array__ implementation doesn't accept a copy keyword, so passing copy=False failed. __array__ must implement 'dtype' and 'copy' keyword arguments. To learn more, see the migration guide https://numpy.org/devdocs/numpy_2_0_migration_guide.html#adapting-to-changes-in-the-copy-keyword
  remapped = np.array(label, copy=True)
Traceback (most recent call last):
  File "/home/faculty1/Brain_Tumor_Segmentation/experiments/baseline_t1ce_multiclass/inference.py", line 357, in <module>
    main()
    ~~~~^^
  File "/home/faculty1/Brain_Tumor_Segmentation/experiments/baseline_t1ce_multiclass/inference.py", line 330, in main
    logits = model(images)
  File "/home/faculty1/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/faculty1/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1790, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/faculty1/anaconda3/lib/python3.13/site-packages/monai/networks/nets/unet.py", line 297, in forward
    x = self.model(x)
  File "/home/faculty1/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/faculty1/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1790, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/faculty1/anaconda3/lib/python3.13/site-packages/torch/nn/modules/container.py", line 253, in forward
    input = module(input)
  File "/home/faculty1/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/faculty1/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1790, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/faculty1/anaconda3/lib/python3.13/site-packages/monai/networks/layers/simplelayers.py", line 128, in forward
    y = self.submodule(x)
  File "/home/faculty1/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/faculty1/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1790, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/faculty1/anaconda3/lib/python3.13/site-packages/torch/nn/modules/container.py", line 253, in forward
    input = module(input)
  File "/home/faculty1/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/faculty1/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1790, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/faculty1/anaconda3/lib/python3.13/site-packages/monai/networks/layers/simplelayers.py", line 131, in forward
    return torch.cat([x, y], dim=self.dim)
           ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^
  File "/home/faculty1/anaconda3/lib/python3.13/site-packages/monai/data/meta_tensor.py", line 283, in __torch_function__
    ret = super().__torch_function__(func, types, args, kwargs)
  File "/home/faculty1/anaconda3/lib/python3.13/site-packages/torch/_tensor.py", line 1703, in __torch_function__
    ret = func(*args, **kwargs)
RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 39 but got size 40 for tensor number 1 in the list.
(base) faculty1@labadmin3213:~/Brain_Tumor_Segmentation/experiments/baseline_t1ce_multiclass$ 

