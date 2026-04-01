Verifying one training batch...
/home/faculty1/Brain_Tumor_Segmentation/experiments/baseline_t1ce_multiclass/train_B.py:159: DeprecationWarning: __array__ implementation doesn't accept a copy keyword, so passing copy=False failed. __array__ must implement 'dtype' and 'copy' keyword arguments. To learn more, see the migration guide https://numpy.org/devdocs/numpy_2_0_migration_guide.html#adapting-to-changes-in-the-copy-keyword
  remapped = np.array(label, copy=True)
/home/faculty1/Brain_Tumor_Segmentation/experiments/baseline_t1ce_multiclass/train_B.py:159: DeprecationWarning: __array__ implementation doesn't accept a copy keyword, so passing copy=False failed. __array__ must implement 'dtype' and 'copy' keyword arguments. To learn more, see the migration guide https://numpy.org/devdocs/numpy_2_0_migration_guide.html#adapting-to-changes-in-the-copy-keyword
  remapped = np.array(label, copy=True)
image shape: (2, 1, 128, 128, 128)
label shape: (2, 128, 128, 128)
label unique values: [0, 1, 2, 3]
Traceback (most recent call last):
  File "/home/faculty1/Brain_Tumor_Segmentation/experiments/baseline_t1ce_multiclass/train_B.py", line 809, in <module>
    main()
    ~~~~^^
  File "/home/faculty1/Brain_Tumor_Segmentation/experiments/baseline_t1ce_multiclass/train_B.py", line 772, in main
    sanity_loss_fn = build_loss_function(args.loss_type, parsed_class_weights, num_classes, cfg, device)
  File "/home/faculty1/Brain_Tumor_Segmentation/experiments/baseline_t1ce_multiclass/train_B.py", line 411, in build_loss_function
    return DiceCELoss(
        to_onehot_y=True,
    ...<3 lines>...
        lambda_dice=lambda_dice,
    )
TypeError: DiceCELoss.__init__() got an unexpected keyword argument 'ce_weight'. Did you mean 'weight'?
(base) faculty1@labadmin3213:~/Brain_Tumor_Segmentation/experiments/baseline_t1ce_multiclass$ 

