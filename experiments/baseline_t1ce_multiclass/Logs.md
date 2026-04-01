Traceback (most recent call last):
  File "/home/faculty1/Brain_Tumor_Segmentation/experiments/baseline_t1ce_multiclass/small_file.py", line 161, in <module>
    main()
    ~~~~^^
  File "/home/faculty1/Brain_Tumor_Segmentation/experiments/baseline_t1ce_multiclass/small_file.py", line 89, in main
    gli_test_files = select_split_files(
        all_dataset_dicts,
        split_datasets={"GLI": "test"},
        split_name="test",
    )
  File "/home/faculty1/Brain_Tumor_Segmentation/experiments/baseline_t1ce_multiclass/inference.py", line 217, in select_split_files
    raise ValueError(
    	f"Unknown source split '{source_split}' for dataset '{dataset_name}'"
    )
ValueError: Unknown source split 'test' for dataset 'GLI'

