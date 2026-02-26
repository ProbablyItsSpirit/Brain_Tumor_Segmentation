# BraTS 2024 Dataset Collection
## Overview
- **Total Patients**: 2,728
- **Storage**: 90GB (GLI: 44.8GB, MEN: 12GB, PED: 32.7GB)

## Dataset Breakdown

### BraTS-GLI (Glioma)
- **Source**: Synapse (Dec 2024)
- **Training**: 1,621 cases (BraTS-GLI-00005-100 to BraTS-GLI-03064-100)
- **Validation**: 188 cases (BraTS-GLI-02073-100 to BraTS-GLI-03058-100)
- **Modalities**: T1, T1CE, T2, FLAIR, SEG
- **Metadata**: BraTS-PTG supplementary demographic information and metadata.xlsx
- **Citation**: Included in CITATIONS.bib

### BraTS-MEN-RT (Meningioma with Radiotherapy)
- **Source**: Synapse (Feb 2025)
- **Training**: 500 cases (BraTS-MEN-RT-0002-1 to BraTS-MEN-RT-0625-1)
- **Training Additional**: 1 case (BraTS-MEN-RT-0402-1)
- **Validation**: 70 cases (BraTS-MEN-RT-0010-1 to BraTS-MEN-RT-0701-1)
- **Modalities**: T1CE only? (verify others), GTV segmentation
- **Metadata**: Meningioma radiotherapy supplementary clinical data.xlsx
- **Citation**: Included in CITATION.bib

### BraTS-PED (Pediatric)
- **Source**: Cancer Imaging Archive (Dec 2024)
- **Training**: 257 cases (BraTS-PED-00001-000 to BraTS-PED-00266-000)
- **Validation**: 91 cases (BraTS-PED-00267-000 to BraTS-PED-00357-000)
- **Modalities**: T1, T1CE, T2, FLAIR, SEG
- **Metadata**: view_brats.py (visualization script)


## Dataset Statistics

| **Dataset** | **Split** | **Total Patients** | **With Tumor** | **Tumor %** | **No Tumor** | **Avg Tumor Vol** | **Median Tumor Vol** | **Min Tumor Vol** | **Max Tumor Vol** | **Small %** | **Medium %** | **Large %** | **ET %** | **ED %** | **NCR %** |
|-------------|-----------|-------------------:|---------------:|------------:|-------------:|------------------:|---------------------:|------------------:|------------------:|------------:|-------------:|------------:|---------:|---------:|----------:|
| GLI         | train     | 1,621              | 1,621          | 100.0%      | 0            | 73,266            | 59,234               | 197               | 345,565           | 4.4%        | 70.8%        | 24.7%       | 75.4%    | 99.8%    | 95.7%     |
| MEN         | train     | 500                | 500            | 100.0%      | 0            | 26,956            | 11,743               | 94                | 567,439           | 45.8%       | 49.2%        | 5.0%        | —        | —        | —         |
| MEN         | train_extra | 1                | 1              | 100.0%      | 0            | 45,738            | 45,738               | 45,738            | 45,738            | 0.0%        | 100.0%       | 0.0%        | —        | —        | —         |
| PED         | train     | 257                | 257            | 100.0%      | 0            | 53,506            | 36,969               | 854               | 270,472           | 5.8%        | 79.0%        | 15.2%       | 32.3%    | 99.6%    | 69.3%     |



## File Structure

```
Datasets/
├── BraTS-GLI/
│   ├── train/                    # 1,621 cases
│   ├── val/                      # 188 cases
│   ├── *.bib
│   └── *.xlsx
├── BraTS-MEN-RT/
│   ├── train/                    # 500 cases
│   ├── train_additional/         # 1 case
│   ├── val/                      # 70 cases
│   ├── *.bib
│   └── *.xlsx
└── BraTS-PED/
    ├── train/                    # 257 cases
    ├── val/                      # 91 cases
    ├── view_brats.py
    └── brats_visualization.png
```


## Tumor Volume Distribution

### GLI (Glioma) - Largest tumors
```
Average: 73,266 voxels (MEDIUM-LARGE)

Range: 197 to 345,565 voxels (huge variety!)

Distribution:

Small (<10k): 4.4% (71 patients)

Medium (10k-100k): 70.8% (1,148 patients)

Large (>100k): 24.7% (400 patients)
```

### PED (Pediatric) - Medium tumors
```
Average: 53,506 voxels (MEDIUM)

Range: 854 to 270,472 voxels

Distribution:

Small: 5.8% (15 patients)

Medium: 79.0% (203 patients)

Large: 15.2% (39 patients)
```
### MEN (Meningioma) - Smallest tumors
```
Average: 26,956 voxels (SMALL-MEDIUM)

Range: 94 to 567,439 voxels (interesting - some huge ones!)

Distribution:

Small: 45.8% (229 patients)

Medium: 49.2% (246 patients)

Large: 5.0% (25 patients)
```