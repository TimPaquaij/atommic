# **Reproducing the ATOMMIC paper**

The ATOMMIC paper is available at https://arxiv.org/abs/. In this document, we provide the instructions for reproducing
the results of the paper.

**Note:** You would need to download and preprocess the following datasets to reproduce the results. Please refer to each project folder for more information:
- [Stanford Knee MRI Multi-Task Evaluation (SKM-TEA) 2021 Dataset](../MTL/rs/SKMTEA/README.md).
- [Amsterdam Ultra-high field adult lifespan database (AHEAD)](../qMRI/AHEAD/README.md).
  - In the ATOMMIC paper, we used the first 10 subjects of the AHEAD dataset, as listed on the download page. The first 6 were used for training, the next 2 for validation, and the last 2 for testing.
- [Calgary-Campinas Public Brain MR Dataset (CC359)](../REC/CC359/README.md).
- [fastMRI Brains Multicoil Dataset](../REC/fastMRIBrainsMulticoil/README.md).
- [BraTS2023AdultGlioma](../SEG/BraTS2023AdultGlioma/README.md).
- [ISLES2022SubAcuteStroke](../SEG/ISLES2022SubAcuteStroke/README.md).
- [Stanford Knee MRI Multi-Task Evaluation (SKM-TEA) 2021 Segmentation Dataset](../SEG/SKMTEA/README.md).

## **It is highly recommended to create a safe copy of the whole folder before running the scripts. Just in case setting and renaming paths goes wrong to be easy to revert.**


## **Set the data paths**
Next you need to set the data paths by running:
```bash
.projects/ATOMMIC_paper/set_paths.sh
```

Paths should be set for example ```/data/``` if the AHEAD dataset is located at ```/data/AHEAD/``` and not ```/data/AHEAD/```.

## **Reproducing the results**
To reproduce the results, first run the following script to perform inference with the pre-trained models:
```bash
.projects/ATOMMIC_paper/run_models.sh
```

**Note:** Just before you run lines 13 and 14 on the ```run_models.sh``` script, specifically the following lines:
```bash
atommic run -c projects/ATOMMIC_paper/qMRI/AHEAD/conf/quantitative_test/qcirim.yaml
atommic run -c projects/ATOMMIC_paper/qMRI/AHEAD/conf/quantitative_test/qvarnet.yaml
```
you will need to set the ```initial_predictions_path``` to the output path of lines 7 and 10, respectively. Specifically, you will need to set the ```initial_predictions_path``` to the output path of the following lines:
```bash
atommic run -c projects/ATOMMIC_paper/qMRI/AHEAD/conf/reconstruction_test/cirim_test_set.yaml
atommic run -c projects/ATOMMIC_paper/qMRI/AHEAD/conf/reconstruction_test/varnet_test_set.yaml
```

**You do not need to set any checkpoint paths as the script will automatically download the pre-trained models from HuggingFace.**

Next, run the following script to evaluate the results:
```bash
.projects/ATOMMIC_paper/evaluate.sh
```

## **Results**

## Overview of Performance on the REC Task

| Model      | CC359 - Poisson 2D 5x SSIM | CC359 - Poisson 2D 5x PSNR | CC359 - Poisson 2D 10x SSIM | CC359 - Poisson 2D 10x PSNR | fastMRIBrains - Equispaced 1D 4x SSIM | fastMRIBrains - Equispaced 1D 4x PSNR | fastMRIBrains - Equispaced 1D 8x SSIM | fastMRIBrains - Equispaced 1D 8x | StanfordKnee - Gaussian 2D 12x SSIM | StanfordKnee - Gaussian 2D 12x PSNR |
|------------|----------------------------|----------------------------|-----------------------------|-----------------------------|---------------------------------------|---------------------------------------|---------------------------------------|----------------------------------|-------------------------------------|-------------------------------------|
| CCNN       | 0.845 +/- 0.064            | 28.36 +/- 3.69             | 0.783 +/- 0.089             | 25.95 +/- 3.64              | 0.886 +/- 0.192                       | 33.47 +/- 5.92                        | 0.836 +/- 0.202                       | 29.40 +/- 5.71                   | 0.767 +/- 0.298                     | 31.61 +/- 6.84                      |
| CIRIM      | 0.858 +/- 0.074            | 28.79 +/- 4.23             | 0.816 +/- 0.094             | 26.92 +/- 4.36              | 0.892 +/- 0.184                       | 33.83 +/- 6.11                        | 0.846 +/- 0.202                       | 30.23 +/- 5.66                   | 0.796 +/- 0.311                     | 32.77 +/- 7.23                      |
| CRNN       | 0.774 +/- 0.088            | 25.59 +/- 4.19             | 0.722 +/- 0.088             | 24.48 +/- 3.39              | 0.868 +/- 0.195                       | 31.31 +/- 5.46                        | 0.806 +/- 0.198                       | 27.50 +/- 5.57                   |                                     |                                     |
| JointICNet | 0.872 +/- 0.065            | 29.28 +/- 3.99             | 0.828 +/- 0.086             | 27.36 +/- 4.10              | 0.832 +/- 0.198                       | 28.57 +/- 5.50                        | 0.772 +/- 0.202                       | 25.50 +/- 5.38                   | 0.727 +/- 0.291                     | 29.52 +/- 6.33                      |
| KIKINet    | 0.788 +/- 0.087            | 25.43 +/- 4.16             | 0.742 +/- 0.105             | 24.37 +/- 3.88              | 0.856 +/- 0.201                       | 31.02 +/- 5.68                        | 0.840 +/- 0.208                       | 29.51 +/- 5.93                   | 0.659 +/- 0.241                     | 27.33 +/- 5.55                      |
| LPDNet     | 0.849 +/- 0.075            | 28.26 +/- 4.22             | 0.810 +/- 0.099             | 26.73 +/- 4.23              | 0.870 +/- 0.188                       | 31.44 +/- 5.66                        | 0.805 +/- 0.207                       | 27.78 +/- 5.82                   | 0.737 +/- 0.297                     | 29.79 +/- 6.28                      |
| MoDL       | 0.844 +/- 0.068            | 27.97 +/- 4.20             | 0.793 +/- 0.088             | 25.89 +/- 4.39              | 0.882 +/- 0.201                       | 32.60 +/- 6.78                        | 0.813 +/- 0.192                       | 27.81 +/- 5.86                   | 0.566 +/- 0.283                     | 23.63 +/- 4.64                      |
| RIM        | 0.834 +/- 0.077            | 27.45 +/- 4.32             | 0.788 +/- 0.091             | 25.56 +/- 3.96              | 0.885 +/- 0.190                       | 33.24 +/- 6.15                        | 0.838 +/- 0.199                       | 29.45 +/- 5.58                   | 0.769 +/- 0.304                     | 31.53 +/- 6.79                      |
| RVN        | 0.845 +/- 0.067            | 28.14 +/- 3.53             | 0.787 +/- 0.093             | 26.03 +/- 3.77              | 0.894 +/- 0.180                       | 34.23 +/- 5.97                        | 0.843 +/- 0.195                       | 30.08 +/- 5.68                   | 0.778 +/- 0.300                     | 31.96 +/- 6.90                      |
| UNet       | 0.849 +/- 0.070            | 28.85 +/- 4.17             | 0.810 +/- 0.091             | 27.20 +/- 4.20              | 0.885 +/- 0.182                       | 33.09 +/- 6.02                        | 0.856 +/- 0.216                       | 30.73 +/- 5.94                   | 0.770 +/- 0.295                     | 31.40 +/- 6.55                      |
| VarNet     | 0.874 +/- 0.061            | 29.49 +/- 3.86             | 0.827 +/- 0.087             | 27.51 +/- 4.01              | 0.892 +/- 0.198                       | 34.00 +/- 6.30                        | 0.847 +/- 0.197                       | 29.87 +/- 5.68                   | 0.764 +/- 0.302                     | 31.50 +/- 6.70                      |
| VSNet      | 0.788 +/- 0.079            | 25.51 +/- 3.91             | 0.740 +/- 0.089             | 24.19 +/- 3.27              | 0.856 +/- 0.196                       | 30.37 +/- 5.34                        | 0.796 +/- 0.197                       | 26.88 +/- 5.43                   | 0.708 +/- 0.289                     | 28.51 +/- 5.79                      |
| XPDNet     | 0.761 +/- 0.100            | 24.27 +/- 4.14             | 0.700 +/- 0.112             | 22.65 +/- 3.22              | 0.854 +/- 0.212                       | 31.03 +/- 6.75                        | 0.788 +/- 0.218                       | 26.96 +/- 6.18                   | 0.654 +/- 0.270                     | 27.18 +/- 5.77                      |
| ZeroFilled | 0.679 +/- 0.103            | 19.89 +/- 7.45             | 0.656 +/- 0.092             | 19.24 +/- 7.37              | 0.671 +/- 0.194                       | 24.12 +/- 6.21                        | 0.591 +/- 0.213                       | 21.03 +/- 5.97                   | 0.549 +/- 0.197                     | 18.11 +/- 6.23                      |


## Overview of Performance on the qMRI Task - AHEAD Dataset (Gaussian 2D 12x Undersampling)

| Model   | REC SSIM        | REC PSNR       | REC NMSE        | qMRI SSIM       | qMRI PSNR       | qMRI NMSE       |
|---------|-----------------|----------------|-----------------|-----------------|-----------------|-----------------|
| CIRIM   | 0.909 +/- 0.083 | 32.89 +/- 8.60 | 0.044 +/- 0.075 |                 |
| VarNet  | 0.894 +/- 0.053 | 32.39 +/- 4.80 | 0.047 +/- 0.062 |                 |
| qCIRIM  |                 |                |                 | 0.881 +/- 0.177 | 28.28 +/- 11.31 | 0.124 +/- 0.338 |
| qVarNet |                 |                |                 | 0.784 +/- 0.206 | 24.36 +/- 7.79  | 0.192 +/- 0.334 |


## Overview of Performance on the SEG Task

### BraTS2023AdultGlioma

| Model         | DICE            | F1              | HD95            | IOU             |
|---------------|-----------------|-----------------|-----------------|-----------------|
| AttentionUNet | 0.930 +/- 0.126 | 0.648 +/- 0.763 | 3.836 +/- 3.010 | 0.537 +/- 0.662 |
| DynUNet       | 0.806 +/- 0.276 | 0.104 +/- 0.580 | 5.119 +/- 5.411 | 0.070 +/- 0.419 |
| UNet          | 0.937 +/- 0.118 | 0.671 +/- 0.787 | 3.504 +/- 2.089 | 0.535 +/- 0.663 |
| UNet3D        | 0.936 +/- 0.133 | 0.674 +/- 0.782 | 3.550 +/- 2.162 | 0.528 +/- 0.652 |
| VNet          | 0.733 +/- 0.437 | 0.014 +/- 0.234 | 6.010 +/- 6.097 | 0.000 +/- 0.004 |

### SKMTEA segmentation only

| Model         | DICE            | F1              | HD95            | IOU             |
|---------------|-----------------|-----------------|-----------------|-----------------|
| AttentionUNet | 0.909 +/- 0.088 | 0.637 +/- 0.475 | 6.358 +/- 2.209 | 0.529 +/- 0.361 |
| DynUNet       | 0.689 +/- 0.136 | 0.059 +/- 0.264 | 8.973 +/- 4.507 | 0.015 +/- 0.066 |
| UNet          | 0.912 +/- 0.058 | 0.651 +/- 0.449 | 6.618 +/- 1.793 | 0.516 +/- 0.350 |
| UNet3D        | 0.918 +/- 0.068 | 0.789 +/- 0.404 | 5.893 +/- 2.995 | 0.530 +/- 0.347 |
| VNet          | 0.918 +/- 0.081 | 0.816 +/- 0.426 | 5.540 +/- 3.036 | 0.507 +/- 0.388 |

### ISLES2022SubAcuteStroke

|               | ALD             | ALD             | DICE            | L-F1            |
|---------------|-----------------|-----------------|-----------------|-----------------|
| AttentionUNet | 0.809 +/- 2.407 | 0.548 +/- 3.411 | 0.709 +/- 0.552 | 0.799 +/- 0.579 |
| DynUNet       | 0.752 +/- 2.230 | 0.586 +/- 3.874 | 0.729 +/- 0.529 | 0.802 +/- 0.564 |
| UNet          | 0.909 +/- 3.953 | 0.544 +/- 3.921 | 0.695 +/- 0.559 | 0.786 +/- 0.585 |
| UNet3D        | 0.821 +/- 2.167 | 0.691 +/- 5.458 | 0.687 +/- 0.547 | 0.798 +/- 0.573 |
| VNet          | 2.281 +/- 10.72 | 3.257 +/- 27.43 | 0.490 +/- 0.694 | 0.600 +/- 0.687 |


## Overview of Performance on the MTL Task - SKMTEA Dataset (Poisson 2D 4x Undersampling)

| Model     | SSIM            | PSNR           | DICE            | F1              | HD95            | IOU             |
|-----------|-----------------|----------------|-----------------|-----------------|-----------------|-----------------|
| IDSLR     | 0.836 +/- 0.106 | 30.38 +/- 5.67 | 0.894 +/- 0.127 | 0.256 +/- 0.221 | 4.927 +/- 2.812 | 0.298 +/- 0.309 |
| IDSLRUNET | 0.842 +/- 0.106 | 30.53 +/- 5.59 | 0.870 +/- 0.134 | 0.225 +/- 0.194 | 8.724 +/- 3.298 | 0.212 +/- 0.199 |
| MTLRS     | 0.832 +/- 0.106 | 30.48 +/- 5.30 | 0.889 +/- 0.118 | 0.247 +/- 0.203 | 7.594 +/- 3.673 | 0.218 +/- 0.194 |
| SegNet    | 0.840 +/- 0.107 | 29.95 +/- 5.12 | 0.915 +/- 0.114 | 0.270 +/- 0.284 | 3.002 +/- 1.449 | 0.290 +/- 0.349 |
