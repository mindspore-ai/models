# Contents

- [Contents](#contents)
    - [CTSDG description](#ctsdg-description)
    - [Model architecture](#model-architecture)
    - [Dataset](#dataset)
    - [Environment requirements](#environment-requirements)
    - [Quick start](#quick-start)
    - [Script Description](#script-description)
        - [Script and Sample Code](#script-and-sample-code)
        - [Script Parameters](#script-parameters)
        - [Training Process](#training-process)
        - [Evaluation Process](#evaluation-process)
        - [Export MINDIR](#export-mindir)
    - [Model Description](#model-description)
        - [Training Performance on GPU](#training-performance-gpu)
    - [Description of Random Situation](#description-of-random-situation)
    - [ModelZoo Homepage](#modelzoo-homepage)

## [CTSDG description](#contents)

Deep generative approaches have recently made considerable progress in image inpainting by introducing
structure priors. Due to the lack of proper interaction with image texture during structure reconstruction, however,
current solutions are incompetent in handling the cases with large corruptions, and they generally suffer from distorted
results. This is a novel two-stream network for image inpainting, which models the structure constrained texture
synthesis and texture-guided structure reconstruction in a coupled manner so that they better leverage each other
for more plausible generation. Furthermore, to enhance the global consistency, a Bi-directional Gated Feature Fusion (Bi-GFF)
module is designed to exchange and combine the structure and texture information and a Contextual Feature Aggregation (CFA)
module is developed to refine the generated contents by region affinity learning and multiscale feature aggregation.

> [Paper](https://arxiv.org/pdf/2108.09760.pdf):  Image Inpainting via Conditional Texture and Structure Dual Generation
> Xiefan Guo, Hongyu Yang, Di Huang, 2021.
> [Supplementary materials](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Guo_Image_Inpainting_via_ICCV_2021_supplemental.pdf)

## [Model architecture](#contents)

## [Dataset](#contents)

Dataset used: [CELEBA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), [NVIDIA Irregular Mask Dataset](https://nv-adlr.github.io/publication/partialconv-inpainting)

- From **CELEBA** you need to download (section *Downloads -> Align&Cropped Images*):
    - `img_align_celeba.zip`
    - `list_eval_partitions.txt`
- From **NVIDIA Irregular Mask Dataset** you need to download:
    - `irregular_mask.zip`
    - `test_mask.zip`
- The directory structure is as follows:

  ```text
    .
    ├── img_align_celeba            # images folder
    ├── irregular_mask              # masks for training
    │   └── disocclusion_img_mask
    ├── mask                        # masks for testing
    │   └── testing_mask_dataset
    └── list_eval_partition.txt     # train/val/test splits
  ```

## [Environment requirements](#contents)

- Hardware（GPU）
    - Prepare hardware environment with GPU processor.
- Framework
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)
- Download dataset

## [Quick start](#contents)

### [Pretrained VGG16](#contents)

You need to convert torch VGG16 model for perceptual loss for train CTSDG model .

1. [Download pretrained VGG16](https://download.pytorch.org/models/vgg16-397923af.pth)
2. Convert torch checkpoint to mindspore:

```shell
python converter.py --torch_pretrained_vgg=/path/to/torch_pretrained_vgg
```

Converted mindspore checkpoint will be saved in the same directory as torch model with name `vgg16_feat_extr_ms.ckpt`.

After preparing the dataset and converting VGG16 you can start training and evaluation as follows：

### [Running on GPU](#contents)

#### Train

```shell
# standalone train
bash scripts/run_standalone_train_gpu.sh [DEVICE_ID] [CFG_PATH] [SAVE_PATH] [VGG_PRETRAIN] [IMAGES_PATH] [MASKS_PATH] [ANNO_PATH]

# distribute train
bash scripts/run_distribute_train_gpu.sh [DEVICE_NUM] [CFG_PATH] [SAVE_PATH] [VGG_PRETRAIN] [IMAGES_PATH] [MASKS_PATH] [ANNO_PATH]
```

Example:

```shell
# standalone train
# DEVICE_ID - device number for training
# CFG_PATH - path to config
# SAVE_PATH - path to save logs and checkpoints
# VGG_PRETRAIN - path to pretrained VGG16
# IMAGES_PATH - path to CELEBA dataset
# MASKS_PATH - path to masks for training
# ANNO_PATH - path to file with train/val/test splits
bash scripts/run_standalone_train_gpu.sh 0 ./default_config.yaml /path/to/output /path/to/vgg16_feat_extr.ckpt /path/to/img_align_celeba /path/to/training_mask /path/to/list_eval_partitions.txt

# distribute train (8p)
# DEVICE_NUM - number of devices for training
# other parameters as for standalone train
bash scripts/run_distribute_train_gpu.sh 8 ./default_config.yaml /path/to/output /path/to/vgg16_feat_extr.ckpt /path/to/img_align_celeba /path/to/training_mask /path/to/list_eval_partitions.txt
```

#### Evaluate

```shell
# evaluate
bash scripts/run_eval_gpu.sh [DEVICE_ID] [CFG_PATH] [CKPT_PATH] [IMAGES_PATH] [MASKS_PATH] [ANNO_PATH]
```

Example:

```shell
# evaluate
# DEVICE_ID - device number for evaluating
# CFG_PATH - path to config
# CKPT_PATH - path to ckpt for evaluation
# IMAGES_PATH - path to img_align_celeba dataset
# MASKS_PATH - path to masks for testing
# ANNO_PATH - path to file with train/val/test splits
bash scripts/run_eval_gpu.sh 0 ./default_config.yaml /path/to/ckpt /path/to/img_align_celeba /path/to/testing_mask /path/to/list_eval_partitions.txt  
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```text
.
└── CTSDG
    ├── model_utils
    │   ├── __init__.py                     # init file
    │   └── config.py                       # parse arguments
    ├── scripts
    │   ├── run_distribute_train_gpu.sh     # launch distributed training(8p) on GPU
    │   ├── run_eval_gpu.sh                 # launch evaluating on GPU
    │   ├── run_export_gpu.sh               # launch export mindspore model to mindir
    │   └── run_standalone_train_gpu.sh     # launch standalone traininng(1p) on GPU
    ├── src
    │   ├── discriminator
    │   │   ├── __init__.py                 # init file
    │   │   ├── discriminator.py            # discriminator
    │   │   └── spectral_conv.py            # conv2d with spectral normalization
    │   ├── generator
    │   │   ├── __init__.py                 # init file
    │   │   ├── bigff.py                    # bidirectional gated feature fusion
    │   │   ├── cfa.py                      # contextual feature aggregation
    │   │   ├── generator.py                # generator
    │   │   ├── pconv.py                    # partial convolution
    │   │   ├── projection.py               # feature to texture and texture to structure parts
    │   │   └── vgg16.py                    # VGG16 feature extractor
    │   ├── __init__.py                     # init file
    │   ├── callbacks.py                    # callbacks
    │   ├── dataset.py                      # celeba dataset
    │   ├── initializer.py                  # weight initializer
    │   ├── losses.py                       # model`s losses
    │   ├── trainer.py                      # trainer for ctsdg model
    │   └── utils.py                        # utils
    ├── __init__.py                         # init file
    ├── converter.py                        # convert VGG16 torch checkpoint to mindspore
    ├── default_config.yaml                 # config file
    ├── eval.py                             # evaluate mindspore model
    ├── export.py                           # export mindspore model to mindir format
    ├── README.md                           # readme file
    ├── requirements.txt                    # requirements
    └── train.py                            # train mindspore model
```

### [Script Parameters](#contents)

Training parameters can be configured in `default_config.yaml`

```text
"gen_lr_train": 0.0002,                     # learning rate for generator training stage
"gen_lr_finetune": 0.00005,                 # learning rate for generator finetune stage
"dis_lr_multiplier": 0.1,                   # discriminator`s lr is generator`s lr multiply by this parameter
"batch_size": 6,                            # batch size
"train_iter": 350000,                       # number of training iterations
"finetune_iter": 150000                     # number of finetune iterations
"image_load_size": [256, 256]               # input image size
```

For more parameters refer to the contents of `default_config.yaml`.

### [Training Process](#contents)

#### [Run on GPU](#contents)

##### Standalone training (1p)

```shell
# DEVICE_ID - device number for training (0)
# CFG_PATH - path to config (./default_config.yaml)
# SAVE_PATH - path to save logs and checkpoints (/path/to/output)
# VGG_PRETRAIN - path to pretrained VGG16 (/path/to/vgg16_feat_extr.ckpt)
# IMAGES_PATH - path to CELEBA dataset (/path/to/img_align_celeba)
# MASKS_PATH - path to masks for training (/path/to/training_mask)
# ANNO_PATH - path to file with train/val/test splits (/path/to/list_eval_partitions.txt)
bash scripts/run_standalone_train_gpu.sh 0 ./default_config.yaml /path/to/output /path/to/vgg16_feat_extr.ckpt /path/to/img_align_celeba /path/to/training_mask /path/to/list_eval_partitions.txt
```

Logs will be saved to `/path/to/output/log.txt`

Result:

```text
...
DATE TIME iter: 250, loss_g: 19.7810001373291, loss_d: 1.7710000276565552, step time: 570.67 ms
DATE TIME iter: 375, loss_g: 20.549999237060547, loss_d: 1.8650000095367432, step time: 572.09 ms
DATE TIME iter: 500, loss_g: 25.295000076293945, loss_d: 1.8630000352859497, step time: 572.23 ms
DATE TIME iter: 625, loss_g: 24.059999465942383, loss_d: 1.812999963760376, step time: 573.33 ms
DATE TIME iter: 750, loss_g: 26.343000411987305, loss_d: 1.8539999723434448, step time: 573.18 ms
DATE TIME iter: 875, loss_g: 21.774999618530273, loss_d: 1.8509999513626099, step time: 573.0 ms
DATE TIME iter: 1000, loss_g: 18.062999725341797, loss_d: 1.7960000038146973, step time: 572.41 ms
...
```

##### Distribute training (8p)

```shell
# DEVICE_NUM - number of devices for training (8)
# other parameters as for standalone train
bash scripts/run_distribute_train_gpu.sh 8 ./default_config.yaml /path/to/output /path/to/vgg16_feat_extr.ckpt /path/to/img_align_celeba /path/to/training_mask /path/to/list_eval_partitions.txt
```

Logs will be saved to `/path/to/output/log.txt`

Result:

```text
...
DATE TIME iter: 250, loss_g: 26.28499984741211, loss_d: 1.680999994277954, step time: 757.67 ms
DATE TIME iter: 375, loss_g: 21.548999786376953, loss_d: 1.468000054359436, step time: 758.02 ms
DATE TIME iter: 500, loss_g: 17.89299964904785, loss_d: 1.2829999923706055, step time: 758.57 ms
DATE TIME iter: 625, loss_g: 18.750999450683594, loss_d: 1.2589999437332153, step time: 759.95 ms
DATE TIME iter: 750, loss_g: 21.542999267578125, loss_d: 1.1829999685287476, step time: 759.45 ms
DATE TIME iter: 875, loss_g: 27.972000122070312, loss_d: 1.1629999876022339, step time: 759.62 ms
DATE TIME iter: 1000, loss_g: 18.03499984741211, loss_d: 1.159000039100647, step time: 759.51 ms
...
```

### [Evaluation Process](#contents)

#### GPU

```shell
bash scripts/run_eval_gpu.sh [DEVICE_ID] [CFG_PATH] [CKPT_PATH] [IMAGES_PATH] [MASKS_PATH] [ANNO_PATH]
```

Example:

```shell
# DEVICE_ID - device number for evaluating (0)
# CFG_PATH - path to config (./default_config.yaml)
# CKPT_PATH - path to ckpt for evaluation (/path/to/ckpt)
# IMAGES_PATH - path to img_align_celeba dataset (/path/to/img_align_celeba)
# MASKS_PATH - path to masks for testing (/path/to/testing/mask)
# ANNO_PATH - path to file with train/val/test splits (/path/to/list_eval_partitions.txt)
bash scripts/run_eval_gpu.sh 0 ./default_config.yaml /path/to/ckpt /path/to/img_align_celeba /path/to/testing_mask /path/to/list_eval_partitions.txt
```

Logs will be saved to `./logs/eval_log.txt`.

Result:

```text
PSNR:
0-20%: 38.04
20-40%: 29.39
40-60%: 24.21
SSIM:
0-20%: 0.979
20-40%: 0.922
40-60%: 0.83
```

### [Export MINDIR](#contents)

If you want to infer the network on Ascend 310, you should convert the model to MINDIR.

#### GPU

```shell
bash scripts/run_export_gpu.sh [DEVICE_ID] [CFG_PATH] [CKPT_PATH]
```

Example:

```shell
# DEVICE_ID - device number (0)
# CFG_PATH - path to config (./default_config.yaml)
# CKPT_PATH - path to ckpt for evaluation (/path/to/ckpt)
bash scripts/run_export_gpu.sh 0 ./default_config.yaml /path/to/ckpt
```

Logs will be saved to `./logs/export_log.txt`, converted model will have the same name as ckpt except extension.

## [Model Description](#contents)

### [Training Performance on GPU](#contents)

| Parameter           | CTSDG (1p)                                                                                                                                                                                                   | CTSDG (8p)                                                                                                                                                                                                   |
|---------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Resource            | 1x Nvidia RTX 3090                                                                                                                                                                                           | 8x Nvidia RTX 3090                                                                                                                                                                                           |
| Uploaded date       | 03.06.2022                                                                                                                                                                                                   | 03.06.2022                                                                                                                                                                                                   |
| Mindspore version   | 1.6.1                                                                                                                                                                                                        | 1.6.1                                                                                                                                                                                                        |
| Dataset             | CELEBA, NVIDIA Irregular Mask Dataset                                                                                                                                                                        | CELEBA, NVIDIA Irregular Mask Dataset                                                                                                                                                                        |
| Training parameters | train_iter=350000, finetune_iter=150000, gen_lr_train=0.0002, gen_lr_finetune=0.00005, dis_lr_multiplier=0.1, batch_size=6                                                                                   | train_iter=43750, finetune_iter=18750, gen_lr_train=0.002, gen_lr_finetune=0.0005, dis_lr_multiplier=0.1, batch_size=6                                                                                       |
| Optimizer           | Adam                                                                                                                                                                                                         | Adam                                                                                                                                                                                                         |
| Loss function       | Reconstruction Loss (L1), Perceptual Loss (L1), Style Loss(L1), Adversarial Loss (BCE), Intermediate Loss (L1 + BCE)                                                                                         | Reconstruction Loss (L1), Perceptual Loss (L1), Style Loss(L1), Adversarial Loss (BCE), Intermediate Loss (L1 + BCE)                                                                                         |
| Speed               | 573 ms / step                                                                                                                                                                                                | 759 ms / step                                                                                                                                                                                                |
| Metrics             | <table><tr><td></td><td>0-20%</td><td>20-40%</td><td>40-60%</td></tr><tr><td>PSNR</td><td>38.04</td><td>29.39</td><td>24.21</td></tr><tr><td>SSIM</td><td>0.979</td><td>0.922</td><td>0.83</td></tr></table> | <table><tr><td></td><td>0-20%</td><td>20-40%</td><td>40-60%</td></tr><tr><td>PSNR</td><td>37.74</td><td>29.17</td><td>24.01</td></tr><tr><td>SSIM</td><td>0.978</td><td>0.92</td><td>0.826</td></tr></table> |

## [Description of Random Situation](#contents)

`train.py` script use mindspore.set_seed() to set global random seed, which can be modified.  

## [ModelZoo Homepage](#contents)

Please visit the official website [homepage](https://gitee.com/mindspore/models).
