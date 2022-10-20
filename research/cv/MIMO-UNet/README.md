# Contents

- [Contents](#contents)
- [MIMO-UNet Description](#mimo-unet-description)
- [Model-architecture](#model-architecture)
- [Dataset](#dataset)
- [Environmental requirements](#environmental-requirements)
- [Quickstart](#quickstart)
- [Script description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Training process](#training-process)
        - [Standalone training](#training)
        - [Distributed training](#distributed-training)
    - [Evaluation process](#evaluation-process)
        - [Evaluate](#evaluate)
    - [Inference process](#inference-process)
        - [Export MindIR](#export-mindir)
- [Model description](#model-description)
    - [Performance](#performance)
        - [Training performance](#training-performance)
        - [Evaluation performance](#evaluation-performance)
- [Description of Random Situation](#contents)
- [ModelZoo homepage](#modelzoo-homepage)

# [MIMO-UNet Description](#contents)

Coarse-to-fine strategies have been extensively used for the architecture design of single image deblurring networks.
Conventional methods typically stack sub-networks with multi-scale input images and gradually improve sharpness of
images from the bottom sub-network to the top sub-network, yielding inevitably high computational costs. Toward a fast
and accurate deblurring network design, we revisit the coarse-to-fine strategy and present a multi-input multi-output
U-net (MIMO-UNet). First, the single encoder of the MIMO-UNet takes multi-scale input images to ease the difficulty
of training. Second, the single decoder of the MIMO-UNet outputs multiple deblurred images with different scales to
mimic multi-cascaded U-nets using a single U-shaped network. Last, asymmetric feature fusion is introduced to merge
multi-scale features in an efficient manner. Extensive experiments on the GoPro and RealBlur datasets demonstrate that
the proposed network outperforms the state-of-the-art methods in terms of both accuracy and computational complexity.

[Paper](https://arxiv.org/abs/2108.05054): Rethinking Coarse-to-Fine Approach in Single Image Deblurring.

[Reference github repository](https://github.com/chosj95/MIMO-UNet)

# [Model architecture](#contents)

The architecture of MIMO-UNet is based on a single U-Net with significant modifications for efficient multi-scale
deblurring. The encoder and decoder of MIMO-UNet are composed of three encoder blocks (EBs) and decoder blocks (DBs)
that use convolutional layers to extract features from different stages.

# [Dataset](#contents)

## Dataset used

Dataset link (Google Drive): [GOPRO_Large](https://drive.google.com/file/d/1y4wvPdOG3mojpFCHTqLgriexhbjoWVkK/view?usp=sharing)

GOPRO_Large dataset is proposed for dynamic scene deblurring. Training and Test set are publicly available.

- Dataset size: ~6.2G
    - Train: 3.9G, 2103 image pairs
    - Test: 2.3G, 1111 image pairs
    - Data format: Images
    - Note: Data will be processed in src/data_augment.py and src/data_load.py

## Dataset organize way

```text
.
└─ GOPRO_Large
  ├─ train
  │  ├─ GOPR0xxx_xx_xx
  │  │  ├─ blur
  │  │  │  ├─ ***.png
  │  │  │  └─ ...
  │  │  ├─ blur_gamma
  │  │  │  ├─ ***.png
  │  │  │  └─ ...
  │  │  ├─ sharp
  │  │  │  ├─ ***.png
  │  │  │  └─ ...
  │  │  └─ frames X offset X.txt
  │  └─ ...
  └─ test
     ├─ GOPR0xxx_xx_xx
     │  ├─ blur
     │  │  ├─ ***.png
     │  │  └─ ...
     │  ├─ blur_gamma
     │  │  ├─ ***.png
     │  │  └─ ...
     │  └─ sharp
     │     ├─ ***.png
     │     └─ ...
     └─ ...
```

## Dataset preprocessing

After downloading the dataset, run the `preprocessing.py` script located in the folder `src`.
Below is the file structure of the downloaded dataset.

Parameter description:

- `--root_src` - Path to the original dataset root, containing `train/` and `test/` folders.
- `--root_dst` - Path to the directory, where the pre-processed dataset will be stored.

```bash
python src/preprocessing.py --root_src /path/to/original/dataset/root --root_dst /path/to/preprocessed/dataset/root
```

### Dataset organize way after preprocessing

In the example above, after the test script is executed, the pre-processed images will be stored under
the /path/to/preprocessed/dataset/root path. Below is the file structure of the preprocessed dataset.

```text
.
└─ GOPRO_preprocessed
  ├─ train
  │  ├─ blur
  │  │  ├─ 1.png
  │  │  ├─ ...
  │  │  └─ 2103.png
  │  └─ sharp
  │     ├─ 1.png
  │     ├─ ...
  │     └─ 2103.png
  └─ test
     ├─ blur
     │  ├─ 1.png
     │  ├─ ...
     │  └─ 1111.png
     └─ sharp
        ├─ 1.png
        ├─ ...
        └─ 1111.png
```

# [Environmental requirements](#contents)

- Hardware (GPU)
    - Prepare hardware environment with GPU processor
- Framework
    - [MindSpore](https://www.mindspore.cn/install)
- For details, see the following resources:
    - [MindSpore Tutorial](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)
- Additional python packages:
    - Pillow
    - scikit-image
    - PyYAML

    Install additional packages manually or using `pip install -r requirements.txt` command in the model directory.

## [Quick start](#contents)

After installing MindSpore via the official website and additional packages, you can start training and evaluation as
follows:

- Running on GPU

    ```bash
    # run the training example
    python ./train.py --dataset_root /path/to/dataset/root --ckpt_save_directory /save/checkpoint/directory
    # or
    bash scripts/run_standalone_train_gpu.sh /path/to/dataset/root /save/checkpoint/directory

    # run the distributed training example
    bash scripts/run_distribute_train_gpu.sh /path/to/dataset/root /save/checkpoint/directory

    # run the evaluation example
    python ./eval.py --dataset_root /path/to/dataset/root \
                      --ckpt_file /path/to/eval/checkpoint.ckpt \
                      --img_save_directory /path/to/result/images
    # or
    bash scripts/run_eval_gpu.sh /path/to/dataset/root /path/to/eval/checkpoint.ckpt /path/to/result/images
    ```

# [Script description](#contents)

## [Script and sample code](#contents)

```text
.
└─ cv
  └─ MIMO-UNet
    ├── configs
      ├── gpu_config.yaml                  # Config for training on GPU
    ├── scripts
      ├── run_distribute_train_gpu.sh      # Distributed training on GPU shell script
      ├── run_standalone_train_gpu.sh      # Shell script for single GPU training
      ├── run_eval_gpu.sh                  # GPU evaluation script
    ├─ src
      ├─ data_augment.py                   # Augmentation
      ├─ data_load.py                      # Dataloader
      ├─ init_weights.py                   # Weights initializers
      ├─ layers.py                         # Model layers
      ├─ loss.py                           # Loss function
      ├─ metric.py                         # Metrics
      ├─ mimo_unet.py                       # MIMO-UNet architecture
      ├─ preprocessing.py
    ├─ eval.py                            # test script
    ├─ train.py                           # train script
    ├─ export.py                          # export script
    ├─ requirements.txt                   # requirements file
    └─ README.md                          # MIMO-UNet file English description
```

## [Training process](#contents)

### [Standalone training](#contents)

- Running on GPU

    Description of parameters:

    - `--dataset_root` - Path to the dataset root, containing `train/` and `test/` folders
    - `--ckpt_save_directory` - Output directory, where the data from the train process will be stored

    ```bash
    python ./train.py --dataset_root /path/to/dataset/root --ckpt_save_directory /save/checkpoint/directory
    # or
    bash scripts/run_standalone_train_gpu.sh [DATASET_PATH] [OUTPUT_CKPT_DIR]
    ```

    - DATASET_PATH - Path to the dataset root, containing `train/` and `test/` folders
    - OUTPUT_CKPT_DIR - Output directory, where the data from the train process will be stored

### [Distributed training](#contents)

- Running on GPU

    ```bash
    bash scripts/run_distribute_train_gpu.sh [DATASET_PATH] [OUTPUT_CKPT_DIR]
    ```

    - DATASET_PATH - Path to the dataset root, containing `train/` and `test/` folders
    - OUTPUT_CKPT_DIR - Output directory, where the data from the train process will be stored

## [Evaluation process](#contents)

### [Evaluate](#contents)

Calculate PSNR metric and save deblured images.

When evaluating, select the last generated checkpoint and pass it to the appropriate parameter of the validation script.

- Running on GPU

    Description of parameters:

    - `--dataset_root` - Path to the dataset root, containing `train/` and `test/` folders
    - `--ckpt_file` - path to the checkpoint containing the weights of the trained model.
    - `--img_save_directory` - Output directory, where the images from the validation process will be stored.
    Optional parameter. If not specified, validation images will not be saved.

    ```bash
    python ./eval.py --dataset_root /path/to/dataset/root \
                     --ckpt_file /path/to/eval/checkpoint.ckpt \
                     --img_save_directory /path/to/result/images  # save validation images
    # or
    python ./eval.py --dataset_root /path/to/dataset/root \
                     --ckpt_file /path/to/eval/checkpoint.ckpt  # don't save validation images
    # or
    bash scripts/run_eval_gpu.sh [DATASET_PATH] [CKPT_PATH] [SAVE_IMG_DIR]  # save validation images
    # or
    bash scripts/run_eval_gpu.sh [DATASET_PATH] [CKPT_PATH]  # don't save validation images
    ```

    - DATASET_PATH - Path to the dataset root, containing `train/` and `test/` folders
    - CKPT_PATH - path to the checkpoint containing the weights of the trained model.
    - SAVE_IMG_DIR - Output directory, where the images from the validation process will be stored. Optional parameter. If not specified, validation images will not be saved.

    After the test script is executed, the deblured images are stored in `/path/to/result/img/` if the path was specified.

## [Inference process](#contents)

### [Export MindIR](#contents)

```bash
python export.py --ckpt_file /path/to/mimounet/checkpoint.ckpt --export_device_target GPU --export_file_format MINDIR
```

The script will generate the corresponding MINDIR file in the current directory.

# [Model description](#contents)

## [Performance](#contents)

### [Training Performance](#contents)

| Parameters                 | MIMO-UNet (1xGPU)                                     | MIMO-UNet (8xGPU)                                     |
|----------------------------|-------------------------------------------------------|-------------------------------------------------------|
| Model Version              | MIMO-UNet                                             | MIMO-UNet                                             |
| Resources                  | 1x NV RTX3090-24G                                     | 8x NV RTX3090-24G                                     |
| Uploaded Date              | 04 / 12 / 2022 (month/day/year)                       | 04 / 12 / 2022 (month/day/year)                       |
| MindSpore Version          | 1.6.1                                                 | 1.6.1                                                 |
| Dataset                    | GOPRO_Large                                           | GOPRO_Large                                           |
| Training Parameters        | batch_size=4, lr=0.0001 and bisected every 500 epochs | batch_size=4, lr=0.0005 and bisected every 500 epochs |
| Optimizer                  | Adam                                                  | Adam                                                  |
| Outputs                    | images                                                | images                                                |
| Speed                      | 132 ms/step                                           | 167 ms/step                                           |
| Total time                 | 5d 6h 4m                                              | 9h 15m                                                |
| Checkpoint for Fine tuning | 26MB(.ckpt file)                                      | 26MB(.ckpt file)                                      |

### [Evaluation Performance](#contents)

| Parameters        | MIMO-UNet (1xGPU)               |
|-------------------|---------------------------------|
| Model Version     | MIMO-UNet                       |
| Resources         | 1x NV RTX3090-24G               |
| Uploaded Date     | 04 / 12 / 2022 (month/day/year) |
| MindSpore Version | 1.6.1                           |
| Datasets          | GOPRO_Large                     |
| Batch_size        | 1                               |
| Outputs           | images                          |
| PSNR metric       | 1p: 31.47, 8p: 31.27            |

# [Description of Random Situation](#contents)

In train.py, we set the seed inside the “train" function.

# [ModelZoo homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models)
