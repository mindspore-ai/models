# Contents

- [Contents](#contents)
    - [M2Det Description](#m2det-description)
    - [Model Architecture](#model-architecture)
    - [Dataset](#dataset)
        - [Dataset used: COCO](#dataset-used-ms-coco)
            - [Dataset organize way](#dataset-organize-way)
    - [Environment Requirements](#environment-requirements)
    - [Quick Start](#quick-start)
    - [Script Description](#script-description)
        - [Script and Sample Code](#script-and-sample-code)
        - [Parameter configuration](#parameter-configuration)
        - [Training Process](#training-process)
            - [Training](#training)
                - [Run M2Det on GPU](#run-m2det-on-gpu)
        - [Evaluation Process](#evaluation-process)
            - [Evaluation](#evaluation)
    - [Model Description](#model-description)
        - [Performance](#performance)
            - [Training Performance](#training-performance)
            - [Evaluation Performance](#evaluation-performance)
    - [Description of Random Situation](#description-of-random-situation)
    - [ModelZoo Homepage](#modelzoo-homepage)

## [M2Det Description](#contents)

M2Det (Multi-Level Multi-Scale Detector) is an end-to-end one-stage object detection model. It uses Multi-Level Feature Pyramid Network (MLFPN) to extract features from input image, and then produces dense bounding boxes and category scores.

[Paper](https://qijiezhao.github.io/imgs/m2det.pdf): Q. Zhao, T. Sheng, Y.Wang, Zh. Tang, Y. Chen, L. Cai, H. Ling. M2Det: A Single-Shot Object Detector base on Multi-Level Feature Pyramid Network.

## [Model Architecture](#contents)

M2Det consists of several modules. Feature Fusion Module (FFM) rescales and concatenates features from several backbone feature layers (VGG, ResNet, etc.) to produce base feature for further modules. Thinned U-shape Modules (TUMs) use encoder-decoder architecture to produce multi-level multi-scale features, which afterwards aggregated by Scale-wise Aggregation Module (SFAM). Resulting Multi-Level Feature Pyramid is used by prediction layers to achieve local bounding box regression and classification.

## [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

### Dataset used: [MS-COCO](https://cocodataset.org/#download)

COCO is a large-scale object detection, segmentation, and captioning dataset. The COCO train, validation, and test sets, containing more than 200,000 images and 80 object categories. All object instance are annotated with bounding boxes and detailed segmentation mask.

For training the M2Det model, download the following files:

- 2014 Train images [83K / 13GB]
- 2014 Val images [41K / 6GB]
- 2015 Test images [81K / 12GB]
- 2014 Train/Val annotations [241MB]
- 2014 Testing Image info [1MB]
- 2015 Testing Image info [2MB]

#### Dataset organize way

```text
.
└─ coco
  ├─ annotations
    ├── captions_train2014.json
    ├── captions_val2014.json
    ├── image_info_test2014.json
    ├── image_info_test2015.json
    ├── image_info_test-dev2015.json
    ├── instances_minival2014.json
    ├── instances_train2014.json
    ├── instances_val2014.json
    └── instances_valminusminival2014.json
  ├─images
    ├── test2015
      └── COCO_test2015_*.jpg
    ├── train2014
      └── COCO_train2014_*.jpg
    └── val2014
      └── COCO_val2014_*.jpg

...
```

You can find instances_minival2014.json and instances_valminusminival2014.json here: http://datasets.d2.mpi-inf.mpg.de/hosang17cvpr/coco_minival2014.tar.gz

## [Environment Requirements](#contents)

- Hardware（GPU）
    - Prepare hardware environment with a GPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

## [Quick Start](#contents)

After installing MindSpore via the official website, specify dataset location in `src/config.py` file.
Run Soft-NMS building script with the following command:

```bash
bash ./scripts/make.sh
```

Download pretrained VGG-16 backbone from https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
Convert pretrained VGG-16 backbone to Mindspore format with the following command:

```bash
bash ./scripts/convert_vgg.sh /path/to/vgg16_reducedfc.pth
```

A converted checkpoint will be in the same directory as the original file, but with ".ckpt" extension.

You can start training and evaluation as follows:

- Training

    For GPU training, set `device = 'GPU'` in `src/config.py`.

    ```bash
    # Single GPU training
    bash ./scripts/run_standalone_train.sh [DEVICE_ID] [PRETRAINED_BACKBONE] [DATASET_PATH]

    # Multi-GPU training
    bash ./scripts/run_distributed_train_gpu.sh [RANK_SIZE] [DEVICE_START] [PRETRAINED_BACKBONE] [DATASET_PATH]
    ```

    Example：

    ```bash
    # Single GPU training
    bash ./scripts/run_standalone_train.sh 0 /path/to/vgg16_reducedfc.ckpt /path/to/COCO/

    # Multi-GPU training
    bash ./scripts/run_distributed_train_gpu.sh 8 0 /path/to/vgg16_reducedfc.ckpt /path/to/COCO/
    ```

- Evaluation：

    ```bash
    bash ./scripts/run_eval.sh [DEVICE_ID] [CHECKPOINT_PATH] [DATASET_PATH]
    ```

    Example：

    ```bash
    bash ./scripts/run_eval.sh 0 /path/to/checkpoint /path/to/COCO/
    ```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```text
|-- README.md                                      # English README
|-- convert.py                                     # Script for pretrained VGG backbone conversion
|-- eval.py                                        # Evaluation
|-- export.py                                      # MINDIR model export
|-- requirements.txt                               # pip dependencies
|-- scripts
|   |-- convert_vgg.sh                             # Script for pretrained VGG backbone conversion
|   |-- make.sh                                    # Script for building Soft-NMS function
|   |-- run_distributed_train_gpu.sh               # GPU distributed training script
|   |-- run_eval.sh                                # Evaluation script
|   |-- run_export.sh                              # MINDIR model export script
|   `-- run_standalone_train.sh                    # Single-device training script
|-- src
|   |-- nms
        `-- cpu_nms.pyx                            # Soft-NMS algorithm
|   |-- box_utils.py                               # Function for bounding boxes processing
|   |-- build.py                                   # Script for building Soft-NMS function
|   |-- callback.py                                # Custom callback functions
|   |-- coco_utils.py                              # COCO dataset functions
|   |-- config.py                                  # Configuration file
|   |-- dataset.py                                 # Dataset loader
|   |-- detector.py                                # Bounding box detector
|   |-- loss.py                                    # Multibox loss function
|   |-- lr_scheduler.py                            # Learning rate scheduler utilities
|   |-- model.py                                   # M2Det model architecture
|   |-- priors.py                                  # SSD prior boxes definition
|   `-- utils.py                                   # General utilities
`-- train.py                                       # Training
```

### [Parameter configuration](#contents)

Parameters for both training and evaluation can be set in `src/config.py`.

```python
random_seed = 1
experiment_tag = 'm2det512_vgg16_lr_7.5e-4'

train_cfg = dict(
    lr = 7.5e-4,
    warmup = 5,
    per_batch_size = 7,
    gamma = [0.5, 0.2, 0.1, 0.1],
    lr_epochs = [90, 110, 130, 150, 160],
    total_epochs = 160,
    print_epochs = 10,
    num_workers = 3,
    )

test_cfg = dict(
    cuda = True,
    topk = 0,
    iou = 0.45,
    soft_nms = True,
    score_threshold = 0.1,
    keep_per_class = 50,
    save_folder = 'eval'
    )

optimizer = dict(
    type='SGD',
    momentum=0.9,
    weight_decay=0.00005,
    loss_scale=1,
    dampening=0.0,
    clip_grad_norm=5.)
```

### [Training Process](#contents)

#### Training

##### Run M2Det on GPU

For GPU training, set `device = 'GPU'` in `src/config.py`.

- Training using single device (1p)

    ```bash
    bash ./scripts/run_standalone_train.sh 0 /path/to/vgg16_reducedfc.ckpt /path/to/COCO/
    ```

- Distributed Training (8p)

    ```bash
    bash ./scripts/run_distributed_train_gpu.sh 8 0 /path/to/vgg16_reducedfc.ckpt /path/to/COCO/
    ```

Checkpoints will be saved in `./checkpoints/[EXPERIMENT_TAG]` folder. Checkpoint filename format: `[MODEL.M2DET_CONFIG.BACKBONE]_[MODEL.INPUT_SIZE]-[EPOCH]_[ITERATION].ckpt`. Final checkpoint filename format: `[MODEL.M2DET_CONFIG.BACKBONE]_[MODEL.INPUT_SIZE]-final.ckpt`

### [Evaluation Process](#contents)

#### Evaluation

To start evaluation, run the following command:

```bash
bash ./scripts/run_eval.sh [DEVICE_ID] [CHECKPOINT_PATH] [DATASET_PATH]

# Example:
bash ./scripts/run_eval.sh 0 /path/to/checkpoint /path/to/COCO/
```

## [Model Description](#contents)

### [Performance](#contents)

#### Training Performance

Training performance in the following tables is obtained by the M2Det-512-VGG16 model based on the COCO dataset:

| Parameters | M2Det-512-VGG16 (8GPU) |
| ------------------- | -------------------|
| Model Version | M2Det-512-VGG16 |
| Resource | Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz, 8x V100-PCIE |
| Uploaded Date | 2022-06-29 |
| MindSpore version | 1.5.2 |
| Dataset | COCO |
| Training Parameters | seed=1；epoch=160；batch_size = 7；lr=1e-3；weight_decay = 5e-5, clip_by_global_norm = 4.0 |
| Optimizer | SGD |
| Loss Function | Multibox MSE loss |
| Outputs | Bounding boxes and class scores |
| Loss value | 2.299  |
| Average checkpoint (.ckpt file) size | 507 Mb |
| Speed | 707 ms/step, 1493 s/epoch |
| Total time | 2 days 18 hours 16 minutes |
| Scripts | [M2Det training script](https://gitee.com/mindspore/models/tree/master/research/cv/m2det/train.py) |

#### Evaluation Performance

- Evaluation performance in the following tables is obtained by the M2Det-512-VGG16 model based on the COCO dataset:

| Parameters | M2Det-512-VGG16 (8GPU) |
| ------------------- | ------------------- |
| Model Version | M2Det-512-VGG16 |
| Resource | Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz, 8x V100-PCIE |
| Uploaded Date | 2022-06-29 |
| MindSpore version | 1.5.2 |
| Dataset | COCO |
| Loss Function | Multibox MSE loss |
| AP | 36.2 |
| Scripts | [M2Det evaluation script](https://gitee.com/mindspore/models/tree/master/research/cv/m2det/eval.py) |

## [Description of Random Situation](#contents)

Global training random seed is fixed in `src/config.py` with `random_seed` parameter. 'None' value will execute training without dataset shuffle.

## [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
