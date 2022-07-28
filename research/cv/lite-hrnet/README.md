# Contents

- [Contents](#contents)
    - [Lite-HRNet Description](#lite-hrnet-description)
    - [Model Architecture](#model-architecture)
    - [Dataset](#dataset)
        - [Dataset used: COCO](#dataset-used-coco)
            - [Dataset organize way](#dataset-organize-way)
    - [Environment Requirements](#environment-requirements)
    - [Quick Start](#quick-start)
    - [Script Description](#script-description)
        - [Script and Sample Code](#script-and-sample-code)
        - [Parameter configuration](#parameter-configuration)
        - [Training Process](#training-process)
            - [Training](#training)
                - [Run Lite-HRNet on GPU](#run-lite-hrnet-on-gpu)
        - [Evaluation Process](#evaluation-process)
            - [Evaluation](#evaluation)
    - [Model Description](#model-description)
        - [Performance](#performance)
            - [Training Performance](#training-performance)
            - [Evaluation Performance](#evaluation-performance)
    - [Description of Random Situation](#description-of-random-situation)
    - [ModelZoo Homepage](#modelzoo-homepage)

## [Lite-HRNet Description](#contents)

Lite-HRNet is a modified High-Resolution Network (HRNet), focused on achieving similar performance with reduced computational complexity. Lite-HRNet is implemented for human pose estimation task, but can be easily applied to semantic segmentation.

[Paper](https://arxiv.org/abs/2104.06403): C. Yu, B. Xiao, C. Gao, L. Yuan, L. Zhang, N. Sang, J. Wang. Lite-HRNet: A Lightweight High-Resolution Network.

## [Model Architecture](#contents)

Lite-HRNet consists of several sequential stages, each of those is constructed of 1 to 4 parallel multi-resolution convolution streams. To increase computational performance, Lite-HRNet uses shuffle block alternative, named Conditional channel weighting for information exchange between different resolution streams.

## [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

### Dataset used: [COCO](<https://cocodataset.org/#home>)

COCO is a large-scale object detection, segmentation, and captioning dataset. The COCO Keypoint Detection Task requires localization of person keypoints in challenging, uncontrolled conditions. The keypoint task involves simultaneously detecting people and localizing their keypoints (person locations are not given at test time).

#### Dataset organize way

```text
.
└─coco
  ├─annotations
    ├─person_keypoints_train2017.json
    └─person_keypoints_val2017.json
  ├─person_detection_results
    └─COCO_val2017_detections_AP_H_56_person.json
  ├─train2017
  ├─val2017
...
```

## [Environment Requirements](#contents)

- Hardware（GPU）
    - Prepare hardware environment with GPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

## [Quick Start](#contents)

After installing MindSpore via the official website, choose model architecture in `src/config.py` file.
You can start training and evaluation as follows:

- Training

For GPU training, set `device = 'GPU'` in `src/config.py`.

```bash
# Single GPU training
bash ./scripts/run_standalone_train.sh [DEVICE_ID]

# Multi-GPU training
bash ./scripts/run_distributed_train_gpu.sh [RANK_SIZE] [DEVICE_START]
```

Example：

  ```bash
  # Single GPU training
  bash ./scripts/run_standalone_train.sh 0

  # Multi-GPU training
  bash ./scripts/run_distributed_train_gpu.sh 8 0
  ```

- Evaluation：

```bash
bash ./scripts/run_eval.sh [DEVICE_ID]
```

Example：

  ```bash
  bash ./scripts/run_eval.sh 0
  ```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```text
|-- README.md                                      # English README
|-- eval.py                                        # Evaluation
|-- export.py                                      # MINDIR model export
|-- requirements.txt                               # pip dependencies
|-- configs
|   |-- coco_base.py                               # Basic COCO dataset configuration
|   |-- litehrnet_18_coco_256x192                  # Lite-HRNet-18 model configuration
|   |-- litehrnet_30_coco_256x192                  # Lite-HRNet-30 model configuration
|-- scripts
|   |-- run_distributed_train_gpu.sh               # GPU distributed training script
|   |-- run_eval.sh                                # Evaluation script
|   |-- run_export.sh                              # MINDIR model export script
|   `-- run_standalone_train.sh                    # Single-device training script
|-- src
|   |-- mmpose                                     # MMPose adapted code for top-down dataset
|   |-- callback.py                                # Custom callback functions
|   |-- config.py                                  # Configuration file
|   |-- lr_scheduler.py                            # Learning rate scheduler utilities
|   |-- loss.py                                    # Loss function module
|   |-- model.py                                   # Lite-HRNet model architecture
|   |-- utils.py                                   # General utilities
|   `-- test_utils.py                              # Evaluation utility functions
`-- train.py                                       # Training

```

### [Parameter configuration](#contents)

Parameters for both training and evaluation can be set in `src/config.py`.

```python
experiment_cfg = {
'device': 'GPU',
'model_config': 'litehrnet_18_coco_256x192',
'learning_rate': 5e-5,
'loss_scale': 2 ** 16,
'weight_decay': 1e-6,
'experiment_tag': 'lhrnet18_256_coco_default',
'checkpoint_path': None,
'start_epoch': 0,
'checkpoint_interval': 5,
'random_seed': None
}
```

### [Training Process](#contents)

#### Training

##### Run Lite-HRNet on GPU

For GPU training, set `device = 'GPU'` in `src/config.py`.

- Training using single device (1p)

```bash
bash ./scripts/run_standalone_train.sh 0
```

- Distributed Training (8p)

```bash
bash ./scripts/run_distributed_train_gpu.sh 8 0
```

Checkpoints will be saved in `./checkpoints/` folder. Checkpoint filename format: `[MODEL_NAME].ckpt`.

### [Evaluation Process](#contents)

#### Evaluation

Evaluation script uses checkpoint file with `[MODEL_NAME]` specified in `./src/config.py` as 'model_config'. To start evaluation, run the following command:

```bash
bash ./scripts/run_eval.sh [DEVICE_ID]
# Example:
bash ./scripts/run_eval.sh 0
```

## [Model Description](#contents)

### [Performance](#contents)

#### Training Performance

Training performance in the following tables is obtained by the Lite-HRNet model based on the COCO dataset:

| Parameters | Lite-HRNet-18-256x192 (8GPU) |
| ------------------- | -------------------|
| Model Version | Lite-HRNet-18-256x192 |
| Resource | Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz, 8x V100-PCIE |
| Uploaded Date | 2022-02-22 |
| MindSpore version | 1.5.0 |
| Dataset | COCO |
| Training Parameters | seed=None；epoch=210；batch_size = 64；lr=5e-5；weight_decay = 1e-6, loss_scale = 2 ^ 16 |
| Optimizer | Adam with Weight Decay |
| Loss Function | MSE loss |
| Outputs | Keypoint heatmaps |
| Loss value | 0.0008  |
| Average checkpoint (.ckpt file) size | 4.6 MB |
| Speed | 588 ms/step, 172 s/epoch |
| Total time | 9 hours 59 minutes |
| Scripts | [Lite-HRNet training script](https://gitee.com/mindspore/models/tree/master/research/cv/lite-hrnet/train.py) |

| Parameters | Lite-HRNet-30-256x192 (8GPU) |
| ------------------- | -------------------|
| Model Version | Lite-HRNet-30-256x192 |
| Resource | Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz, 8x V100-PCIE |
| Uploaded Date | 2022-02-22 |
| MindSpore version | 1.5.0 |
| Dataset | COCO |
| Training Parameters | seed=None；epoch=210；batch_size = 64；lr=5e-5；weight_decay = 1e-6, loss_scale = 2 ^ 16 |
| Optimizer | Adam with Weight Decay |
| Loss Function | MSE loss |
| Outputs | Keypoint heatmaps |
| Loss value | 0.0008  |
| Average checkpoint (.ckpt file) size | 7.1 MB |
| Speed | 953 ms/step, 278 s/epoch |
| Total time | 15 hours 9 minutes |
| Scripts | [Lite-HRNet training script](https://gitee.com/mindspore/models/tree/master/research/cv/lite-hrnet/train.py) |

#### Evaluation Performance

- Evaluation performance in the following tables is obtained by the Lite-HRNet model based on the COCO dataset:

| Parameters | Lite-HRNet-18-256x192 (8GPU) |
| ------------------- | ------------------- |
| Model Version | Lite-HRNet-18-256x192 |
| Resource | Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz, 8x V100-PCIE |
| Uploaded Date | 2022-02-22 |
| MindSpore version | 1.5.0 |
| Dataset | COCO |
| Loss Function | MSE loss |
| AP | 0.626 |
| AP50 | 0.859 |
| AP75 | 0.705 |
| AR | 0.689 |
| AR50 | 0.903 |
| Scripts | [Lite-HRNet evaluation script](https://gitee.com/mindspore/models/tree/master/research/cv/lite-hrnet/eval.py) |

| Parameters | Lite-HRNet-30-256x192 (8GPU) |
| ------------------- | ------------------- |
| Model Version | Lite-HRNet-30-256x192 |
| Resource | Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz, 8x V100-PCIE |
| Uploaded Date | 2022-02-22 |
| MindSpore version | 1.5.0 |
| Dataset | COCO |
| Loss Function | MSE loss |
| AP | 0.652 |
| AP50 | 0.871 |
| AP75 | 0.735 |
| AR | 0.715 |
| AR50 | 0.915 |
| Scripts | [Lite-HRNet evaluation script](https://gitee.com/mindspore/models/tree/master/research/cv/lite-hrnet/eval.py) |

## [Description of Random Situation](#contents)

Global training random seed is fixed in `src/config.py` with `random_seed` parameter. 'None' value will execute training without dataset shuffle.

## [ModelZoo Homepage](#contents)  

Please check the official [homepage](https://gitee.com/mindspore/models).  
