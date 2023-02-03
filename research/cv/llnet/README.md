# Contents

- [LLNet Description](#LLNet-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
    - [Evaluation Process](#evaluation-process)
    - [Inference Process](#inference-process)
        - [Export MindIR](#export-mindir)
        - [Infer on Ascend310](#infer-on-ascend310)
        - [result](#result-2)
- [Model Description](#model-description)
    - [Performance](#performance)  
        - [Training Performance](#evaluation-performance)
        - [Inference Performance](#evaluation-performance)
- [ModelZoo Homepage](#modelzoo-homepage)

# [LLNet Description](#contents)

The authors proposed a deep autoencoder-based approach to identify signal features from low-light images handcrafting and adaptively brighten images without over-amplifying the lighter parts in images (i.e., without saturation of image pixels) in high dynamic range. The network can then be successfully applied to naturally low-light environment and/or hardware degraded images.

[Paper](https://arxiv.org/abs/1511.03995v3): Kin Gwn Lore, Adedotun Akintayo, Soumik Sarkar: LLNet: A Deep Autoencoder Approach to Natural Low-light Image Enhancement. 2015.

# [Model architecture](#contents)

The overall description of LLNet is show below:

[Link](https://arxiv.org/pdf/1511.03995)

# [Dataset](#contents)

Dataset used: [dbimagenes](http://decsai.ugr.es/cvg/dbimagenes/)

- Dataset size: 1.1G, 170 gray images
    - Train: 526.3M, 163 images, 1250 patches per image.
    - Val: 526.3M, 163 images, 1250 patches per image.
    - Test: 1.3M, 5 images, 29241 patches per image.
- Data format: gray images.
    - Note: Data will be processed in src/dataset.py

# [Environment Requirements](#contents)

- Hardware Ascend
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

# [Script description](#contents)

## [Script and sample code](#contents)

```text
.
└─llnet
  ├─ README.md
  ├─ README_CN.md
  ├─── ascend310_infer
  │    ├── build.sh                            # build bash
  │    ├── CMakeLists.txt                      # CMakeLists
  │    ├── inc
  │    │   └── utils.h                         # utils head
  │    └── src
  │        ├── main.cc                         # main function of ascend310_infer
  │        └── utils.cc                        # utils function of ascend310_infer
  ├─── scripts
  │    ├─ run_standalone_train.sh              # launch standalone training with GPU(1p) or Ascend 910 platform(1p)
  │    ├─ run_distribute_train_for_ascend.sh   # launch distributed training with Ascend 910 platform(8p)
  │    ├─ run_infer_310.sh                     # launch evaluating with Ascend 310 platform
  │    └─ run_eval.sh                          # launch evaluating with CPU, GPU, or Ascend 910 platform
  ├─ src
  │    ├─ model_utils
  │    │  ├─ config.py                         # parameter configuration
  │    │  ├─ device_adapter.py                 # device adapter
  │    │  ├─ local_adapter.py                  # local adapter
  │    │  └─ moxing_adapter.py                 # moxing adapter
  │    ├─ dataset.py                           # data reading
  │    ├─ lr_generator.py                      # learning rate generator
  │    └─ llnet.py                             # network definition
  ├─ default_config.yaml                       # parameter configuration
  ├─ eval.py                                   # eval net
  ├─ export.py                                 # export checkpoint for ascend310_infer
  ├─ postprocess.py                            # post process for ascend310_infer
  ├─ preprocess.py                             # pre process for ascend310_infer
  ├─ requirements.txt                          # the pyaml package required by this network
  ├─ test.py                                   # the test for LLNet network
  ├─ train.py                                  # train net
  └─ write_mindrecords.py                      # write the mindrecords for train, eval, and test

```

## [Script Parameters](#contents)

Parameters for both training and evaluating can be set in default_config.yaml.

```text
'random_seed': 1,                              # fix random seed
'rank': 0,                                     # local rank of distributed
'group_size': 1,                               # world size of distributed
'work_nums': 8,                                # number of workers to read the data
'pretrain_epoch_size': 5,                      # the epoch number for pretrain
'finetrain_epoch_size': 300,                   # the epoch number for finetrain
'keep_checkpoint_max': 20,                     # max numbers to keep checkpoints
'save_ckpt_path': './',                        # save checkpoint path
'train_batch_size': 500,                       # input batch size for training
'val_batch_size': 1250,                        # input batch size for evaluation
'lr_init': [0.01, 0.01, 0.001, 0.001],         # initiate learning rate for the first three layer's pretrain and finetrain
'weight_decay': 0.0,                           # weight decay
'momentum': 0.9,                               # momentum
```

## [Training Process](#contents)

### Usage

```bash
# distribute training for Ascend(8p)
bash run_distribute_train_ascend.sh [RANK_TABLE_FILE] [DATASET_PATH]
# standalone training for GPU(1p) or Ascend 910(1p)
bash run_standalone_train.sh [DEVICE_ID] [DATASET_PATH]
```

### Launch

```bash
# distributed training example(8p) for Ascend
bash run_distribute_train_for_ascend.sh /home/hccl_8p_01234567.json /dataset
# standalone training example for GPU(1p) or Ascend 910(1p)
bash run_standalone_train.sh 0 ../dataset
```

You can find checkpoint file together with result in log.

## [Evaluation Process](#contents)

### Usage

```bash
# Evaluation
bash run_eval.sh [DEVICE_ID] [DATASET_PATH] [CHECKPOINT]
```

### Launch

```bash
# Evaluation with checkpoint
bash run_eval.sh 5 ../dataset ./ckpt_5/llnet-rank5-286_408.ckpt
```

### Result

Evaluation result will be stored in the scripts path. Under this, you can find result like the followings in log.
PSNR=21.593(dB) SSIM=0.617

## Inference Process

### [Export MindIR](#contents)

Export MindIR on local

```python
python export.py --device_target [PLATFORM] --device_id [DEVICE_ID] --checkpoint [CHECKPOINT_FILE] --file_format [FILE_FORMAT] --file_name [FILE_NAME]
```

The checkpoint_file parameter is required,
`PLATFORM` should be in ["Ascend", "GPU", "CPU"]
`DEVICE_ID` should be in [0-7]
`FILE_FORMAT` should be in ["AIR", "ONNX", "MINDIR"], the default value is MINDIR
`FILE_NAME` the base name for the exported model, the default value is llnet

### Infer on Ascend310

Before performing inference, the mindir file must bu exported by `export.py` script. We only provide an example of inference using MINDIR model.
Current batch_Size can only be set to 1.

```bash
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATASET_PATH] [NEED_PREPROCESS] [DEVICE_ID]
```

- `MINDIR_PATH` should be the filename of the MINDIR model.
- `DATASET_PATH` should be the path of the dataset.
- `NEED_PREPROCESS` can be y or n. It should by y for the first time of running.
- `DEVICE_ID` is optional, default value is 0.

### result

Inference result is saved in current path, you can find result like this in acc.log file.
PSNR:  21.582 (dB)
SSIM:   0.604

# [Model description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 | Ascend 910                    |
| -------------------------- | ----------------------------- |
| Model Version              | LLNet                         |
| Resource                   | Ascend 910                    |
| uploaded Date              | 07/23/2022 (month/day/year)   |
| MindSpore Version          | 1.5.1                         |
| Dataset                    | dbimagenes                    |
| Training Parameters        | default_config.yaml           |
| Optimizer                  | Adam                          |
| Loss Function              | MSE                           |
| Loss                       | 0.0105                        |
| Total time                 | 0 h 17 m 21 s 2ps             |
| Checkpoint for Fine tuning | 21.5 M(.ckpt file)            |

### Inference Performance

| Parameters                 | Ascend 910                    |
| -------------------------- | ----------------------------- |
| Model Version              | LLNet                         |
| Resource                   | Ascend 910                    |
| uploaded Date              | 07/23/2022 (month/day/year)   |
| MindSpore Version          | 1.5.1                         |
| Dataset                    | dbimagenes                    |
| batch_size                 | 1250                          |
| outputs                    | 289 pixels reconstructed      |
| Accuracy                   | PSNR = 21.593  SSIM = 0.617   |

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
