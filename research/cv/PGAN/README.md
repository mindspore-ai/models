# content

<!-- TOC -->

- [directory](#directory)
- [PGAN model introduction](#model-introduction)
- [model-architecture](#model-architecture)
- [dataset](#dataset)
- [environmental requirements](#environmental-requirements)
- [Quickstart](#quickstart)
- [Script description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [training process](#training-process)
        - [training](#training)
        - [distributed training](#distributed-training)
    - [evaluation process](#evaluation-process)
        - [assessment](#assessment)
    - [inference process](#inference-process)
        - [Export MindIR](#export-mindir)
        - [Execute reasoning on Ascend310](#execute-reasoning-on-ascend310)
        - [result](#result)
- [model description](#model-description)
    - [performance](#performance)
        - [evaluate performance](#evaluate-performance)
            - [PGAN on CelebA](#pgan-on-celeba)
- [ModelZoo homepage](#modelzoo-homepage)

# model introduction

PGAN refers to Progressive Growing of GANs for Improved Quality, Stability, and Variation, this network is
characterized by the progressive generation of face images

[Paper](https://arxiv.org/abs/1710.10196): Progressive Growing of GANs for Improved Quality, Stability,
and Variation//2018 ICLR

[Reference github](https://github.com/tkarras/progressive_growing_of_gans)

# Model architecture

The entire network structure consists of generator and discriminator. The core idea of ​​the network is to
generate the image with low resolution, add new layers as the training progresses, and gradually begin to generate
more detailed image. Doing so speeds up training and stabilizes it. In addition, this code implements core tricks
such as equalized learning rate, exponential running average, residual structure, and WGANGPGradientPenalty
in the paper.

# Dataset

Dataset web-page: [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

> Note: For this task we use the "Align&Cropped Images" dataset (from the "Downloads" section on the official web-page.).

Dataset link (1.34 GB): [Celeba Aligned and Cropped Images](https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=sharing&resourcekey=0-dYn9z10tMJOBAkviAcfdyQ)

After unpacking the dataset, it should look as follows:

```text
.
└── Celeba
    └── img_align_celeba
        ├── 000001.jpg
        ├── 000002.jpg
        └── ...
```

CelebFaces Attributes Dataset (CelebA) is a large-scale face attributes dataset with over 200K celebrity images,
each with 40 attribute annotations. CelebA is diverse, numerous, and annotated, including

- 10,177 number of identities,
- 202,599 number of face images, and 5 landmark locations, 40 binary attributes annotations per image.

This dataset can be used as a training and test set for the following computer vision tasks: face attribute recognition,
face detection, and face editing and synthesis.

# Environmental requirements

- Hardware (Ascend, GPU)
    - Use Ascend or GPU to build the hardware environment.
- Framework
    - [MindSpore](https://www.mindspore.cn/install)
- For details, see the following resources:
    - [MindSpore Tutorial](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# Quick start

After installing MindSpore through the official website, you can follow the steps below for training and evaluation:

- Ascend processor environment to run

  ```bash
  # run the training example
  export DEVICE_ID=0
  export RANK_SIZE=1
  python train_data_path.py --train_data_path /path/data/image --config_path ./910_config.yaml
  OR
  bash run_standalone_train.sh /path/data/image device_id ./910_config.yaml

  # run the distributed training example
  bash run_distributed_train.sh /path/data/image /path/hccl_config_file ./910_config.yaml
  # run the evaluation example
  export DEVICE_ID=0
  export RANK_SIZE=1
  python eval.py --checkpoint_g=/path/checkpoint --device_id=0
  OR
  bash run_eval.sh /path/checkpoint 0
  ```

- GPU environment to run

  ```bash
  # run the training example
  python ./train.py --config_path /path/to/gpu_config.yaml --train_data_path /path/to/img_align_celeba > ./train.log  2>&1 &
  OR
  bash script/run_standalone_train_gpu.sh ./gpu_config.yaml /path/to/img_align_celeba

  # run the distributed training example
  bash script/run_distribute_train_gpu.sh /path/to/gpu_config.yaml /path/to/img_align_celeba

  # run the evaluation example
  python -u eval.py \
       --checkpoint_g=/path/to/checkpoint \
       --device_target GPU \
       --device_id=/path/to/checkpoint \
       --measure_ms_ssim=True \
       --original_img_dir=/path/to/img_align_celeba
  OR
  bash script/run_eval_gpu.sh /path/to/checkpoint 0 True /path/to/img_align_celeba
  ```

  For evaluation scripts, the checkpoint file is placed by default by the training script in
  In the `/output/{scale}/checkpoint` directory, you need to pass the name of the checkpoint file (Generator)
  as a parameter when executing the script.

# script description

## Script and sample code

```text
.
└─ cv
  └─ PGAN
    ├── script
      ├──run_distribute_train_gpu.sh      # Distributed training on GPU shell script
      ├──run_distributed_train_ascend.sh  # Distributed training shell script
      ├──run_infer_310.sh                 # Inference on Ascend 310
      ├──run_standalone_train.sh          # Shell script for single card training
      ├──run_standalone_train_GPU.sh      # Shell script for single GPU training
      ├──run_eval_ascend.sh               # evaluation script
      ├──run_eval_GPU.sh                  # GPU evaluation script
    ├─ src
      ├─ customer_layer.py                # Basic cell
      ├─ dataset.py                       # data loading
      ├─ image_transform.py               # process image function
      ├─ metrics.py                       # Metric function
      ├─ network_D.py                     # Discriminate network
      ├─ network_G.py                     # Generate network
      ├─ optimizer.py                     # loss calculation
      ├─ time_monitor.py                  # time monitor
    ├─ eval.py                            # test script
    ├─ export.py                          # MINDIR model export script
    ├─ 910_config.yaml                    # Ascend config
    ├─ gpu_config.yaml                    # GPU config
    ├─ modelarts_config.yaml              # Ascend config
    ├─ README_CN.md                       # PGAN file description
    └─ README.md                          # PGAN file English description
```

## training process

### train

- Ascend processor environment to run

  ```bash
  export DEVICE_ID=0
  export RANK_SIZE=1
  python train.py --train_data_path /path/data/image --config_path ./910_config.yaml
  # or
  bash run_standalone_train.sh /path/data/image device_id ./910_config.yaml
  ```

- GPU environment to run

  ```bash
  python train.py --config_path ./gpu_config.yaml --train_data_path /path/to/img_align_celeba
  # or
  bash run_standalone_train_gpu.sh /path/to/gpu_config.yaml /path/to/img_align_celeba
  ```

  After the training, the output directory will be generated in the current directory. In this directory,
  the corresponding subdirectory will be generated according to the ckpt_dir parameter you set, and the parameters
  of each scale will be saved during training.

### Distributed training

- Ascend processor environment to run

  ```bash
  bash run_distributed_train.sh /path/to/img_align_celeba /path/hccl_config_file ./910_config.yaml
  ```

- GPU environment to run

  ```bash
  bash script/run_distributed_train.sh /path/to/gpu_config.yaml /path/to/img_align_celeba
  ```

  The above shell script will run distributed training in the background. The script will generate the corresponding
  LOG{RANK_ID} directory under the script directory, and the output of each process will be recorded in the
  log_distribute file under the corresponding LOG{RANK_ID} directory. The checkpoint file is saved under
  output/rank{RANK_ID}.

## Evaluation process

### evaluate

- Generate images in the Ascend environment
  User-generated 64 face pictures

  When evaluating, select the generated checkpoint file and pass it into the test script as a parameter.
  The corresponding parameter is `checkpoint_g` (the checkpoint of the generator is saved)

- Use a checkpoint which name starts with `AvG` (for example, AvG_12000.ckpt)

- Ascend processor environment to run

  ```bash
  bash run_eval.sh /path/to/avg/checkpoint 0
  ```

- GPU environment to run

  ```bash
  bash script/run_eval_gpu.sh [CKPT_PATH] [DEVICE_ID] [MEASURE_MSSIM] [DATASET_DIR]
  ```

    - CKPT_PATH - path to the checkpoint
    - DEVICE_ID - device ID
    - MEASURE_MSSIM - Flag to calculate objective metrics. If True, MS-SSIM is calculated,
      otherwise, the script will only generated images of faces.
    - DATASET_DIR - path to the dataset images

  After the test script is executed, the generated images are stored in `img_eval/`.

## Reasoning process

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

### Export MindIR

```bash
python export.py --checkpoint_g [GENERATOR_CKPT_NAME] --device_id [DEVICE_ID]  --device_target [DEVICE_TARGET]
```

- GENERATOR_CKPT_NAME - path to the trained checkpoint (Use AvG_xx.ckpt)
- DEVICE_TARGET - Device target: Ascend or GPU
- DEVICE_ID - Device ID

The script will generate the corresponding MINDIR file in the current directory.

### Perform inference on Ascend310

Before performing inference, the MINDIR model must be exported via the export script. The following commands show
how to edit the properties of images on Ascend310 through commands:

```bash
bash run_infer_310.sh [MINDIR_PATH] [NEED_PREPROCESS] [NIMAGES] [DEVICE_ID]
````

- `MINDIR_PATH` path to the MINDIR file
- `NEED_PREPROCESS` indicates whether the attribute editing file needs to be preprocessed, which can be selected
  from y or n. If y is selected, it means preprocessing (it needs to be set to y when the inference is executed
  for the first time)
- `NIMAGES` indicates the number of generated images.
- `DEVICE_ID` is optional, default is 0.

### result

The inference results are saved in the directory where the script is executed, the pictures after attribute editing
are saved in the `result_Files/` directory, and the time statistics results of inference are saved in the
`time_Result/` directory. The edited image is saved in the format `generated_{NUMBER}.png`.

# model description

## performance

### Evaluate performance

#### PGAN on CelebA

| Parameters          | Ascend 910                                                                     | GPU                                                                            |
|---------------------|--------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| Model Version       | PGAN                                                                           | PGAN                                                                           |
| Resources           | Ascend                                                                         | GPU                                                                            |
| Upload Date         | 09/31/2021 (month/day/year)                                                    | 02/08/2022 (month/day/year)                                                    |
| MindSpore Version   | 1.3.0                                                                          | 1.5.0                                                                          |
| Datasets            | CelebA                                                                         | CelebA                                                                         |
| Training parameters | batch_size=16, lr=0.001                                                        | batch_size=16 for scales 4-64,batch_size=8 for scale128, , lr=0.002            |
| Optimizer           | Adam                                                                           | Adam                                                                           |
| generator output    | image                                                                          | image                                                                          |
| Speed ​​            | 8p: 9h 26m 54ы; 1p: 76h 23m 39s; 1.1s/step                                     | 8: 10h 28m 37s; 1: 83h 45m 34s                                                 |
| Convergence loss    | G:[-232.61 to 273.87] loss D:[-27.736 to 2.601]                                | G:[-232.61 to 273.87] D:[-27.736 to 2.601]                                     |
| MS-SSIM metric      |                                                                                | 0.2948                                                                         |
| Script              | [PGAN script](https://gitee.com/mindspore/models/tree/master/research/cv/PGAN) | [PGAN script](https://gitee.com/mindspore/models/tree/master/research/cv/PGAN) |

> Note: For measuring the metrics and generating the images we are using the checkpoint with prefix AvG (AvG_xxxx.ckpt)

# ModelZoo homepage

Please visit the official website [homepage](https://gitee.com/mindspore/models)
