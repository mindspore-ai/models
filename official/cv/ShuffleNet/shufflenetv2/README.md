# Contents

- [Contents](#contents)
- [ShuffleNetV2 Description](#shufflenetv2-description)
- [Model architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Script description](#script-description)
    - [Script and sample code](#script-and-sample-code)
    - [Training process](#training-process)
        - [Usage](#usage)
        - [Launch](#launch)
        - [Result](#result)
    - [Eval process](#eval-process)
        - [Usage](#usage-1)
        - [Launch](#launch-1)
        - [Result](#result-1)
    - [Inference Process](#inference-process)
        - [Export MindIR](#export-mindir)
        - [Infer](#infer)
        - [result](#result-2)
        - [Infer on CPU After Transfermation](#infer-on-cpu-after-transfermation)
        - [result](#result-3)
- [Model description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
        - [Inference Performance](#inference-performance)
- [ModelZoo Homepage](#modelzoo-homepage)

# [ShuffleNetV2 Description](#contents)

ShuffleNetV2 is a much faster and more accurate network than the previous networks on different platforms such as Ascend or GPU.
[Paper](https://arxiv.org/pdf/1807.11164.pdf) Ma, N., Zhang, X., Zheng, H. T., & Sun, J. (2018). Shufflenet v2: Practical guidelines for efficient cnn architecture design. In Proceedings of the European conference on computer vision (ECCV) (pp. 116-131).

# [Model architecture](#contents)

The overall network architecture of ShuffleNetV2 is show below:

[Link](https://arxiv.org/pdf/1807.11164.pdf)

# [Dataset](#contents)

Dataset used: [imagenet](http://www.image-net.org/)

- Dataset size: ~125G, 1.2M colorful images in 1000 classes
    - Train: 120G, 1.2M images
    - Test: 5G, 50000 images
- Data format: RGB images.
    - Note: Data will be processed in src/dataset.py

Dataset for transfermation:[flower_photos](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz)

- Dataset size: 221MB, 3670 colorful images in 5 classes
    - Train: 177MB, 2934images
    - Test: 44MB, 736 images
- Data format: RGB images.
    - Note: Data will be processed in src/dataset.py

# [Environment Requirements](#contents)

- Hardware(Ascend/GPU/CPU)
    - Prepare hardware environment with Ascend, GPU or CPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below:
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

# [Script description](#contents)

## [Script and sample code](#contents)

```text
+-- ShuffleNetV2
  +-- Readme.md     # descriptions about ShuffleNetV2
  +-- scripts
    +--run_distribute_train_for_ascebd.sh   # shell script for distributed Ascend training
    +--run_distribute_train_for_gpu.sh      # shell script for distributed GPU training
    +--run_eval_for_ascend.sh               # shell script for Ascend evaluation
    +--run_eval_for_gpu.sh                  # shell script for GPU evaluation
    +--run_standalone_train_for_gpu.sh      # shell script for standalone GPU training
  +-- src
    +--config.py                            # parameter configuration
    +--CrossEntropySmooth.py                # loss function for GPU training
    +--dataset.py                           # creating dataset
    +--loss.py                              # loss function for network
    +--lr_generator.py                      # learning rate config
    +--shufflenetv2.py                      # ShuffleNetV2 model network
  +-- cpu_transfer.py                       # transfer script
  +-- dataset_split.py                      # splitting dataset for transfermation script
  +-- quick_start.py                        # quick_start script
  +-- train.py                              # training script
  +-- eval.py                               # evaluation script
```

## [Training process](#contents)

### Usage

You can start training using python or shell scripts. The usage of shell scripts as follows:

- Distributed training on Ascend: bash run_distribute_train_for_ascend.sh [RANK_TABLE_FILE] [DATASET_PATH]
- Distributed training on GPU: bash run_standalone_train_for_gpu.sh [DEVICE_NUM] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)] [DATASET_PATH]
- Standalone training on GPU: bash run_standalone_train_for_gpu.sh [DATASET_PATH]

### Launch

```bash
# training example
  python:
      GPU: mpirun --allow-run-as-root -n 8 --output-filename log_output --merge-stderr-to-stdout python train.py --is_distributed=True --platform='GPU' --dataset_path='~/imagenet' > train.log 2>&1 &
      CPU: python cpu_transfer.py --checkpoint_input_path ./input_ckpt/shufflenetv2_top1acc69.63_top5acc88.72.ckpt --checkpoint_save_path ./save_ckpt/Graph_mode --train_dataset ./data/flower_photos_split/train --use_pynative_mode False --platform CPU
  shell:
      GPU: cd scripts & sh run_distribute_train_for_gpu.sh 8 0,1,2,3,4,5,6,7 ~/imagenet
```

### Result

Training result will be stored in the example path. Checkpoints will be stored at `./checkpoint` by default, and training log will be redirected to `./train/train.log`.

## [Eval process](#contents)

### Usage

You can start evaluation using python or shell scripts. The usage of shell scripts as follows:

- Ascend: bash run_eval_for_ascend.sh [DATASET_PATH] [CHECKPOINT]
- GPU: bash run_eval_for_gpu.sh [DATASET_PATH] [CHECKPOINT]

### Launch

```bash
# infer example
  python:
      Ascend: python eval.py --platform='Ascend' --dataset_path='~/imagenet' --checkpoint='checkpoint_file' > eval.log 2>&1 &
      GPU: CUDA_VISIBLE_DEVICES=0 python eval.py --platform='GPU' --dataset_path='~/imagenet/val/' --checkpoint='checkpoint_file'> eval.log 2>&1 &
      CPU: python eval.py --dataset_path ./data/flower_photos_split/eval --checkpoint_dir ./save_ckpt/Graph_mode --platform CPU --checkpoint ./save_ckpt/Graph_mode/shufflenetv2_1-154_18.ckpt --enable_checkpoint_dir True --use_pynative_mode False
  shell:
      Ascend: cd scripts & sh run_eval_for_ascend.sh '~/imagenet' 'checkpoint_file'
      GPU: cd scripts & sh run_eval_for_gpu.sh '~/imagenet' 'checkpoint_file'
```

### Result

Inference result will be stored in the example path, you can find result in `eval.log`.

## Inference Process

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

### [Export MindIR](#contents)

Export MindIR on local

```shell
python export.py --device_target [PLATFORM] --ckpt_file [CKPT_FILE] --file_format [FILE_FORMAT] --file_name [OUTPUT_FILE_BASE_NAME]
```

The checkpoint_file_path parameter is required,
`PLATFORM` should be in ["Ascend", "GPU", "CPU"]
`FILE_FORMAT` should be in ["AIR", "ONNX", "MINDIR"]

### Infer

Before performing inference, the mindir file must bu exported by `export.py` script. We only provide an example of inference using MINDIR model.
Current batch_Size can only be set to 1.

```shell
bash run_infer_cpp.sh [MINDIR_PATH] [DATASET_NAME] [DATASET_PATH] [NEED_PREPROCESS] [DEVICE_TYPE] [DEVICE_ID]
```

- `MINDIR_PATH` should be the filename of the MINDIR model.
- `DATASET_NAME` should be imagenet2012.
- `DATASET_PATH` should be the path of the val in imaagenet2012 dataset.
- `NEED_PREPROCESS` can be y or n.
- `DEVICE_ID` is optional, default value is 0.

### result

Inference result is saved in current path, you can find result like this in acc.log file.
Top1 acc:  0.69608
Top5 acc:  0.88726

### Infer on CPU After Transfermation

```Python
# CPU inference
python eval.py --dataset_path [eval dataset] --checkpoint_dir [ckpt dir for eavl ] --platform [CPU] --checkpoint [ckpt path for eval] --enable_checkpoint_dir [True/False]--use_pynative_mode [True/False]
```

### result

Inference result is saved in current path, you can find result like this in acc.log file.
Top1 acc:  0.86
Top5 acc:  1

# [Model description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 | Ascend 910                    | GPU                           |CPU(Transfer)                           |
| -------------------------- | ----------------------------- |-------------------------------|-------------------------------|
| Model Version              | ShuffleNetV2                  | ShuffleNetV2                  | ShuffleNetV2                  |
| Resource                   | Ascend 910                    | NV SMX2 V100-32G              |Intel(R)Core(TM) i5-7200U CPU@2.50GHz(4 CPUs) |
| uploaded Date              | 10/09/2021 (month/day/year)   | 09/24/2020 (month/day/year)   | 08/30/2022 (month/day/year)   |
| MindSpore Version          | 1.3.0                         | 1.0.0                         | 1.8                           |
| Dataset                    | ImageNet                      | ImageNet                      |Flower_photos                      |
| Training Parameters        | src/config.py                 | src/config.py                 | src/config.py                 |
| Optimizer                  | Momentum                      | Momentum                      | Momentum                      |
| Loss Function              | SoftmaxCrossEntropyWithLogits | CrossEntropySmooth            | CrossEntropySmooth            |
| Accuracy                   | 69.59%(TOP1)                  | 69.4%(TOP1)                   | 86.4%(TOP1)                   |
| Total time                 | 11.6 h 8ps                    | 49 h 8ps                      |15h18m8.6s                      |

### Inference Performance

| Parameters                 | Ascend 910                    | GPU                           | CPU(Transfer)                           |
| -------------------------- | ----------------------------- |-------------------------------|-------------------------------|
| Resource                   | Ascend 910                    | NV SMX2 V100-32G              |Intel(R)Core(TM) i5-7200U CPU@2.50GHz(4 CPUs)              |
| uploaded Date              | 10/09/2021 (month/day/year)   | 09/24/2020 (month/day/year)   | 08/30/2022 (month/day/year)   |
| MindSpore Version          | 1.3.0                         | 1.0.0                         | 1.8.0                         |
| Dataset                    | ImageNet                      | ImageNet                      |Flower_photos                      |
| batch_size                 | 125                           | 128                           |128                     |
| outputs                    | probability                   | probability                   | probability                   |
| Accuracy                   | acc=69.59%(TOP1)              | acc=69.4%(TOP1)               | acc=86.4%(TOP1)               |

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
