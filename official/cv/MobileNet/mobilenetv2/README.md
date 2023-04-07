# Contents

- [Contents](#contents)
- [MobileNetV2 Description](#mobilenetv2-description)
- [Model architecture](#model-architecture)
- [Dataset](#dataset)
- [Features](#features)
    - [Mixed Precision(Ascend)](#mixed-precisionascend)
- [Environment Requirements](#environment-requirements)
- [Script description](#script-description)
    - [Script and sample code](#script-and-sample-code)
    - [Training process](#training-process)
        - [Usage](#usage)
        - [Launch](#launch)
        - [Result](#result)
    - [Evaluation process](#evaluation-process)
        - [Usage](#usage-1)
        - [Launch](#launch-1)
        - [Result](#result-1)
    - [ONNX export and evaluation process](#onnx-export-and-evaluation-process)
        - [Export](#export)
        - [Evaluation](#evaluation)
        - [Result](#result-2)
    - [Training with dataset on NFS](#training-with-dataset-on-nfs)
    - [Inference process](#inference-process)
        - [Export MindIR](#export-mindir)
        - [Infer](#infer)
        - [result](#result-3)
- [Model description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
        - [Inference Performance](#inference-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [MobileNetV2 Description](#contents)

MobileNetV2 is tuned to mobile phone CPUs through a combination of hardware- aware network architecture search (NAS) complemented by the NetAdapt algorithm and then subsequently improved through novel architecture advances.Nov 20, 2019.

[Paper](https://arxiv.org/pdf/1801.04381.pdf) Howard, Andrew, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang et al. "Searching for MobileNetV2." In Proceedings of the IEEE International Conference on Computer Vision, pp. 1314-1324. 2019.

# [Model architecture](#contents)

The overall network architecture of MobileNetV2 is show below:

[Link](https://arxiv.org/pdf/1801.04381.pdf)

# [Dataset](#contents)

Dataset used: [imagenet](http://www.image-net.org/)

- Dataset size: ~125G, 224*224 colorful images in 1000 classes
    - Train: 120G, 1281167 images
    - Test: 5G, 50000 images
- Data format: RGB images.
    - Note: Data will be processed in src/dataset.py

# [Features](#contents)

## [Mixed Precision(Ascend)](#contents)

The [mixed precision](https://www.mindspore.cn/tutorials/en/master/advanced/mixed_precision.html) training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data formats, and maintains the network precision achieved by the single-precision training at the same time. Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware.
For FP16 operators, if the input data type is FP32, the backend of MindSpore will automatically handle it with reduced precision. Users could check the reduced-precision operators by enabling INFO log and then searching ‘reduce precision’.

# [Environment Requirements](#contents)

- Hardware（Ascend/GPU/CPU）
    - Prepare hardware environment with Ascend, GPU or CPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

- Running on [ModelArts](https://support.huaweicloud.com/modelarts/)

    ```bash
    # Train 8p with Ascend
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on default_config.yaml file.
    #          Set "distribute=True" on default_config.yaml file.
    #          Set "need_modelarts_dataset_unzip=True" on default_config.yaml file.
    #          Set "modelarts_dataset_unzip_name='ImageNet_Original'" on default_config.yaml file.
    #          Set "dataset_path='/cache/data'" on default_config.yaml file.
    #          Set "epoch_size: 200" on default_config.yaml file.
    #          (optional)Set "checkpoint_url='s3://dir_to_your_pretrained/'" on default_config.yaml file.
    #          Set other parameters on default_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "need_modelarts_dataset_unzip=True" on the website UI interface.
    #          Add "modelarts_dataset_unzip_name='ImageNet_Original'" on the website UI interface.
    #          Add "distribute=True" on the website UI interface.
    #          Add "dataset_path=/cache/data" on the website UI interface.
    #          Add "epoch_size: 200" on the website UI interface.
    #          (optional)Add "checkpoint_url='s3://dir_to_your_pretrained/'" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Prepare model code
    # (3) Upload or copy your pretrained model to S3 bucket if you want to finetune.
    # (4) Perform a or b. (suggested option a)
    #       a. First, zip MindRecord dataset to one zip file.
    #          Second, upload your zip dataset to S3 bucket.(you could also upload the origin mindrecord dataset, but it can be so slow.)
    #       b. Upload the original dataset to S3 bucket.
    #           (Data set conversion occurs during training process and costs a lot of time. it happens every time you train.)
    # (5) Set the code directory to "/path/mobilenetv2" on the website UI interface.
    # (6) Set the startup file to "train.py" on the website UI interface.
    # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (8) Create your job.
    #
    # Train 1p with Ascend
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on default_config.yaml file.
    #          Set "need_modelarts_dataset_unzip=True" on default_config.yaml file.
    #          Set "modelarts_dataset_unzip_name='ImageNet_Original'" on default_config.yaml file.
    #          Set "dataset_path='/cache/data'" on default_config.yaml file.
    #          Set "epoch_size: 200" on default_config.yaml file.
    #          (optional)Set "checkpoint_url='s3://dir_to_your_pretrained/'" on default_config.yaml file.
    #          Set other parameters on default_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "need_modelarts_dataset_unzip=True" on the website UI interface.
    #          Add "modelarts_dataset_unzip_name='ImageNet_Original'" on the website UI interface.
    #          Add "dataset_path='/cache/data'" on the website UI interface.
    #          Add "epoch_size: 200" on the website UI interface.
    #          (optional)Add "checkpoint_url='s3://dir_to_your_pretrained/'" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Prepare model code
    # (3) Upload or copy your pretrained model to S3 bucket if you want to finetune.
    # (4) Perform a or b. (suggested option a)
    #       a. zip MindRecord dataset to one zip file.
    #          Second, upload your zip dataset to S3 bucket.(you could also upload the origin mindrecord dataset, but it can be so slow.)
    #       b. Upload the original dataset to S3 bucket.
    #           (Data set conversion occurs during training process and costs a lot of time. it happens every time you train.)
    # (5) Set the code directory to "/path/mobilenetv2" on the website UI interface.
    # (6) Set the startup file to "train.py" on the website UI interface.
    # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (8) Create your job.
    #
    # Eval 1p with Ascend
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on default_config.yaml file.
    #          Set "need_modelarts_dataset_unzip=True" on default_config.yaml file.
    #          Set "modelarts_dataset_unzip_name='ImageNet_Original'" on default_config.yaml file.
    #          Set "checkpoint_url='s3://dir_to_your_trained_model/'" on base_config.yaml file.
    #          Set "checkpoint='./mobilenetv2/mobilenetv2_trained.ckpt'" on default_config.yaml file.
    #          Set "dataset_path='/cache/data'" on default_config.yaml file.
    #          Set other parameters on default_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "need_modelarts_dataset_unzip=True" on the website UI interface.
    #          Add "modelarts_dataset_unzip_name='ImageNet_Original'" on the website UI interface.
    #          Add "checkpoint_url='s3://dir_to_your_trained_model/'" on the website UI interface.
    #          Add "checkpoint='./mobilenetv2/mobilenetv2_trained.ckpt'" on the website UI interface.
    #          Add "dataset_path='/cache/data'" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Prepare model code
    # (3) Upload or copy your trained model to S3 bucket.
    # (4) Perform a or b. (suggested option a)
    #       a. First, zip MindRecord dataset to one zip file.
    #          Second, upload your zip dataset to S3 bucket.(you could also upload the origin mindrecord dataset, but it can be so slow.)
    #       b. Upload the original dataset to S3 bucket.
    #           (Data set conversion occurs during training process and costs a lot of time. it happens every time you train.)
    # (5) Set the code directory to "/path/mobilenetv2" on the website UI interface.
    # (6) Set the startup file to "eval.py" on the website UI interface.
    # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (8) Create your job.
    ```

- Export on ModelArts (If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start evaluating as follows)

1. Export s8 multiscale and flip with voc val dataset on modelarts, evaluating steps are as follows:

    ```python
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on base_config.yaml file.
    #          Set "file_name='mobilenetv2'" on base_config.yaml file.
    #          Set "file_format='MINDIR'" on base_config.yaml file.
    #          Set "checkpoint_url='/The path of checkpoint in S3/'" on beta_config.yaml file.
    #          Set "ckpt_file='/cache/checkpoint_path/model.ckpt'" on base_config.yaml file.
    #          Set other parameters on base_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "file_name='mobilenetv2'" on the website UI interface.
    #          Add "file_format='MINDIR'" on the website UI interface.
    #          Add "checkpoint_url='/The path of checkpoint in S3/'" on the website UI interface.
    #          Add "ckpt_file='/cache/checkpoint_path/model.ckpt'" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Upload or copy your trained model to S3 bucket.
    # (3) Set the code directory to "/path/mobilenetv2" on the website UI interface.
    # (4) Set the startup file to "export.py" on the website UI interface.
    # (5) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (6) Create your job.
    ```

# [Script description](#contents)

## [Script and sample code](#contents)

```text
├── MobileNetV2
  ├── README.md                  # descriptions about MobileNetV2
  ├── ascend310_infer            # application for 310 inference
  ├── scripts
  │   ├──run_infer_310.sh        # shell script for 310 infer
  │   ├──run_train.sh            # shell script for train, fine_tune or incremental  learn with CPU, GPU or Ascend
  │   ├──run_eval.sh             # shell script for evaluation with CPU, GPU or Ascend
  │   ├──cache_util.sh           # a collection of helper functions to manage cache
  │   ├──run_train_nfs_cache.sh  # shell script for train with NFS dataset and leverage caching service for better performance
  ├── src
  │   ├──aipp.cfg                # aipp config
  │   ├──dataset.py              # creating dataset
  │   ├──lr_generator.py         # learning rate config
  │   ├──mobilenetV2.py          # MobileNetV2 architecture
  │   ├──models.py               # contain define_net and Loss, Monitor
  │   ├──utils.py                # utils to load ckpt_file for fine tune or incremental learn
  │   └──model_utils
  │      ├──config.py            # Processing configuration parameters
  │      ├──device_adapter.py    # Get cloud ID
  │      ├──local_adapter.py     # Get local ID
  │      └──moxing_adapter.py    # Parameter processing
  ├── default_config.yaml        # Training parameter profile(ascend)
  ├── default_config_boost.yaml        # Training parameter profile(ascend boost)
  ├── default_config_cpu.yaml    # Training parameter profile(cpu)
  ├── default_config_gpu.yaml    # Training parameter profile(gpu)
  ├── train.py                   # training script
  ├── eval.py                    # evaluation script
  ├── export.py                  # export mindir script
  ├── mindspore_hub_conf.py      #  mindspore hub interface
  ├── postprocess.py             # postprocess script
```

## [Training process](#contents)

### Usage

You can start training using python or shell scripts. The usage of shell scripts as follows:

- Ascend: bash run_train.sh Ascend [CONFIG_PATH] [DEVICE_NUM] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)] [DATASET_PATH] [CKPT_PATH(optional)] [FREEZE_LAYER(optional)] [FILTER_HEAD(optional)]
- GPU: bash run_trian.sh GPU [CONFIG_PATH] [DEVICE_NUM] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)] [DATASET_PATH] [CKPT_PATH] [FREEZE_LAYER] [FILTER_HEAD]
- CPU: bash run_trian.sh CPU [CONFIG_PATH] [DATASET_PATH] [CKPT_PATH] [FREEZE_LAYER] [FILTER_HEAD]

> Notice! Currently, when using the Ascend platform, the `VISIABLE_DEVICES` parameter is invalid, that is, you can NOT specify the computing device through this parameter.

`DATASET_PATH` is the dataset path. We use `ImageFolderDataset` as default dataset, which is a source dataset that reads images from a tree of directories. The directory structure is as follows, and you should use `DATASET_PATH=dataset/` for training and evaluation:

```path
        └─dataset
            └─train
              ├─class1
                ├─0001.jpg
                ......
                └─xxxx.jpg
              ......
              ├─classx
                ├─0001.jpg
                ......
                └─xxxx.jpg
            └─validation_preprocess
              ├─class1
                ├─0001.jpg
                ......
                └─xxxx.jpg
              ......
              ├─classx
                ├─0001.jpg
                ......
                └─xxxx.jpg
```

`CKPT_PATH` `FREEZE_LAYER` and `FILTER_HEAD` are optional, when set `CKPT_PATH`, `FREEZE_LAYER` must be set. `FREEZE_LAYER` should be in ["none", "backbone"], and if you set `FREEZE_LAYER`="backbone", the parameter in backbone will be freezed when training and the parameter in head will not be load from checkpoint. if `FILTER_HEAD`=True, the parameter in head will not be load from checkpoint.

### Launch

```shell
# training example
  python:
      Ascend: python train.py --platform Ascend --config_path [CONFIG_PATH] --dataset_path [TRAIN_DATASET_PATH]
      GPU: python train.py --platform GPU --config_path [CONFIG_PATH] --dataset_path [TRAIN_DATASET_PATH]
      CPU: python train.py --platform CPU --config_path [CONFIG_PATH] --dataset_path [TRAIN_DATASET_PATH]

  shell:
      Ascend: bash run_train.sh Ascend [CONFIG_PATH] [DEVICE_NUM] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)] [DATASET_PATH]
      # example: bash run_train.sh Ascend ../default_config.yaml 8 0,1,2,3,4,5,6,7 /home/DataSet/ImageNet_Original/
      Training on multi cards using hccl: bash run_distributed_train_ascend.sh DATA_PATH RANK_SIZE RANK_TABLE_FILE
      # example: bash scripts/run_distributed_train_ascend.sh /path/dataset 8 /path/hccl.json
      GPU: bash run_train.sh GPU [CONFIG_PATH] 8 0,1,2,3,4,5,6,7 [TRAIN_DATASET_PATH]
      CPU: bash run_train.sh CPU [CONFIG_PATH] [TRAIN_DATASET_PATH]

# finetune whole network example
  python:
      Ascend: python train.py --platform Ascend --config_path [CONFIG_PATH] --dataset_path [TRAIN_DATASET_PATH] --pretrain_ckpt [CKPT_PATH] --freeze_layer none --filter_head True
      GPU: python train.py --platform GPU --config_path [CONFIG_PATH] --dataset_path [TRAIN_DATASET_PATH] --pretrain_ckpt [CKPT_PATH] --freeze_layer none --filter_head True
      CPU: python train.py --platform CPU --config_path [CONFIG_PATH] --dataset_path [TRAIN_DATASET_PATH] --pretrain_ckpt [CKPT_PATH] --freeze_layer none --filter_head True

  shell:
      Ascend: bash run_train.sh Ascend [CONFIG_PATH] [DEVICE_NUM] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)] [DATASET_PATH] [CKPT_PATH] [FREEZE_LAYER] [FILTER_HEAD]
      # example: bash run_train.sh Ascend ../default_config.yaml 8 0,1,2,3,4,5,6,7 /home/DataSet/ImageNet_Original/ /home/model/mobilenetv2/predtrain/mobilenet-200_625.ckpt none True

      GPU: bash run_train.sh GPU --config_path [CONFIG_PATH] 8 0,1,2,3,4,5,6,7 [TRAIN_DATASET_PATH] [CKPT_PATH] none True
      CPU: bash run_train.sh CPU [TRAIN_DATASET_PATH] [CKPT_PATH] none True

# finetune full connected layers example
  python:
      Ascend: python train.py --platform Ascend --config_path ../default_config.yaml --dataset_path [TRAIN_DATASET_PATH]--pretrain_ckpt [CKPT_PATH] --freeze_layer backbone
      GPU: python train.py --platform GPU --config_path ../default_config_gpu.yaml --dataset_path [TRAIN_DATASET_PATH] --pretrain_ckpt [CKPT_PATH] --freeze_layer backbone
      CPU: python train.py --platform CPU --config_path ../default_config_cpu.yaml --dataset_path [TRAIN_DATASET_PATH] --pretrain_ckpt [CKPT_PATH] --freeze_layer backbone

  shell:
      Ascend: bash run_train.sh Ascend [CONFIG_PATH] [DEVICE_NUM] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)] [DATASET_PATH] [CKPT_PATH] [FREEZE_LAYER]
      # example: bash run_train.sh Ascend ../default_config.yaml 8 0,1,2,3,4,5,6,7 /home/DataSet/ImageNet_Original/ /home/model/mobilenetv2/backbone/mobilenet-200_625.ckpt backbone

      GPU: bash run_train.sh GPU [CONFIG_PATH] 8 0,1,2,3,4,5,6,7 [TRAIN_DATASET_PATH] [CKPT_PATH] backbone
      CPU: bash run_train.sh CPU [TRAIN_DATASET_PATH] [CKPT_PATH] backbone
```

### Result

Training result will be stored in the example path. Checkpoints will be stored at `. /checkpoint` by default, and training log  will be redirected to `./train.log` like followings with the platform CPU and GPU.

```log
epoch: [  0/200], step:[  624/  625], loss:[5.258/5.258], time:[140412.236], lr:[0.100]
epoch time: 140522.500, per step time: 224.836, avg loss: 5.258
epoch: [  1/200], step:[  624/  625], loss:[3.917/3.917], time:[138221.250], lr:[0.200]
epoch time: 138331.250, per step time: 221.330, avg loss: 3.917
```

## [Evaluation process](#contents)

### Usage

You can start training using python or shell scripts.If the train method is train or fine tune, should not input the `[CHECKPOINT_PATH]` The usage of shell scripts as follows:

- Ascend: bash run_eval.sh Ascend [CONFIG_PATH] [DATASET_PATH] [CHECKPOINT_PATH]
- GPU: bash run_eval.sh GPU [CONFIG_PATH] [DATASET_PATH] [CHECKPOINT_PATH]
- CPU: bash run_eval.sh CPU [CONFIG_PATH] [DATASET_PATH] [BACKBONE_CKPT_PATH]

### Launch

```shell
# eval example
  python:
      Ascend: python eval.py --platform Ascend --config_path [CONFIG_PATH] --dataset_path [VAL_DATASET_PATH] --pretrain_ckpt ./ckpt_0/mobilenetv2_15.ckpt
      GPU: python eval.py --platform GPU --config_path [CONFIG_PATH] --dataset_path [VAL_DATASET_PATH] --pretrain_ckpt ./ckpt_0/mobilenetv2_15.ckpt
      CPU: python eval.py --platform CPU --config_path [CONFIG_PATH] --dataset_path [VAL_DATASET_PATH] --pretrain_ckpt ./ckpt_0/mobilenetv2_15.ckpt

  shell:
      Ascend: bash run_eval.sh Ascend [CONFIG_PATH] [DATASET_PATH] [CHECKPOINT_PATH]
      # example: bash run_eval.sh Ascend ../default_config.yaml /home/DataSet/ImageNet_Original/ /home/model/mobilenetV2/ckpt/mobilenet-200_625.ckpt

      GPU: bash run_eval.sh GPU [CONFIG_PATH] [VAL_DATASET_PATH] ./checkpoint/mobilenetv2_head_15.ckpt
      CPU: bash run_eval.sh CPU [CONFIG_PATH] [VAL_DATASET_PATH] ./checkpoint/mobilenetv2_head_15.ckpt
```

> checkpoint can be produced in training process.

### Result

Inference result will be stored in the example path, you can find result like the followings in `eval.log`.

```log
result: {'acc': 0.71976314102564111} ckpt=./ckpt_0/mobilenet-200_625.ckpt
```

## [ONNX export and evaluation process](#contents)

### Export

```shell
# export example
  python:
      python export.py --platform [PLATFORM] --ckpt_file ./ckpt_0/mobilenetv2_15.ckpt --file_format ONNX --file_name /path/to/exported.onnx

  shell:
      bash run_export.sh [PLATFORM] ./ckpt_0/mobilenetv2_15.ckpt ONNX /path/to/exported.onnx
```

### Evaluation

```shell
# eval example
  python:
      python eval_onnx.py --platform [PLATFORM] --dataset_path [VAL_DATASET_PATH] --file_name /path/to/exported.onnx

  shell:
      bash run_eval_onnx.sh [PLATFORM] [VAL_DATASET_PATH] /path/to/exported.onnx
```

### Result

Inference result will be stored in the example path, you can find result like the following in `eval_onnx.log`.

```log
accuracy: 0.7199
```

## [Training with dataset on NFS](#contents)

You can use script `run_train_nfs_cache.sh` for running training with a dataset located on Network File System (NFS). By default, a standalone cache server will be started to cache all images in tensor format in memory to improve performance.

Please refer to [Training Process](#training-process) for the usage of this shell script.

```shell
# training with NFS dataset example
Ascend: bash run_train_nfs_cache.sh Ascend 8 0,1,2,3,4,5,6,7 hccl_config.json [TRAIN_DATASET_PATH]
GPU: bash run_train_nfs_cache.sh GPU 8 0,1,2,3,4,5,6,7 [TRAIN_DATASET_PATH]
CPU: bash run_train_nfs_cache.sh CPU [TRAIN_DATASET_PATH]
```

> With cache enabled, a standalone cache server will be started in the background to cache the dataset in memory. However, Please make sure the dataset fits in memory (around 120GB of memory is required for caching ImageNet train dataset).
> Users can choose to shutdown the cache server after training or leave it alone for future usage.

## [Inference process](#contents)

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

### Export MindIR

```shell
python export.py --platform [PLATFORM] --ckpt_file [CKPT_PATH] --file_format [EXPORT_FORMAT]
```

The ckpt_file parameter is required,
`EXPORT_FORMAT` should be in ["AIR", "MINDIR"]

### Infer

Before performing inference, the mindir file must be exported by `export.py` script. We only provide an example of inference using MINDIR model.
Current batch_size can only be set to 1.

```shell
bash run_infer_cpp.sh [MINDIR_PATH] [DATA_PATH] [LABEL_PATH] [DVPP] [DEVICE_TYPE] [DEVICE_ID]
```

- `LABEL_PATH` label.txt path. Write a py script to sort the category under the dataset, map the file names under the categories and category sort values,Such as[file name : sort value], and write the mapping results to the labe.txt file.
- `DVPP` is mandatory, and must choose from ["DVPP", "CPU"], it's case-insensitive.The size of the picture that MobilenetV2 performs inference is [224, 224], the DVPP hardware limits the width of divisible by 16, and the height is divisible by 2. The network conforms to the standard, and the network can pre-process the image through DVPP.
- `DEVICE_ID` is optional, default value is 0.

### result

Inference result is saved in current path, you can find result like this in acc.log file.

```bash
'Accuracy': 0.71654
```

# Apply algorithm in MindSpore Golden Stick

MindSpore Golden Stick is a compression algorithm set for MindSpore. We usually apply algorithm in Golden Stick before training for smaller model size, lower power consuming or faster inference process. MindSpore Golden Stick provides SimQAT algorithm for Mobilenetv2. SimQAT is a quantization-aware training algorithm that trains the quantization parameters of certain layers in the network by introducing fake-quantization nodes, so that the model can perform inference with less power consumption or higher performance during the deployment phase.

## Training Process

### Running on GPU

```text
# distributed training
cd ./golden_stick/scripts/
# PYTHON_PATH represents path to directory of 'train.py'.
bash run_distribute_train_gpu.sh [PYTHON_PATH] [CONFIG_FILE] [DEVICE_NUM] [DATASET_PATH] [CKPT_TYPE](optional) [CKPT_PATH](optional)

# distributed training example, apply SimQAT and train from beginning
cd ./golden_stick/scripts/
bash run_distribute_train_gpu.sh ../quantization/simqat/ ../quantization/simqat/ 4 mobilenetv2_cifar10_config.yaml /path/to/dataset

# distributed training example, apply SimQAT and train from full precision checkpoint
cd ./golden_stick/scripts/
bash run_distribute_train_gpu.sh ../quantization/simqat/ ../quantization/simqat/ 4 mobilenetv2_cifar10_config.yaml /path/to/dataset FP32 /path/to/fp32_ckpt

# distributed training example, apply SimQAT and train from pretrained checkpoint
cd ./golden_stick/scripts/
bash run_distribute_train_gpu.sh ../quantization/simqat/ ../quantization/simqat/ 4 mobilenetv2_cifar10_config.yaml /path/to/dataset PRETRAINED /path/to/pretrained_ckpt

# standalone training
cd ./golden_stick/scripts/
# PYTHON_PATH represents path to directory of 'train.py'.
bash run_standalone_train_gpu.sh [PYTHON_PATH] [CONFIG_FILE] [DATASET_PATH] [CKPT_TYPE](optional) [CKPT_PATH](optional)

# standalone training example, apply SimQAT and train from beginning
cd ./golden_stick/scripts/
bash run_standalone_train_gpu.sh ../quantization/simqat/ ../quantization/simqat/mobilenetv2_cifar10_config.yaml /path/to/dataset

# standalone training example, apply SimQAT and train from full precision checkpoint
cd ./golden_stick/scripts/
bash run_standalone_train_gpu.sh ../quantization/simqat/ ../quantization/simqat/mobilenetv2_cifar10_config.yaml /path/to/dataset FP32 /path/to/fp32_ckpt

# standalone training example, apply SimQAT and train from pretrained checkpoint
cd ./golden_stick/scripts/
bash run_standalone_train_gpu.sh ../quantization/simqat/ ../quantization/simqat/mobilenetv2_cifar10_config.yaml /path/to/dataset PRETRAINED /path/to/pretrained_ckpt
```

## Evaluation Process

### Running on GPU

```text
# evaluation
cd ./golden_stick/scripts/
# PYTHON_PATH represents path to directory of 'eval.py'.
bash run_eval_gpu.sh [PYTHON_PATH] [CONFIG_FILE] [DATASET_PATH] [CHECKPOINT_PATH]
```

```text
# evaluation example
cd ./golden_stick/scripts/
bash run_eval_gpu.sh ../quantization/simqat/ ../quantization/simqat/mobilenetv2_cifar10_config.yaml /path/to/dataset /path/to/ckpt
```

### Result

Evaluation result will be stored in the example path, whose folder name is "eval". Under this, you can find result like the following in log.

- Apply SimQAT on Mobilenetv2, and evaluating with CIFAR-10 dataset:

```text
result:{'accuracy': 0.9356, ckpt=~/mobilenetv2/train_parallel0/mobilenetv2-200_166.ckpt}
```

# [Model description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 | MobilenetV2                                                |                           |
| -------------------------- | ---------------------------------------------------------- | ------------------------- |
| Model Version              | V1                                                         | V1                        |
| Resource                   | Ascend 910; cpu 2.60GHz, 192cores; memory 755G; OS Euler2.8              | NV SMX2 V100-32G          |
| uploaded Date              | 07/05/2021                                                 | 07/05/2021                |
| MindSpore Version          | 1.3.0                                                      | 1.3.0                     |
| Dataset                    | ImageNet                                                   | ImageNet                  |
| Training Parameters        | src/config.py                                              | src/config.py             |
| Optimizer                  | Momentum                                                   | Momentum                  |
| Loss Function              | SoftmaxCrossEntropy                                        | SoftmaxCrossEntropy       |
| outputs                    | probability                                                | probability               |
| Loss                       | 1.908                                                      | 1.913                     |
| Accuracy                   | ACC1[71.78%]                                               | ACC1[71.08%] |
| Total time                 | 753 min                                                    | 845 min                   |
| Params (M)                 | 3.3 M                                                      | 3.3 M                     |
| Checkpoint for Fine tuning | 27.3 M                                                     | 27.3 M                    |
| Scripts                    | [Link](https://gitee.com/mindspore/models/tree/master/official/cv/MobileNet/mobilenetv2)|

### Inference Performance

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | MobilenetV2                 |
| Resource            | Ascend 310; CentOS 3.10     |
| Uploaded Date       | 11/05/2021 (month/day/year) |
| MindSpore Version   | 1.2.0                       |
| Dataset             | ImageNet                    |
| batch_size          | 1                           |
| outputs             | Accuracy                    |
| Accuracy            | Accuracy=0.71654            |
| Model for inference | 27.3M(.ckpt file)           |

# [Description of Random Situation](#contents)

<!-- In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py. -->
In train.py, we set the seed which is used by numpy.random, mindspore.common.Initializer, mindspore.ops.composite.random_ops and mindspore.nn.probability.distribution.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
