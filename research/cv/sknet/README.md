# Contents

- [Contents](#contents)
- [SK-Net Description](#sk-net-description)
- [Description](#description)
- [Paper](#paper)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Features](#features)
- [Mixed Precision](#mixed-precision)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
- [Script and Sample Code](#script-and-sample-code)
- [Script Parameters](#script-parameters)
- [Training Process](#training-process)
- [Usage](#usage)
- [Running on Ascend](#running-on-ascend)
- [Result](#result)
- [Evaluation Process](#evaluation-process)
- [Usage](#usage-1)
- [Running on Ascend](#running-on-ascend-1)
- [Result](#result-1)
- [Inference Process](#inference-process)
- [Export MindIR](#export-mindir)
- [Infer on Ascend310](#infer-on-ascend310)
- [Result](#result-2)
- [Model Description](#model-description)
- [Performance](#performance)
- [Evaluation Performance](#evaluation-performance)
- [SKNet50 on CIFRA10](#sknet50-on-cifra10)
- [Inference Performance](#inference-performance)
- [SKNet50 on CIFAR10](#sknet50-on-cifar10)
- [310 Inference Performance](#310-inference-performance)
- [SKNet50 on CIFAR10](#sknet50-on-cifar10-1)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [SK-Net Description](#contents)

## Description

  Selective Kernel Networks is inspired by cortical neurons that can dynamically adjust their own receptive field according to different stimuli. It is a product of combining SE operator, Merge-and-Run Mappings, and attention on inception block ideas. Carry out Selective Kernel transformation for all convolution kernels> 1 to make full use of the smaller theory brought by group/depthwise convolution, so that the design of adding multiple channels and dynamic selection will not bring a big overhead

  this is example of training SKNET50 with CIFAR-10 dataset in MindSpore. Training SKNet50 for just 90 epochs using 8 Ascend 910, we can reach top-1 accuracy of 94.49% on CIFAR10.

## Paper

[paper](https://arxiv.org/abs/1903.06586): Xiang Li, Wenhai Wang, Xiaolin Hu, Jian Yang. "Selective Kernel Networks"

# [Model Architecture](#contents)

The overall network architecture of Net is show below:
[Link](https://arxiv.org/pdf/1903.06586.pdf)

# [Dataset](#contents)

Dataset used: [CIFAR10](https://www.kaggle.com/c/cifar-10)

- Dataset size 32*32 colorful images in 10 classes
    - Train：50000 images  
    - Test： 10000 images
- Data format：binary files
    - Note：Data will be processed in dataset.py
- Download the dataset, the directory structure is as follows:

```bash
├─cifar-10-batches-bin
│
└─cifar-10-verify-bin
```

# [Features](#contents)

## Mixed Precision

The [mixed precision](https://www.mindspore.cn/tutorials/en/master/advanced/mixed_precision.html) training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data types, and maintains the network precision achieved by the single-precision training at the same time. Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware.
For FP16 operators, if the input data type is FP32, the backend of MindSpore will automatically handle it with reduced precision. Users could check the reduced-precision operators by enabling INFO log and then searching ‘reduce precision’.

# [Environment Requirements](#contents)

- Hardware（Ascend）
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- Running on Ascend

```bash
# standalone training
python train.py --data_url=/data/cifar10 --device_id=Ascend --device_id=0

# run evaluation

## just test one ckpt
python eval.py --checkpoint_path=/resnet/sknet_90.ckpt --data_url=/data/cifar10 --device_id=Ascend --device_id=0
## test all .ckpt in dir
python eval.py --checkpoint_path=/resnet --data_url=/data/cifar10 --device_id=Ascend --device_id=0
```

- Running on GPU

```bash
# standalone training
python train.py --data_url=/data/cifar10 --device_id=GPU --device_id=0

# run evaluation

## just test one ckpt
python eval.py --checkpoint_path=/resnet/sknet_90.ckpt --data_url=/data/cifar10 --device_id=GPU --device_id=0
## test all .ckpt in dir
python eval.py --checkpoint_path=/resnet --data_url=/data/cifar10 --device_id=GPU --device_id=0
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```text
└──SK-Net
  ├── README.md
  ├── ascend310_infer
    ├── inc
    ├── src
    ├── build.sh                         # make process  
    ├── CMakeLists.txt                   # cmake configuration  
  ├── scripts
    ├── run_distribute_train.sh            # launch ascend distributed training(8 pcs)
    ├── run_eval.sh                        # launch ascend evaluation
    ├── run_standalone_train.sh            # launch ascend standalone training(1 pcs)
    ├── run_infer_310.sh                   # launch 310 infer  
  ├── src
    ├── config.py                          # parameter configuration
    ├── CrossEntropySmooth.py              # loss definition
    ├── dataset.py                         # data preprocessing
    ├── lr_generator.py                    # generate learning rate for each step
    ├── sknet50.py                         # sket50 backbone
    ├── var_init.py                        # convlution init function
    └── util.py                            # group convlution
  ├── export.py                            # export model for inference
  ├── eval.py                              # eval net
  └── train.py                             # train net
  ├── preprocess.py                        # preprocess scripts
  ├── postprocess.py                       # postprocess scripts
```

## [Script Parameters](#contents)

Parameters for both training and evaluation can be set in config.py.

- Config for SKNET50, CIFAR10 dataset

```bash
"class_num": 10,                # dataset class number
"batch_size": 32,                 # batch size of input tensor
"loss_scale": 1024,               # loss scale
"momentum": 0.9,                  # momentum optimizer
"weight_decay": 1e-4,             # weight decay
"epoch_size": 90,                 # only valid for taining, which is always 1 for inference
"pretrain_epoch_size": 0,         # epoch size that model has been trained before loading pretrained checkpoint, actual training epoch size is equal to epoch_size minus pretrain_epoch_size
"save_checkpoint": True,          # whether save checkpoint or not
"save_checkpoint_epochs": 5,      # the epoch interval between two checkpoints. By default, the last checkpoint will be saved after the last epoch
"keep_checkpoint_max": 10,        # only keep the last keep_checkpoint_max checkpoint
"save_checkpoint_path": "./ckpt",     # path to save checkpoint relative to the executed path
"warmup_epochs": 5,               # number of warmup epoch
"lr_decay_mode": "ploy",        # decay mode for generating learning rate
"lr_init": 0.01,                     # initial learning rate
"lr_max": 0.00001,                    # maximum learning rate
"lr_end": 0.1,                    # minimum learning rate
```

## [Training Process](#contents)

### Usage

#### Running on Ascend

```bash
# distributed training
Usage:
bash run_distribute_train_ascend.sh [DATA_URL] [RANK_TABLE_FILE] [DEVICE_NUM]

# standalone training
Usage:
bash run_standalone_train_ascend.sh [DATA_URL] [DEVICE_ID]
```

For distributed training, a hccl configuration file with JSON format needs to be created in advance.

Please follow the instructions in the link [hccn_tools](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools).

Training result will be stored in the example path, whose folder name begins with "train" or "train_parallel". Under this, you can find checkpoint file together with result like the following in log.

#### Running on GPU

```bash
# distributed training
Usage:
bash run_distribute_train_gpu.sh [DATA_URL] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)] [DEVICE_NUM]

# standalone training
Usage:
bash run_standalone_train_gpu.sh [DATA_URL] [DEVICE_ID]
```

## [Evaluation Process](#contents)

### Usage

#### Running on Ascend

```bash
bash run_eval_ascend.sh [DATA_URL]  [CHECKPOINT_PATH / CHECKPOINT_DIR] [DEVICE_ID]
```

#### Running on GPU

```bash
bash run_eval_gpu.sh [DATA_URL]  [CHECKPOINT_PATH / CHECKPOINT_DIR] [DEVICE_ID]
```

### Result

- Evaluating SKNet50 with CIFAR10 dataset

#### Result on Ascend

```text
result: {'top_1_accuracy': 0.9493189102564102, 'top_5_accuracy': 0.9982972756410257, 'loss': 0.26100944656317526} ckpt= ./scripts/train_parallel0/ckpt_0/sknet-87_390.ckpt
```

#### Result on GPU

```text
result: {'top_1_accuracy': 0.9525240384615384, 'top_5_accuracy': 0.9987980769230769, 'loss': 0.24511159359111354} ckpt= ./train/ckpt_1/sknet-73_390.ckpt
```

## [Inference Process](#contents)

### Export MindIR

```bash
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

The ckpt_file parameter is required,
`FILE_NAME` is the name of the AIR/ONNX/MINDIR file.
`FILE_FORMAT` should be in ["AIR","ONNX", "MINDIR"]

### Infer on Ascend310

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

Before performing inference, the mindir file must be exported by `export.py` script. We only provide an example of inference using MINDIR model.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATASET_NAME] [DATASET_PATH] [NEED PREPROCESS] [DEVICE_ID]
```

- DATASET_NAME can choose from ['cifar10', 'imagenet2012'].
- NEED_PREPROCESS means weather need preprocess or not, it's value is 'y' or 'n'.
- DEVICE_ID is optional, it can be set by environment variable device_id, otherwise the value is zero"
- `DVPP` is mandatory, and must choose from ["DVPP", "CPU"], it's case-insensitive. SE-net only support CPU mode.
- `DEVICE_ID` is optional, default value is 0.

### Result

Inference result is saved in current path, you can find result like this in acc.log file.

```bash
result: {'top_1_accuracy': 0.9449118589743589}
```

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

#### SKNet50 on CIFRA10

| Parameters                 | Ascend 910| A100|
| -------------------------- | --------------------------------------| ------ |
| Model Version              | SKNet50                                               | Sknet50|
| Resource                   | CentOs 8.2, Ascend 910，CPU 2.60GHz 96cores，Memory 1028G  | Ubuntu18.04, A100, CPU 2.4Ghz, 64cores, Memory 384G|
| uploaded Date              | 27/11/2021 (month/day/year)                   | 27/11/2021 (month/day/year)        |
| MindSpore Version          | 1.3.0(ModelArts)                                                 |1.5.0|
| Dataset                    | CIFAR10                                                |CIFAR10|
| Training Parameters        | epoch=90, steps per epoch=390, batch_size = 32             |epoch=90, steps per epoch=390, batch_size = 32|
| Optimizer                  | Momentum              |Momentum            |
| Loss Function              | Softmax Cross Entropy       |Softmax Cross Entropy   |
| outputs                    | probability                 |probability |
| Loss                       | 0.261      | 0.244|
| Speed                      |  95.360 ms/step（4pcs）    |133.638 ms/step（4pcs） |
| Total time                 | 1h11min    |1h42min|
| Parameters (M)             | 27.5M|      27.5M|
| Checkpoint for Fine tuning | 194.96M(.ckpt file)     |194.96M(.ckpt file)|
| Scripts                    | [Link](https://gitee.com/mindspore/models/tree/master/research/cv/sknet) |[Link](https://gitee.com/mindspore/models/tree/master/research/cv/sknet)|

### Inference Performance

#### SKNet50 on CIFAR10

| Parameters          | Ascend                      | GPU |
| ------------------- | --------------------------- | ------ |
| Model Version       | SKNet50                 |SKNet50      |
| Resource            | Ascend 910                  | A100 |
| Uploaded Date       | 27/11/2021 (month/day/year)  |27/11/2021 (month/day/year)        |
| MindSpore Version   | 1.3.0                 |1.5.0                 |
| Dataset             | CIFAR10                |CIFAR10                |
| batch_size          | 32(4p)                          |32(4p)     |
| Accuracy            | 95.02%                 |95.26%|

### 310 Inference Performance

#### SKNet50 on CIFAR10

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | SKNet50                 |
| Resource            | Ascend 310                  |
| Uploaded Date       | 09/23/2021 (month/day/year) |
| MindSpore Version   | 1.3.0                 |
| Dataset             | CIFAR10                |
| batch_size          | 32                          |
| Accuracy            | 95.49%                      |

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside "create_dataset" function. We also use random seed in train.py.

# [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/models).
