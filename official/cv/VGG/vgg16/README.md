# Contents

- [Contents](#contents)
    - [VGG Description](#vgg-description)
    - [Model Architecture](#model-architecture)
    - [Dataset](#dataset)
        - [Dataset used: CIFAR-10](#dataset-used-cifar-10)
        - [Dataset used: ImageNet2012](#dataset-used-imagenet2012)
        - [Dataset used: Custom Dataset](#dataset-used-custom-dataset)
            - [Dataset organize way](#dataset-organize-way)
    - [Features](#features)
        - [Mixed Precision](#mixed-precision)
    - [Environment Requirements](#environment-requirements)
    - [Quick Start](#quick-start)
    - [Script Description](#script-description)
        - [Script and Sample Code](#script-and-sample-code)
        - [Script Parameters](#script-parameters)
            - [Training](#training)
            - [Evaluation](#evaluation)
        - [Parameter configuration](#parameter-configuration)
        - [Training Process](#training-process)
            - [Training](#training-1)
                - [Run vgg16 on Ascend](#run-vgg16-on-ascend)
                - [Run vgg16 on GPU](#run-vgg16-on-gpu)
        - [Evaluation Process](#evaluation-process)
            - [Evaluation](#evaluation-1)
            - [ONNX Evaluation](#onnx-evaluation)
    - [Migration process](#migration-process)
        - [Dataset split](#dataset-split)
        - [Migration](#migration)
        - [Eval](#eval)
        - [Model quick start](#model-quick-start)
    - [Inference Process](#inference-process)
        - [Export MindIR](#export-mindir)
        - [Infer](#infer)
        - [result](#result)
    - [Model Description](#model-description)
        - [Performance](#performance)
            - [Training Performance](#training-performance)
            - [Evaluation Performance](#evaluation-performance)
    - [Description of Random Situation](#description-of-random-situation)
    - [ModelZoo Homepage](#modelzoo-homepage)

## [VGG Description](#contents)

VGG, a very deep convolutional networks for large-scale image recognition, was proposed in 2014 and won the 1th place in object localization and 2th place in image classification task in ImageNet Large-Scale Visual Recognition Challenge 2014 (ILSVRC14).

[Paper](https://arxiv.org/abs/1409.1556): Simonyan K, zisserman A. Very Deep Convolutional Networks for Large-Scale Image Recognition[J]. arXiv preprint arXiv:1409.1556, 2014.

## [Model Architecture](#contents)

VGG 16 network is mainly consisted by several basic modules (including convolution and pooling layer) and three continuous Dense layer.
here basic modules mainly include basic operation like:  **3×3 conv** and **2×2 max pooling**.

## [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

### Dataset used: [CIFAR-10](<http://www.cs.toronto.edu/~kriz/cifar.html>)

- CIFAR-10 Dataset size：175M，60,000 32*32 colorful images in 10 classes
    - Train：146M，50,000 images
    - Test：29.3M，10,000 images
    - Data format: binary files
    - Note: Data will be processed in src/dataset.py

### Dataset used: [ImageNet2012](http://www.image-net.org/)

- Dataset size: ~146G, 1.28 million colorful images in 1000 classes
    - Train: 140G, 1,281,167 images
    - Test: 6.4G, 50, 000 images
    - Data format: RGB images
    - Note: Data will be processed in src/dataset.py

### Dataset used: Custom Dataset

- Data format: RGB images
    - Note: Data will be processed in src/data_split.py,Used to divide training and validation sets.

#### Dataset organize way

  CIFAR-10

  > Unzip the CIFAR-10 dataset to any path you want and the folder structure should be as follows:
  >
  > ```bash
  > .
  > ├── cifar-10-batches-bin  # train dataset
  > └── cifar-10-verify-bin   # infer dataset
  > ```

  ImageNet2012

  > Unzip the ImageNet2012 dataset to any path you want and the folder should include train and eval dataset as follows:
  >
  > ```bash
  > .
  > └─dataset
  >   ├─ilsvrc                # train dataset
  >   └─validation_preprocess # evaluate dataset
  > ```

  Custom Dataset

  > Unzip the custom dataset to any path, the folder structure should contain the folder with the class name and all the pictures under this folder, as shown below:
  >
  > ```bash
  > .
  > └─dataset
  >   ├─class_name1                # class name
  >     ├─xx.jpg                    # All images corresponding to the class name
  >     ├─ ...
  >     ├─xx.jpg
  >   ├─class_name2
  >   ├─  ...
  > ```

## [Features](#contents)

### Mixed Precision

The [mixed precision](https://www.mindspore.cn/tutorials/en/master/advanced/mixed_precision.html) training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data formats, and maintains the network precision achieved by the single-precision training at the same time. Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware.

For FP16 operators, if the input data type is FP32, the backend of MindSpore will automatically handle it with reduced precision. Users could check the reduced-precision operators by enabling INFO log and then searching ‘reduce precision’.

## [Environment Requirements](#contents)

- Hardware（Ascend/GPU）
    - Prepare hardware environment with Ascend or GPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

## [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- Running on Ascend

```python
# run training example
python train.py --config_path=[YAML_CONFIG_PATH] --data_dir=[DATA_PATH] --dataset=[DATASET_TYPE] > output.train.log 2>&1 &

# run distributed training example
bash scripts/run_distribute_train.sh [RANL_TABLE_JSON] [DATA_PATH] [DATASET_TYPE](optional)

# run evaluation example
python eval.py --config_path=[YAML_CONFIG_PATH] --data_dir=[DATA_PATH]  --pre_trained=[PRE_TRAINED] --dataset=[DATASET_TYPE] > output.eval.log 2>&1 &
```

For distributed training, a hccl configuration file with JSON format needs to be created in advance.
Please follow the instructions in the link below:
<https://gitee.com/mindspore/models/tree/r2.0/utils/hccl_tools>

- Running on GPU

```bash
# run training example
python train.py --config_path=[YAML_CONFIG_PATH] --device_target="GPU" --dataset=[DATASET_TYPE] --data_dir=[DATA_PATH] > output.train.log 2>&1 &

# run distributed training example
bash scripts/run_distribute_train_gpu.sh [DATA_PATH] --dataset=[DATASET_TYPE]

# run evaluation example
python eval.py --config_path=[YAML_CONFIG_PATH] --device_target="GPU" --dataset=[DATASET_TYPE] --data_dir=[DATA_PATH]  --pre_trained=[PRE_TRAINED] > output.eval.log 2>&1 &
```

- Running on CPU

```python

# run dataset processing example
python src/data_split.py --split_path [SPLIT_PATH]

# run finetune example
python tine_tune.py --config_path [YAML_CONFIG_PATH]

# run eval example
python eval.py --config_path [YAML_CONFIG_PATH]

# quick start
python quick_start.py --config_path [YAML_CONFIG_PATH]
```

- Running on [ModelArts](https://support.huaweicloud.com/modelarts/)

```bash
# Train Cifar10 1p on ModelArts
# (1) Add "config_path=/path_to_code/cifar10_config.yaml" on the website UI interface.
# (2) Perform a or b.
#       a. Set "enable_modelarts=True" on cifar10_config.yaml file.
#          Set "data_dir='/cache/data/cifar10'" on cifar10_config.yaml file.
#          Set "is_distributed=0" on cifar10_config.yaml file.
#          Set "dataset='cifar10'" on cifar10_config.yaml file.
#          Set other parameters on cifar10_config.yaml file you need.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add "data_dir=/cache/data/cifar10" on the website UI interface.
#          Add "is_distributed=0" on the website UI interface.
#          Add "dataset=cifar10" on the website UI interface.
#          Add other parameters on the website UI interface.
# (3) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
# (4) Set the code directory to "/path/vgg16" on the website UI interface.
# (5) Set the startup file to "train.py" on the website UI interface.
# (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
# (7) Create your job.
#
# Train Cifar10 8p on ModelArts
# (1) Add "config_path=/path_to_code/cifar10_config.yaml" on the website UI interface.
# (2) Perform a or b.
#       a. Set "enable_modelarts=True" on cifar10_config.yaml file.
#          Set "data_dir='/cache/data/cifar10'" on cifar10_config.yaml file.
#          Set "is_distributed=1" on cifar10_config.yaml file.
#          Set "dataset='cifar10'" on cifar10_config.yaml file.
#          Set other parameters on cifar10_config.yaml file you need.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add "data_dir=/cache/data/cifar10" on the website UI interface.
#          Add "is_distributed=1" on the website UI interface.
#          Add "dataset=cifar10" on the website UI interface.
#          Add other parameters on the website UI interface.
# (3) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
# (4) Set the code directory to "/path/vgg16" on the website UI interface.
# (5) Set the startup file to "train.py" on the website UI interface.
# (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
# (7) Create your job.
#
# Train Imagenet 8p on ModelArts
# (1) Add "config_path=/path_to_code/imagenet2012_config.yaml" on the website UI interface.
# (2) Perform a or b.
#       a. Set "enable_modelarts=True" on imagenet2012_config.yaml file.
#          Set "data_dir='/cache/data/ImageNet/train'" on imagenet2012_config.yaml file.
#          Set "is_distributed=1" on imagenet2012_config.yaml file.
#          Set "dataset='imagenet2012'" on imagenet2012_config.yaml file.
#          Set other parameters on imagenet2012_config.yaml file you need.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add "data_dir=/cache/data/ImageNet/train" on the website UI interface.
#          Add "is_distributed=1" on the website UI interface.
#          Add "dataset=imagenet2012" on the website UI interface.
#          Add other parameters on the website UI interface.
# (3) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
# (4) Set the code directory to "/path/vgg16" on the website UI interface.
# (5) Set the startup file to "train.py" on the website UI interface.
# (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
# (7) Create your job.
#
# Eval Cifar10 1p on ModelArts
# (1) Add "config_path=/path_to_code/cifar10_config.yaml" on the website UI interface.
# (2) Perform a or b.
#       a. Set "enable_modelarts=True" on cifar10_config.yaml file.
#          Set "data_dir='/cache/data/cifar10'" on cifar10_config.yaml file.
#          Set "dataset='cifar10'" on cifar10_config.yaml file.
#          Set "checkpoint_url='s3://dir_to_your_trained_model/'" on cifar10_config.yaml file.
#          Set "pre_trained='/cache/checkpoint_path/model.ckpt'" on cifar10_config.yaml file.
#          Set other parameters on cifar10_config.yaml file you need.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add "data_dir=/cache/data/cifar10" on the website UI interface.
#          Add "dataset=cifar10" on the website UI interface.
#          Add "checkpoint_url=s3://dir_to_your_trained_model/" on the website UI interface.
#          Add "pre_trained=/cache/checkpoint_path/model.ckpt" on the website UI interface.
#          Add other parameters on the website UI interface.
# (3) Upload or copy your pretrained model to S3 bucket.
# (4) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
# (5) Set the code directory to "/path/vgg16" on the website UI interface.
# (6) Set the startup file to "eval.py" on the website UI interface.
# (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
# (8) Create your job.
#
# Eval ImageNet 1p on ModelArts
# (1) Add "config_path=/path_to_code/imagenet2012_config.yaml" on the website UI interface.
# (2) Perform a or b.
#       a. Set "enable_modelarts=True" on imagenet2012_config.yaml file.
#          Set "data_dir='/cache/data/ImageNet/validation_preprocess'" on imagenet2012_config.yaml file.
#          Set "dataset='imagenet2012'" on imagenet2012_config.yaml file.
#          Set "checkpoint_url='s3://dir_to_your_trained_model/'" on imagenet2012_config.yaml file.
#          Set "pre_trained='/cache/checkpoint_path/model.ckpt'" on imagenet2012_config.yaml file.
#          Set other parameters on imagenet2012_config.yaml file you need.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add "data_dir=/cache/data/ImageNet/validation_preprocess" on the website UI interface.
#          Add "dataset=imagenet2012" on the website UI interface.
#          Add "checkpoint_url=s3://dir_to_your_trained_model/" on the website UI interface.
#          Add "pre_trained=/cache/checkpoint_path/model.ckpt" on the website UI interface.
#          Add other parameters on the website UI interface.
# (3) Upload or copy your trained model to S3 bucket.
# (4) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
# (5) Set the code directory to "/path/vgg16" on the website UI interface.
# (6) Set the startup file to "eval.py" on the website UI interface.
# (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
# (8) Create your job.
#
# Export 1p on ModelArts
# (1) Add "config_path=/path_to_code/imagenet2012_config.yaml" on the website UI interface.
# (2) Perform a or b.
#       a. Set "enable_modelarts=True" on imagenet2012_config.yaml file.
#          Set "file_name='vgg16'" on imagenet2012_config.yaml file.
#          Set "file_format='MINDIR'" on imagenet2012_config.yaml file.
#          Set "checkpoint_url='s3://dir_to_your_trained_model/'" on imagenet2012_config.yaml file.
#          Set "ckpt_file='/cache/checkpoint_path/model.ckpt'" on imagenet2012_config.yaml file.
#          Set other parameters on imagenet2012_config.yaml file you need.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add "file_name=vgg16" on the website UI interface.
#          Add "file_format='MINDIR'" on the website UI interface.
#          Add "checkpoint_url=s3://dir_to_your_trained_model/" on the website UI interface.
#          Add "ckpt_file=/cache/checkpoint_path/model.ckpt" on the website UI interface.
#          Add other parameters on the website UI interface.
# (3) Upload or copy your trained model to S3 bucket.
# (4) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
# (5) Set the code directory to "/path/vgg16" on the website UI interface.
# (6) Set the startup file to "export.py" on the website UI interface.
# (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
# (8) Create your job.
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```bash
├── models
    ├── README.md                                 // descriptions about all the models
    ├── official/cv/vgg16
        ├── README.md                             // descriptions about vgg
        ├── README_CN.md                          // descriptions about vgg with Chinese
        ├── ascend310_infer                       // Ascend310 infer folder
        ├── infer                                 // MindX infer folder
        ├── model_utils
        │   ├── __init__.py                       // init file
        │   ├── config.py                         // Parse arguments
        │   ├── device_adapter.py                 // Device adapter for ModelArts
        │   ├── local_adapter.py                  // Local adapter
        │   ├── moxing_adapter.py                 // Moxing adapter for ModelArts
        ├── scripts
        │   ├── run_distribute_train.sh           // shell script for distributed training on Ascend
        │   ├── run_distribute_train_gpu.sh       // shell script for distributed training on GPU
        │   ├── run_eval.sh                       // shell script for eval on Ascend
        │   ├── run_infer_310.sh                  // shell script for infer on Ascend 310
        │   ├── run_onnx_eval.sh                  // shell script for ONNX eval
        ├── src
        │   ├── utils
        │   │   ├── logging.py                    // logging format setting
        │   │   ├── sampler.py                    // create sampler for dataset
        │   │   ├── util.py                       // util function
        │   │   ├── var_init.py                   // network parameter init method
        │   ├── crossentropy.py                   // loss calculation
        │   ├── dataset.py                        // creating dataset
        │   ├── data_split.py                     // CPU dataset split script
        │   ├── linear_warmup.py                  // linear leanring rate
        │   ├── warmup_cosine_annealing_lr.py     // consine anealing learning rate
        │   ├── warmup_step_lr.py                 // step or multi step learning rate
        │   ├──vgg.py                             // vgg architecture
        ├── train.py                              // training script
        ├── eval.py                               // evaluation script
        ├── eval_onnx.py                          // ONNX evaluation script
        ├── finetune.py                           // CPU transfer script
        ├── quick_start.py                        // CPU quick start script
        ├── postprocess.py                        // postprocess script
        ├── preprocess.py                         // preprocess script
        ├── mindspore_hub_conf.py                 // mindspore_hub_conf script
        ├── cifar10_config.yaml                   // Configurations for cifar10
        ├── imagenet2012_config.yaml              // Configurations for imagenet2012
        ├── cpu_config.yaml                       // Configurations for CPU transfer
        ├── export.py                             // model convert script
        └── requirements.txt                      // requirements
```

### [Script Parameters](#contents)

#### Training

```bash
usage: train.py [--config_path YAML_CONFIG_PATH]
                [--device_target TARGET][--data_dir DATA_PATH]
                [--dataset  DATASET_TYPE][--is_distributed VALUE]
                [--pre_trained PRE_TRAINED]
                [--ckpt_path CHECKPOINT_PATH][--ckpt_interval INTERVAL_STEP]

parameters/options:
  --config_path         the storage path of YAML_CONFIG_FILE
  --device_target       the training backend type, Ascend or GPU, default is Ascend.
  --dataset             the dataset type, cifar10 or imagenet2012.
  --is_distributed      the  way of traing, whether do distribute traing, value can be 0 or 1.
  --data_dir            the storage path of dataset
  --pre_trained         the pretrained checkpoint file path.
  --ckpt_path           the path to save checkpoint.
  --ckpt_interval       the epoch interval for saving checkpoint.

```

#### Evaluation

```bash
usage: eval.py [--config_path YAML_CONFIG_PATH]
               [--device_target TARGET][--data_dir DATA_PATH]
               [--dataset  DATASET_TYPE][--pre_trained PRE_TRAINED]

parameters/options:
  --config_path         the storage path of YAML_CONFIG_FILE
  --device_target       the evaluation backend type, Ascend or GPU, default is Ascend.
  --dataset             the dataset type, cifar10 or imagenet2012.
  --data_dir            the storage path of dataset.
  --pre_trained         the checkpoint file path used to evaluate model.
```

### [Parameter configuration](#contents)

Parameters for both training and evaluation can be set in cifar10_config.yaml/imagenet2012_config.yaml.

- config for vgg16, CIFAR-10 dataset

```bash
num_classes: 10                      # dataset class num
lr: 0.01                             # learning rate
lr_init: 0.01                        # initial learning rate
lr_max: 0.1                          # max learning rate
lr_epochs: '30,60,90,120'            # lr changing based epochs
lr_scheduler: "step"                 # learning rate mode
warmup_epochs: 5                     # number of warmup epoch
batch_size: 64                       # batch size of input tensor
max_epoch: 70                        # only valid for taining, which is always 1 for inference
momentum: 0.9                        # momentum
weight_decay: 0.0005                 # weight decay
loss_scale: 1.0                      # loss scale
label_smooth: 0                      # label smooth
label_smooth_factor: 0               # label smooth factor
buffer_size: 10                      # shuffle buffer size
image_size: [224,224]                # image size
pad_mode: 'same'                     # pad mode for conv2d
padding: 0                           # padding value for conv2d
has_bias: False                      # whether has bias in conv2d
batch_norm: True                     # whether has batch_norm in conv2d
keep_checkpoint_max: 10              # only keep the last keep_checkpoint_max checkpoint
initialize_mode: "XavierUniform"     # conv2d init mode
has_dropout: True                    # whether using Dropout layer
```

- config for vgg16, ImageNet2012 dataset

```bash
num_classes: 1000                   # dataset class num
lr: 0.01                            # learning rate
lr_init: 0.01                       # initial learning rate
lr_max: 0.1                         # max learning rate
lr_epochs: '30,60,90,120'           # lr changing based epochs
lr_scheduler: "cosine_annealing"    # learning rate mode
warmup_epochs: 0                    # number of warmup epoch
batch_size: 32                      # batch size of input tensor
max_epoch: 150                      # only valid for taining, which is always 1 for inference
momentum: 0.9                       # momentum
weight_decay: 0.0001                # weight decay
loss_scale: 1024                    # loss scale
label_smooth: 1                     # label smooth
label_smooth_factor: 0.1            # label smooth factor
buffer_size: 10                     # shuffle buffer size
image_size: [224,224]               # image size
pad_mode: 'pad'                     # pad mode for conv2d
padding: 1                          # padding value for conv2d
has_bias: True                      # whether has bias in conv2d
batch_norm: False                   # whether has batch_norm in conv2d
keep_checkpoint_max: 10             # only keep the last keep_checkpoint_max checkpoint
initialize_mode: "KaimingNormal"    # conv2d init mode
has_dropout: True                   # whether using Dropout layer
```

- config for vgg16, custom dataset

```bash
num_classes: 5                    # number of dataset categories
lr: 0.001                         # learning rate
batch_size: 64                    # batch size of input tensor
num_epoch: 10                     # number of training epochs
momentum: 0.9                     # momentum
pad_mode: 'pad'                   # pad mode for conv2d
padding: 0                        # padding value for conv2d
has_bias: False                   # whether has bias in conv2d
batch_norm: False                 # whether has batch_norm in conv2d
initialize_mode: "KaimingNormal"  # conv2d init mode
has_dropout: True                 # whether using Dropout layer
ckpt_file: "./vgg16_bn_ascend_v170_imagenet2012_official_cv_top1acc74.33_top5acc92.1.ckpt" # The path to the pretrained weights file used by the migration
save_file: "./vgg16.ckpt"         # Weight file path saved after migration
train_path: "./datasets/train/"   # Migration train set path
eval_path: "./datasets/test/"     # Migration valid set path
split_path: "./datasets/"         # Migration dataset path
infer_ckpt_path: "./vgg16.ckpt"   # Weight file path used by CPU inference

```

### [Training Process](#contents)

#### Training

##### Run vgg16 on Ascend

- Training using single device(1p), using CIFAR-10 dataset in default

```bash
python train.py --config_path=/dir_to_code/cifar10_config.yaml --data_dir=your_data_path > out.train.log 2>&1 &
```

The python command above will run in the background, you can view the results through the file `out.train.log`.

After training, you'll get some checkpoint files in specified ckpt_path, default in ./output directory.

You will get the loss value as following:

```bash
# grep "loss is " output.train.log
epoch: 1 step: 781, loss is 2.093086
epcoh: 2 step: 781, loss is 1.827582
...
```

- Distributed Training

```bash
bash run_distribute_train.sh rank_table.json your_data_path
```

The above shell script will run distribute training in the background, you can view the results through the file `train_parallel[X]/log`.

You will get the loss value as following:

```bash
# grep "result: " train_parallel*/log
train_parallel0/log:epoch: 1 step: 97, loss is 1.9060308
train_parallel0/log:epcoh: 2 step: 97, loss is 1.6003821
...
train_parallel1/log:epoch: 1 step: 97, loss is 1.7095519
train_parallel1/log:epcoh: 2 step: 97, loss is 1.7133579
...
...
```

> About rank_table.json, you can refer to the [distributed training tutorial](https://www.mindspore.cn/tutorials/experts/en/r2.0/parallel/introduction.html).
> **Attention** This will bind the processor cores according to the `device_num` and total processor numbers. If you don't expect to run pretraining with binding processor cores, remove the operations about `taskset` in `scripts/run_distribute_train.sh`

##### Run vgg16 on GPU

- Training using single device(1p)

```bash
python train.py  --config_path=/dir_to_code/imagenet2012_config.yaml --device_target="GPU" --dataset="imagenet2012" --is_distributed=0 --data_dir=$DATA_PATH  > output.train.log 2>&1 &
```

- Distributed Training

```bash
# distributed training(8p)
bash scripts/run_distribute_train_gpu.sh /path/ImageNet2012/train"
```

### [Evaluation Process](#contents)

#### Evaluation

- Do eval as follows, need to specify dataset type as "cifar10" or "imagenet2012"

```bash
# when using cifar10 dataset
python eval.py --config_path=/dir_to_code/cifar10_config.yaml --data_dir=your_data_path --dataset="cifar10" --device_target="Ascend" --pre_trained=./*-70-781.ckpt > output.eval.log 2>&1 &

# when using imagenet2012 dataset
python eval.py --config_path=/dir_to_code/imagenet2012.yaml --data_dir=your_data_path --dataset="imagenet2012" --device_target="GPU" --pre_trained=./*-150-5004.ckpt > output.eval.log 2>&1 &
```

- The above python command will run in the background, you can view the results through the file `output.eval.log`. You will get the accuracy as following:

```bash
# when using cifar10 dataset
# grep "result: " output.eval.log
result: {'acc': 0.92}

# when using the imagenet2012 dataset
after allreduce eval: top1_correct=36636, tot=50000, acc=73.27%
after allreduce eval: top5_correct=45582, tot=50000, acc=91.16%
```

#### ONNX Evaluation

- Export your model to ONNX

```bash
# when using cifar10 dataset
python export.py --config_path cifar10_config.yaml --ckpt_file /path/to/checkpoint.ckpt --file_name /path/to/exported.onnx --file_format ONNX --device_target CPU

# when using imagenet2012 dataset
python export.py --config_path imagenet2012_config.yaml --ckpt_file /path/to/checkpoint.ckpt --file_name /path/to/exported.onnx --file_format ONNX --device_target CPU
```

- Run ONNX evaluation. Specify dataset type as "cifar10" or "imagenet2012"

```bash
# when using cifar10 dataset
python eval_onnx.py --config_path cifar10_config.yaml --file_name /path/to/vgg16_cifar.onnx --data_dir /path/to/cifar10-bin --device_target GPU > output.eval_onnx.log 2>&1 &

# when using imagenet2012 dataset
python eval_onnx.py --config_path imagenet2012_config.yaml --file_name /path/to/vgg16_imagenet.onnx --data_dir /path/to/imagenet2012/validation --device_target GPU > output.eval_onnx.log 2>&1 &
```

- The above python command will run in the background, you can view the results through the file `output.eval_onnx.log`. You will get the accuracy as following:

```bash
# when using cifar10 dataset
# grep "accuracy" output.eval_onnx.log
accuracy: 0.9894

# when using the imagenet2012 dataset
top-1 accuracy: 0.7332
top-5 accuracy: 0.9149
```

## Migration process

### Dataset split

- The data set division process is as follows, the /train and /test folders will be generated in the dataset directory, and the training and validation set images will be saved.

```bash
python src/data_split.py --split_path /dir_to_code/{SPLIT_PATH}
```

### Migration

- The migration process is as follows. The pre-training weight file needs to be downloaded [(https://download.mindspore.cn/models/r1.7/vgg16_bn_ascend_v170_imagenet2012_official_cv_top1acc74.33_top5acc92.1.ckpt)](https://download.mindspore.cn/models/r1.7/vgg16_bn_ascend_v170_imagenet2012_official_cv_top1acc74.33_top5acc92.1.ckpt) to ./vgg16 folder. After the training is completed, the file is saved as ./vgg16.ckpt by default.

```bash
python fine_tune.py --config_path /dir_to_code/cpu_config.yaml
```

### Eval

- The migration process is as follows, you need to specify the weight file to be migrated (default is ./vgg16.ckpt).

```bash
python eval.py --config_path /dir_to_code/cpu_config.yaml
```

### Model quick start

- The quick start process is as follows, you need to specify the weight file path and dataset path after training.

```bash
python quick_start.py --config_path /dir_to_code/cpu_config.yaml
```

## Inference Process

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

### [Export MindIR](#contents)

```shell
python export.py --config_path [YMAL_CONFIG_PATH] --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

The ckpt_file parameter is required,
`FILE_FORMAT` should be in ["AIR", "MINDIR"]

### Infer

Before performing inference, the mindir file must be exported by `export.py` script. We only provide an example of inference using MINDIR model.
Current batch_Size for imagenet2012 dataset can only be set to 1.

```shell
bash run_infer_cpp.sh [MINDIR_PATH] [DATASET_NAME] [DATASET_PATH] [NEED_PREPROCESS] [DEVICE_TYPE] [DEVICE_ID]
```

- `DATASET_NAME` can choose from ['cifar10', 'imagenet2012'].
- `NEED_PREPROCESS` means weather need preprocess or not, it's value is 'y' or 'n', if you choose y, the cifar10 dataset will be processed in bin format, the imagenet2012 dataset will generate label json file.
- `DEVICE_ID` is optional, default value is 0.

### result

Inference result is saved in current path, you can find result like this in acc.log file.

```bash
'acc': 0.92
```

## [Model Description](#contents)

### [Performance](#contents)

#### Training Performance

| Parameters                 | VGG16(Ascend)                                  |
| -------------------------- | ---------------------------------------------- |
| Model Version              | VGG16                                          |
| Resource                   |  Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8 |
| uploaded Date              | 07/05/2021                                     |
| MindSpore Version          | 1.3.0                                          |
| Dataset                    | CIFAR-10                                       |
| Training Parameters        | epoch=70, steps=781, batch_size = 64, lr=0.1   |
| Optimizer                  | Momentum                                       |
| Loss Function              | SoftmaxCrossEntropy                            |
| outputs                    | probability                                    |
| Loss                       | 0.01                                           |
| Speed                      | 1pc: 79 ms/step;  8pcs: 104 ms/step            |
| Total time                 | 1pc: 72 mins;  8pcs: 11.8 mins                 |
| Checkpoint for Fine tuning | 1.1G(.ckpt file)                               |
| Scripts                    |[vgg16](https://gitee.com/mindspore/models/tree/r2.0/official/cv/VGG/vgg16) |

| Parameters                 | VGG16(Ascend)                                  | VGG16(GPU)                                      |
| -------------------------- | ---------------------------------------------- |------------------------------------|
| Model Version              | VGG16                                          | VGG16                                           |
| Resource                   |  Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8 |NV SMX2 V100-32G                                 |
| uploaded Date              | 07/05/2021                                     | 07/05/2021                                       |
| MindSpore Version          | 1.3.0                                          | 1.3.0                                             |
| Dataset                    | ImageNet2012                                   |ImageNet2012                                     |
| Training Parameters        | epoch=70, steps=40036, batch_size = 32, lr=0.1 |epoch=150, steps=40036, batch_size = 32, lr=0.1  |
| Optimizer                  | Momentum                                       |Momentum                                         |
| Loss Function              | SoftmaxCrossEntropy                            |SoftmaxCrossEntropy                              |
| outputs                    | probability                                    |probability                                                 |
| Loss                       | 1.5~2.0                                        |1.5~2.0                                          |
| Speed                      | 1pc: 58 ms/step;  8pcs: 58.2 ms/step           |1pc: 81 ms/step; 8pcs 94.4ms/step                |
| Total time                 | 1pc: ~32h;  8pcs: ~4h                          |8pcs: 19.7 hours                                 |
| Checkpoint for Fine tuning | 1.1G(.ckpt file)                               |1.1G(.ckpt file)                                 |
| Scripts                    |[vgg16](https://gitee.com/mindspore/models/tree/r2.0/official/cv/VGG/vgg16) |                   |

#### Evaluation Performance

| Parameters          | VGG16(Ascend)               |
| ------------------- | --------------------------- |
| Model Version       | VGG16                       |
| Resource            | Ascend 910; OS Euler2.8     |
| Uploaded Date       | 07/05/2021                  |
| MindSpore Version   | 1.3.0                       |
| Dataset             | CIFAR-10, 10,000 images     |
| batch_size          |   64                        |
| outputs             | probability                 |
| Accuracy            | 1pc: 93.4%                  |

| Parameters          | VGG16(Ascend)               | VGG16(GPU)                     |
| ------------------- | --------------------------- |---------------------           |
| Model Version       | VGG16                       |    VGG16                       |
| Resource            | Ascend 910; OS Euler2.8     |   GPU                          |
| Uploaded Date       | 07/05/2021                  | 07/05/2021                     |
| MindSpore Version   | 1.3.0                       | 1.3.0                          |
| Dataset             | ImageNet2012, 5000 images   |ImageNet2012, 5000 images       |
| batch_size          |   64                        |    32                          |
| outputs             | probability                 |    probability                 |
| Accuracy            | 1pc: 70.0%               |1pc: 73.0%;                        |

## [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py.

## [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
