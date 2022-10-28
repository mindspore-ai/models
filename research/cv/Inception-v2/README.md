# Contents

- [Inception-v2 Description](#inception-v2-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Feature](#feature)
    - [Mixed precision（Ascend](#mixed-precision-ascend)
- [Environmental Requirements](#environmental-requirements)
- [Script Description](#script-description)
    - [Script and sample code](#script-and-sample-code)
    - [Script parameters](#script-paramenters)
    - [Training process](#training-process)
    - [Evaluation](#evaluation)
    - [Export process](#export-process)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training performance](#training-performance)
            - [Inception-v2 on ImageNet-1k](#inception-v2-on-imagenet-1k)
        - [Evaluation performance](#evaluation-performance)
            - [Inception-v2 on ImageNet-1k](#inception-v2-on-imagenet-1k)
- [ModelZoo Homepage](#modelzoo-homepage)

# [Inception-v2 Description](#contents)

Google's Inception-v2 is the second release in a series of deep learning convolutional architectures. Inception-v2 mainly adds BatchNorm to Inception-v1 and modifies
the previous Inception architecture to reduce the consumption of computing resources. This idea was proposed in the article ['Rethinking the Inception Architecture for
Computer Vision'](https://arxiv.org/pdf/1512.00567.pdf) published in 2015.

# [Model Architecture](#contents)

The overall architecture of the Inception-v2 is described in the article:

[Paper](https://arxiv.org/pdf/1512.00567.pdf)

# [Dataset](#contents)

Dataset used: [ImageNet2012](http://www.image-net.org/)

- Dataset size: a total of 1000 classes, 224*224 color images
    - Training set: 1,281,167 images in total
    - Test set: 50,000 images in total
- Data format: JPEG
    - Note: Data is processed in dataset.py.
- Download the dataset, the directory structure is as follows:

  ```text
  └─dataset
    ├─train                 # training dataset
    └─val                   # validation dataset
  ```

# [Feature](#contents)

## Mixed precision

The training method using [mixed precision](https://mindspore.cn/tutorials/en/master/advanced/mixed_precision.html) uses support for single-precision and
half-precision data to improve the training speed of deep learning neural networks, while maintaining the network accuracy that single-precision training can achieve.
Mixed-precision training increases computational speed and reduces memory usage while enabling training of larger models on specific hardware or enabling larger batches of
training.

Taking the FP16 operator as an example, if the input data type is FP32, the MindSpore background will automatically reduce the precision to process the data. You can open
the INFO log and search for "reduce precision" to view operators with reduced precision.

# [Environmental Requirements](#contents)

- Hardware
    - Use GPU to build the hardware environment
- Frame
    - [MindSpore](https://www.mindspore.cn/install/en)
- For details，see the following resources:
    - [MindSpore tutorial](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# [Script Description](#contents)

## Script and sample code

```text
Inception-v2
├── scripts
│   ├── run_distributed_train_gpu.sh   # GPU 8 card training
│   ├── run_eval_gpu.sh                # Evaluation on GPU
│   └── run_standalone_train_gpu.sh    # GPU single-card training
├── src
│   ├── config.py                      # Get GPU, Ascend, and CPU configuration parameters
│   ├── dataset.py                     # Dataset
│   ├── inception_v2.py                # Network definition
│   ├── loss.py                        # Custom cross entropy loss function
│   └── lr_generator.py                # Learning rate generator
├── eval.py                            # Network evaluation script
├── export.py                          # Script for export into AIR and MINDIR formats
├── README.md                          # Readme (in English)
├── requirements.txt                   # The list of required packages
└── train.py                           # Network training script
```

## Script parameters

The main parameters in config.py are as follows:

```text
'decay_method': 'cosine'             # Learning rate scheduler mode
"loss_scale": 1024                   # Loss scale
'batch_size': 128                    # Batch size
'epoch_size': 250                    # The number of epochs
'num_classes': 1000                  # The number of classes
'smooth_factor': 0.1                 # Label smoothing factor
'lr_init': 4e-5                      # Initial learning rate
'lr_max': 4e-1                       # Maximum learning rate
'lr_end': 4e-6                       # Minimum learning rate
'warmup_epochs': 1                   # The number of warmup epochs
'weight_decay': 4e-5                 # Weight decay
'momentum': 0.9                      # Momentum
'opt_eps': 1.0                       # Epsilon
'dropout_keep_prob': 0.8             # The probability to keep the input data for a dropout layer
'amp_level': O3                      # The option of the parameter `level` in `mindspore.amp.build_train_network`, choose from [O0, O2, O3]
```

Refer to the script `config.py` for more configuration details.

## Training process

After installing MindSpore through the official website, run the following command from directory `research/cv/Inception-v2`:

```bash
pip install -r requirements.txt
```

Then you can follow the steps below for training and evaluation:

- GPU:

  Training on GPU:

  ```bash
  # standalone training
  bash scripts/run_standalone_train_gpu.sh [DEVICE_ID] [DATASET_PATH] [<PRE_TRAINED_PATH>]
  # multi-gpu training
  bash scripts/run_distributed_train_gpu.sh [RANK_SIZE] [DATASET_PATH] [<PRE_TRAINED_PATH>]
  ```

  Example:

  ```bash
  # standalone training
  bash scripts/run_standalone_train_gpu.sh 0 /path/to/imagenet
  # multi-gpu training
  bash scripts/run_distributed_train_gpu.sh 8 /path/to/imagenet
  ```

## Evaluation

- GPU:

  Evaluation on GPU:

  ```bash
  bash scripts/run_eval_gpu.sh [DEVICE_ID] [DATASET_PATH] [CHECKPOINT_PATH]
  ```

  Example:

  ```bash
  bash scripts/run_eval_gpu.sh 0 /path/to/imagenet /path/to/trained.ckpt
  ```

## Export process

```bash
python export.py --ckpt_file [CKPT_FILE]  --platform [PLATFORM] --file_format [FILE FORMAT]
```

For FILE_FORMAT choose MINDIR or AIR.

Example:

```bash
python export.py --ckpt_file /path/to/trained.ckpt  --platform GPU --file_format MINDIR
```

The exported model will be named after the structure of the model and saved in the current directory.

# [Model Description](#contents)

## Performance

### Training performance

#### Inception-v2 on ImageNet-1k

| Parameters          | GPU                                                    |
|---------------------|--------------------------------------------------------|
| Model               | Inception-v2                                           |
| Resources           | GPU: GeForce RTX 3090  CPU 2.90GHz, 64 cores  RAM:252G |
| Upload date         | 05 / 13 / 2022 (mm / dd / yyyy)                        |
| MindSpore version   | 1.6.1                                                  |
| Dataset             | ImageNet-1k Train，1,281,167 images                     |
| Training parameters | epoch=250, batch_size=128                              |
| Optimizer           | Momentum                                               |
| Loss function       | CrossEntropy                                           |
| Loss                | 1.8897                                                 |
| Output              | Probability                                            |
| Accuracy            | 8P: top1: 76.25% top5: 92.92%                          |
| Speed               | 8P: 295 ms/step                                        |
| Training time       | 25h 37m 41s                                            |

### Evaluation performance

#### Inception-v2 on ImageNet-1k

| Parameters        | GPU                                                    |
|-------------------|--------------------------------------------------------|
| Model             | Inception-v2                                           |
| Resources         | GPU: GeForce RTX 3090  CPU 2.90GHz, 64 cores  RAM:252G |
| Upload date       | 05 / 13 / 2022  (mm / dd / yyyy)                       |
| MindSpore version | 1.6.1                                                  |
| Dataset           | ImageNet-1k Val，50,000 images                          |
| Accuracy          | top1: 76.25% top5: 92.92%                              |

# [ModelZoo Homepage](#contents)

Please visit the official [website](https://gitee.com/mindspore/models)