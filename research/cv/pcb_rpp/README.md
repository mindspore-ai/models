# Contents

- [Contents](#contents)
    - [Model Description](#model-description)
    - [Model Architecture](#model-architecture)
    - [Dataset](#dataset)
    - [Environment Requirements](#environment-requirements)
    - [Quick Start](#quick-start)
        - [Running scripts](#running-scripts)
    - [Script Description](#script-description)
        - [Script and Sample Code](#script-and-sample-code)
        - [Script Parameters](#script-parameters)
            - [Training Script Parameters](#training-script-parameters)
            - [Running Options](#running-options)
            - [Network Parameters](#network-parameters)
    - [Training Process](#training-process)
    - [Evaluation Process](#evaluation-process)
    - [Inference Process](#inference-process)
        - [Export MindIR](#export-mindir)
        - [result](#result)
    - [Model Description](#model-description)
        - [Performance](#performance)
            - [Training Performance](#training-performance)
            - [Evaluation Performance](#evaluation-performance)
    - [Description of Random Situation](#description-of-random-situation)
    - [ModelZoo Homepage](#modelzoo-homepage)

## [Model Description](#contents)

Models target at learning discriminative part-informed features for person retrieval and make two contributions. (i) A network named Part-based Convolutional Baseline (PCB).
Given an image input, it outputs a convolutional descriptor consisting of several part-level features.
With a uniform partition strategy, PCB achieves competitive results with the state-of-the-art methods, proving itself as a strong convolutional baseline for person retrieval.
(ii) A refined part pooling (RPP) method. Uniform partition inevitably incurs outliers in each part, which are in fact more similar to other parts. RPP re-assigns these outliers to the parts they are closest to, resulting in refined parts with enhanced within-part consistency.

[Paper](https://arxiv.org/abs/1711.09349):  Sun Y., Zheng L., et al. “Beyond Part Models: Person Retrieval with Refined Part Pooling (and a Strong Convolutional Baseline)”. COMPUTER VISION - ECCV 2018, PT IV, (2018): 501-518.

## [Model Architecture](#contents)

First, authors propose a network named Part-based Convolutional Baseline (PCB) which conducts uniform partition on the conv-layer for learning part-level features.
It does not explicitly partition the images.
PCB takes a whole image as the input and outputs a convolutional feature.
Being a classification net, the architecture of PCB is concise, with slight modifications on the backbone network.
The training procedure is standard and requires no bells and whistles.
Authors show that the convolutional descriptor has much higher discriminative ability than the commonly used fully-connected (FC) descriptor.

Second, authors propose an adaptive pooling method to refine the uniform partition.
Authors consider the motivation that within each part the contents should be consistent.
Authors observe that under uniform partition, there exist outliers in each part.
These outliers are, in fact, closer to contents in some other part, implying within-part inconsistency.
Therefore, authors refine the uniform partition by relocating those outliers to the part they are closest to, so that the within-part consistency is reinforced.

## [Dataset](#contents)

[Market1501](http://zheng-lab.cecs.anu.edu.au/Project/project_reid.html) dataset is used to train and test model.

Market1501 contains 32,668 images of 1,501 labeled persons of six camera views.
There are 751 identities in the training set and 750 identities in the testing set.
In the original study on this proposed dataset, the author also uses mAP as the evaluation criteria to test the algorithms.

Data structure:

```text
Market-1501-v15.09.15
├── bounding_box_test [19733 entries]
├── bounding_box_train [12937 entries]
├── gt_bbox [25260 entries]
├── gt_query [6736 entries]
├── query [3369 entries]
└── readme.txt
```

## [Environment Requirements](#contents)

- Hardware（GPU）
    - Prepare hardware environment with GPU processor.
- Framework
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

## [Quick Start](#contents)

### [Running scripts](#contents)

Model uses [pre-trained backbone](https://disk.yandex.ru/d/QG_OjyzNxSZMTA/pretrained_resnet50.ckpt) ResNet50 trained on ImageNet2012.

After dataset preparation, you can start training and evaluation as follows:

```bash
# run training example
# PCB
bash train_PCB_market_gpu.sh /path/to/market1501/ ../config/train_PCB_market.yaml /path/to/pretrained_resnet50.ckpt
# PCB + RPP
bash train_RPP_market_gpu.sh /path/to/market1501/ ../config/train_RPP_market/ /path/to/pretrained_resnet50.ckpt

# run distributed training example
# PCB
bash train_distribute_PCB_market_gpu.sh 8 /path/to/market1501/ ../config/train_PCB_market.yaml /path/to/pretrained_resnet50.ckpt
# PCB + RPP
bash train_distribute_RPP_market_gpu.sh 8 /path/to/market1501/ ../config/train_RPP_market/ /path/to/pretrained_resnet50.ckpt

# run evaluation example
# PCB
bash eval_PCB_market.sh /path/to/market1501 ../config/eval_PCB_market.yaml /your/path/checkpoint_file True GPU
# PCB + RPP
bash eval_RPP_market.sh /path/to/market1501 ../config/eval_RPP_market.yaml /your/path/checkpoint_file True GPU
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```text
PCB_RPP
├── config
│   ├── base_config.yaml
│   ├── eval_PCB_market.yaml  # Evaluation config for PCB model on Market1501 dataset
│   ├── eval_RPP_market.yaml  # Evaluation config for RPP model on Market1501 dataset
│   ├── train_PCB_market.yaml  # Training config for PCB model on Market1501 dataset
│   └── train_RPP_market  # Training configs for RPP model on Market1501 dataset
│       ├── train_PCB.yaml
│       └── train_RPP.yaml
├── scripts
│   ├── eval_PCB_market.sh  # Evaluation script for PCB model on Market1501 dataset
│   ├── eval_RPP_market.sh  # Evaluation script for RPP model on Market1501 dataset
│   ├── train_distribute_PCB_market_gpu.sh  # Distr training script for PCB model on Market1501 dataset on GPU
│   ├── train_distribute_RPP_market_gpu.sh  # Distr training script for RPP model on Market1501 dataset on GPU
│   ├── train_PCB_market_gpu.sh  # Training script for PCB model on Market1501 dataset on GPU
│   ├── train_RPP_market_gpu.sh  # Training script for RPP model on Market1501 dataset on GPU
├── src
│   ├── __init__.py
│   ├── dataset.py  # Produce the dataset
│   ├── datasets
│   │   ├── __init__.py
│   │   ├── cuhk03.py  # CUHK03 dataset processing
│   │   ├── duke.py  # Duke dataset processing
│   │   └── market.py  # Market1501 dataset processing
│   ├── eval_callback.py  # Evaluation callback when training
│   ├── eval_utils.py  # Evaluation utils
│   ├── logging.py  # Logging helper
│   ├── lr_generator.py # Learning rate scheduler
│   ├── meters.py  # Average storage
│   ├── model_utils
│   │   ├── config.py # Config parser
│   │   ├── device_adapter.py # Device adapter for ModelArts
│   │   ├── local_adapter.py # Environment variables parser
│   │   └── moxing_adapter.py # Moxing adapter for ModelArts
│   ├── pcb.py  # PCB 50 network structure
│   ├── resnet.py  # ResNet 50 network structure
│   └── rpp.py  # RPP 50 network structure
├── eval.py # Evaluate the network
├── export.py # Export the network
├── train.py # Train the network
├── pip-requirements.txt
├── README.md # README for GPU running
```

### [Script Parameters](#contents)

#### Training Script Parameters

```text
usage: train.py  --config_path CONFIG_PATH [--run_distribute DISTRIBUTE] [--device_target DEVICE]
                 [--dataset_path DATASET_PATH] [--checkpoint_file_path CHECKPOINT_FILE]
                 [--output_path OUT_PATH] [--device_num DEVICE_NUM]
                 [--log_save_path LOG_PATH] [--checkpoint_save_path CHECKPOINT_PATH]

options:
    --config_path              path to .yml config file
    --run_distribute           pre_training by several devices: "true"(training by more than 1 device) | "false", default is "false"
    --device_target            target device ("GPU" | "CPU")
    --dataset_path             path to dataset
    --checkpoint_file_path     path to pretrained checkpoint file
    --output_path              path to output checkpoint and logs
    --device_num               number of devices for distributed training
    --log_save_path            relative to output_path to saving logs
    --checkpoint_save_path     relative to output_path to saving checkpoint
```

#### Running Options

You can set parameters in `.yaml` configs or as running script parameters `python train.py ... --<param_name> <param_value>`

```text
    batch_size                      training batch size
    epoch_size                      number of epochs
    step_size                       epoch to decay
    warmup                          use learning rate warmup
    save_checkpoint                 should save checkpoints
    save_checkpoint_epochs          number of epochs between saving checkpoints
    keep_checkpoint_max             max number of last saved checkpoints
```

#### Network Parameters

```text
Parameters for dataset and network (Training/Evaluation):
    model_name                      model name ("PCB" | "RPP")
    num_parallel_workers            the number of dataset readers

Parameters for learning rate:
    learning_rate                   initial learning rate
    weight_decay                    optimizer weight decay
    warmup                          use learning rate warmup
    lr_mult                         backbone decay multiplier
    decay_rate                      step decay multiplier
```

## [Training Process](#contents)

- Set options in corresponding `.sh`scripts if needed.

- Run `train_PCB_market_gpu.sh` for non-distributed training of PCB model.

    ```bash
    bash train_PCB_market_gpu.sh /path/to/market1501/ ../config/train_PCB_market.yaml /path/to/pretrained_resnet50.ckpt
    ```

- Run `train_distribute_PCB_market_gpu.sh` for distributed training of PCB model.

    ```bash
    bash train_distribute_PCB_market_gpu.sh 8 /path/to/market1501/ ../config/train_PCB_market.yaml /path/to/pretrained_resnet50.ckpt
    ```

- Run `train_RPP_market_gpu.sh` for non-distributed training of PCB+RPP model.

    ```bash
    bash train_RPP_market_gpu.sh /path/to/market1501/ ../config/train_RPP_market/ /path/to/pretrained_resnet50.ckpt
    ```

- Run `train_distribute_RPP_market_gpu.sh` for distributed training of PCB+RPP model.

    ```bash
    bash train_distribute_RPP_market_gpu.sh 8 /path/to/market1501/ ../config/train_RPP_market/ /path/to/pretrained_resnet50.ckpt
    ```

## [Evaluation Process](#contents)

- Set options in `market1501_config.yaml`.

- Run `bash eval_PCB_market.sh` for evaluation of PCB model.

    ```bash
    bash eval_PCB_market.sh /path/to/market1501 ../config/eval_PCB_market.yaml /your/path/checkpoint_file True GPU
    ```

- Run `bash eval_RPP_market.sh` for evaluation of PCB+RPP model.

    ```bash
    bash eval_RPP_market.sh /path/to/market1501 ../config/eval_RPP_market.yaml /your/path/checkpoint_file True GPU
    ```

## Inference Process

### [Export MindIR](#contents)

```text
python export.py --model_name [MODEL_NAME] --file_name [FILE_NAME] --file_format [FILE_FORMAT] --checkpoint_file_path [CKPT_PATH] --use_G_feature [USE_G_FEATURE] --config_path [CONFIG_PATH] --device_target GPU

options:
    --model_name               model name ("PCB" | "RPP")
    --config_path              path to .yml config file
    --checkpoint_file_path     checkpoint file
    --file_name                output file name
    --file_format              output file format, choices in ['MINDIR']
    --use_G_feature            use G features (if False use H features)
```

The ckpt_file and config_path parameters are required,
`FILE_FORMAT` should be "MINDIR"

### result

Inference result will be shown in the terminal

## [Model Description](#contents)

### [Performance](#contents)

#### PCB

##### Training Performance

| Parameters                 | GPU                                                            |
| -------------------------- | -------------------------------------------------------------- |
| Resource                   | 1x Tesla V100-PCIE 32G                                         |
| uploaded Date              | 01/20/2022 (month/day/year)                                    |
| MindSpore Version          | 1.5.0                                                          |
| Dataset                    | Market1501                                                     |
| Training Parameters        | learning_rate=0.1, batch_size=64, epoch_size=60, step_size=40  |
| Optimizer                  | SGD                                                            |
| Loss Function              | SoftmaxCrossEntropyWithLogits                                  |
| Speed                      | 231ms/step (1pcs)                                              |
| Loss                       | 0.057                                                          |
| Params (M)                 | 27.2                                                           |
| Checkpoint for inference   | 327Mb (.ckpt file)                                             |
| Scripts                    | [PCB scripts](scripts) |

##### Evaluation Performance

| Parameters          | GPU                         |
| ------------------- | --------------------------- |
| Resource            | 1x Tesla V100-PCIE 32G      |
| Uploaded Date       | 01/21/2022 (month/day/year) |
| MindSpore Version   | 1.5.0                       |
| Dataset             | Market1501                  |
| batch_size          | 64                          |
| outputs             | mAP, Rank-1                 |
| Accuracy            | mAP: 77.6%, rank-1: 92.6%   |

#### PCB + RPP

##### Training Performance

1 stage (PCB)

| Parameters                 | GPU                                                            |
| -------------------------- | -------------------------------------------------------------- |
| Resource                   | 1x Tesla V100-PCIE 32G                                         |
| uploaded Date              | 01/21/2022 (month/day/year)                                    |
| MindSpore Version          | 1.5.0                                                          |
| Dataset                    | Market1501                                                     |
| Training Parameters        | learning_rate=0.1, batch_size=64, epoch_size=20, step_size=20  |
| Optimizer                  | SGD                                                            |
| Loss Function              | SoftmaxCrossEntropyWithLogits                                  |
| Speed                      | 231ms/step (1pcs)                                              |
| Loss                       | 0.613                                                          |
| Params (M)                 | 27.2                                                           |
| Checkpoint for inference   | 327Mb (.ckpt file)                                             |
| Scripts                    | [PCB scripts](scripts) |

2 stage (RPP)

| Parameters                 | GPU                                                            |
| -------------------------- | -------------------------------------------------------------- |
| Resource                   | 1x Tesla V100-PCIE 32G                                         |
| uploaded Date              | 01/21/2022 (month/day/year)                                    |
| MindSpore Version          | 1.5.0                                                          |
| Dataset                    | Market1501                                                     |
| Training Parameters        | learning_rate=0.01, batch_size=64, epoch_size=45, step_size=15 |
| Optimizer                  | SGD                                                            |
| Loss Function              | SoftmaxCrossEntropyWithLogits                                  |
| Speed                      | 187ms/step (1pcs)                                              |
| Loss                       | 0.048                                                          |
| Params (M)                 | 27.2                                                           |
| Checkpoint for inference   | 327Mb (.ckpt file)                                             |
| Scripts                    | [RPP scripts](scripts) |

##### Evaluation Performance

| Parameters          | GPU                         |
| ------------------- | --------------------------- |
| Resource            | 1x Tesla V100-PCIE 32G      |
| Uploaded Date       | 01/21/2022 (month/day/year) |
| MindSpore Version   | 1.5.0                       |
| Dataset             | Market1501                  |
| batch_size          | 64                          |
| outputs             | mAP, Rank-1                 |
| Accuracy            | mAP: 81.2%, rank-1: 93.1%   |

## [Description of Random Situation](#contents)

There are three random situations:

- Shuffle in the dataset.
- Random flip images.
- Initialization of some model weights.

Some seeds have already been set in train.py to avoid the randomness of dataset shuffle and weight initialization.

## [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
