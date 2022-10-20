# Contents

- [TRN Description](#trn-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Parameter configuration](#parameter-configuration)
    - [Training Process](#training-process)
    - [Evaluation Process](#evaluation-process)
    - [Model Export](#model-export)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
        - [Evaluation Performance](#evaluation-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

## [TRN Description](#contents)

Temporal Relationship Network (TRN) is an interpreted network module that allows
for temporal relational reasoning in neural networks and aims to describe the
temporal relations between observations in videos. TRN can learn and discover
possible temporal relations at multiple time scales. TRN can be used in a
plug-and-play fashion with any existing CNN architecture (in this work use BN Inception).

[Paper](https://arxiv.org/pdf/1711.08496v2.pdf):
Zhou, Bolei & Andonian, Alex & Torralba, Antonio. (2017).
Temporal Relational Reasoning in Videos. arXiv:1711.08496v2 [cs.CV] 25 Jul 2018.

## [Model Architecture](#contents)

The model uses a BNInception backbone. The last fully-connected layer is substituted with
Dropout and FC layer to match the necessary number of features, extracted from the video frames.
The TRN head consists of the Perceptron layers which aggregate feature from
the different combinations of the frames features.

## [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

### Dataset description

- Dataset link: [Jester](https://developer.qualcomm.com/software/ai-datasets/jester)
- Jester Dataset size: 22.9GB, 148,092 videos (37 frames each), 27 classes
    - Train: 18,3GB, 118,562 videos
    - Validation: 2,3GB，14,787 videos
    - Test: 2,3GB, 14,743 videos
    - Data format: folders with video frames

### Prepare the dataset

#### 1. Download the Jester dataset

Dataset [link](https://developer.qualcomm.com/software/ai-datasets/jester)

After logging you will need to download 24 ZIP archives containing
video frames data and labels lists.

File structure should look like this:

```text
JESTER/
  ├── 20bn-jester-download-package-labels.zip
  ├── 20bn-jester-v1-00.zip
  ├── … … …
  ├── 20bn-jester-v1-21.zip
  └── 20bn-jester-v1-22.zip
```

#### 2. Unpack the dataset

You can unpack it manually or use a convenience script. `unpack_jester_dataset.sh`

From the model root run

```bash
bash scripts/unpack_jester_dataset.sh [DATA_PATH] [TARGET_PATH]
```

- DATA_PATH - Path to the folder containing the ZIP archives from the original dataset (JESTER)
- TARGET_PATH - Path, where the unpacked dataset will be stored.

Example:

```bash
bash scripts/unpack_jester_dataset.sh /path/to/JESTER /path/to/unpacked_JESTER
```

The log in the terminal should look like this:

```text
Target data folder: /path/to/unpacked_JESTER
Unzip...
…
Extract tar archives...
…
Remove unnecessary data...
…
DONE!
```

The structure of the unpacked will be as follows:

```text
unpacked_JESTER/
 ├── 20bn-jester-v1/
 │   ├── 1/
 │   │   ├── 00001.jpg
 │   │   ├── 00002.jpg
 │   │   ├── …
 │   │   └── 00037.jpg
 │   ├── 2/
 │   ├── 3/
 │   ├── …
 │   └── 148092/
 └── labels/
     ├── labels.csv
     ├── test-answers.csv
     ├── test.csv
     ├── train.csv
     └── validation.csv
```

#### 3. Prepare dataset markup files

Install required packages using this command:

```bash
pip install -r requirements.txt
```

And then run the script `preprocess_jester_dataset.sh`:

```bash
bash scripts/preprocess_jester_dataset.sh [DATASET_ROOT]
```

- DATASET_ROOT - Path to the unpacked dataset

Example:

```bash
bash scripts/preprocess_jester_dataset.sh /path/to/unpacked_JESTER
```

The standard output will look similar to the following:

```text
dataset path: /path/to/unpacked_JESTER
labels path: /path/to/unpacked_JESTER/labels
labels save to: /path/to/unpacked_JESTER

Prepare training folders list
…
Prepare validation folders list
…
```

> Please, make sure you have permissions to write to the dataset root directory.

After that, you can find three new files in the dataset root directory:
`categories.txt`, `train_videofolder.txt`, `val_videofolder.txt`.

```text
unpacked_JESTER/
 ├── 20bn-jester-v1/
 ├── labels/
 ├── categories.txt
 ├── train_videofolder.txt
 └── val_videofolder.txt
```

## [Environment Requirements](#contents)

- Hardware（GPU）
    - Prepare hardware environment with GPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)
- Checkpoint of the BNInception model, trained on ImageNet2012.
    - Download the original model from [here](https://yjxiong.blob.core.windows.net/models/bn_inception-9f5701afb96c8044.pth)
    - Convert the checkpoint into MindSpore format (Please refer the section ["Quick Start"](#quick-start)).

## [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

Download the checkpoint of the BNInception model trained on ImageNet: [link](https://yjxiong.blob.core.windows.net/models/bn_inception-9f5701afb96c8044.pth)

Convert the checkpoint into the MindSpore format (make sure you have `torch` installed):

```bash
bash scripts/convert_bn_inception.sh /path/to/bn_inception-9f5701afb96c8044.pth /out/path/to/bn_inception.ckpt
```

Train the model:

```bash
bash scripts/run_standalone_train_gpu.sh 0 /path/to/Jester/dataset /path/to/bn_inception.ckpt
```

Perform the model evaluation:

```bash
bash scripts/run_eval_gpu.sh /path/to/Jester/dataset /path/to/trained-trn-model.ckpt
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```text
 .
 ├── configs
 │   └── jester_config.yaml
 ├── model_utils
 │   ├── __init__.py
 │   ├── config.py
 │   ├── device_adapter.py
 │   ├── local_adapter.py
 │   ├── logging.py
 │   ├── moxing_adapter.py
 │   └── util.py
 ├── scripts
 │   ├── convert_bn_inception.sh
 │   ├── preprocess_jester_dataset.sh
 │   ├── run_distributed_train_gpu.sh
 │   ├── run_eval_gpu.sh
 │   ├── run_export_gpu.sh
 │   ├── run_standalone_train_gpu.sh
 │   └── unpack_jester_dataset.sh
 ├── src
 │   ├── __init__.py
 │   ├── bn_inception.py
 │   ├── convert_bn_inception.py
 │   ├── preprocess_jester_dataset.py
 │   ├── train_cell.py
 │   ├── transforms.py
 │   ├── trn.py
 │   ├── tsn.py
 │   ├── tsn_dataset.py
 │   └── utils.py
 ├── eval.py
 ├── export.py
 ├── README.md
 ├── requirements.txt
 └── train.py
 ```

### [Script Parameters](#contents)

Parameters for both training and evaluation can be set in jester_config.yaml/jester_config.yaml.

Default configuration fpr training the TRN model on the JESTER dataset

```yaml
# Example for the 1P training
# Model options
num_segments: 8                           # Number of input frames
subsample_num: 3                          # Number of sub-samples for each TRN head

# Dataset options
image_size: 224                           # Size for resize input image
img_feature_dim: 256                      # Backbone out channels

# Training options
lr: 0.001                                 # Learning rate
clip_grad_norm: 20.0                      # Maximum gradients norm
update_lr_epochs: 50                      # Number of epochs before learning rate update
epochs_num: 120                           # Number of epochs
train_batch_size: 24                      # The batch size to be used for training
momentum: 0.9                             # Momentum
dropout: 0.8                              # Dropout probability
weight_decay: 0.0005                      # Weight decay
```

### [Training Process](#contents)

#### Usage

- Training using a single device(1p)

    ```bash
    bash scripts/run_standalone_train_gpu.sh [DEVICE_ID] [DATASET_ROOT] [PRETRAIN_BNINCEPTION_CKPT]
    ```

    - DEVICE_ID - Device ID
    - DATASET_ROOT - Path to the Jester dataset root directory
    - PRETRAIN_BNINCEPTION_CKPT - Path to the BNInception model, trained on

- Distributed Training

    ```bash
    bash scripts/run_distributed_train_gpu.sh [DATASET_ROOT] [PRETRAIN_BNINCEPTION_CKPT]
    ```

    - DATASET_ROOT - Path to the Jester dataset root directory
    - PRETRAIN_BNINCEPTION_CKPT - Path to the BNInception model, trained on

#### Result

Example of the training output:

```text
...
epoch: 1 step: 1800, loss is 1.4732825
epoch: 1 step: 1800, loss is 2.1691096
epoch time: 439722.374 ms, per step time: 237.431 ms
epoch time: 439731.201 ms, per step time: 237.436 ms
epoch time: 439671.819 ms, per step time: 237.404 ms
...
epoch: 120 step: 1812, loss is 0.65822375
epoch: 120 step: 1812, loss is 0.73821753
epoch: 120 step: 1812, loss is 0.92534214
epoch: 120 step: 1812, loss is 0.25449076
epoch time: 389752.648 ms, per step time: 210.450 ms
epoch time: 389747.472 ms, per step time: 210.447 ms
epoch time: 389751.522 ms, per step time: 210.449 ms
```

### [Evaluation Process](#contents)

#### Usage

```bash
bash scripts/run_eval_gpu.sh [DATASET_ROOT] [CKPT_PATH]
```

- DATASET_ROOT - Path to the Jester dataset root directory
- CKPT_PATH - Path to the trained TRN-MultiScale model

#### Result

The evaluation results will be stored in the ./eval-logs directory.

Example of the evaluation output:

```text
[DATE/TIME]:INFO:start evaluation
[DATE/TIME]:INFO:evaluation finished
[DATE/TIME]:INFO:Result:
[DATE/TIME]:INFO:Steps: 100; top1: 0.939394; top5: 1.000000
[DATE/TIME]:INFO:Steps: 200; top1: 0.949749; top5: 1.000000
...
[DATE/TIME]:INFO:Steps: 14600; top1: 0.945133; top5: 0.997808
[DATE/TIME]:INFO:Steps: 14700; top1: 0.945166; top5: 0.997755
epoch time: 3663958.959 ms, per step time: 247.782 ms
[DATE/TIME]:INFO:Top1: 94.50%
[DATE/TIME]:INFO:Top5: 99.78%
```

### [Model Export](#contents)

```bash
bash scripts/run_export_gpu.sh [DATASET_ROOT] [CKPT_PATH]
```

- DATASET_ROOT - Path to the Jester dataset root directory (necessary for retrieving the number of classes)
- CKPT_PATH - Path to the trained TRN-MultiScale model

> The exported models will be stored in the ./export-logs directory.

## [Model Description](#contents)

### [Performance](#contents)

#### Training Performance

| Parameters                 | GPU 8p                            |
| -------------------------- | --------------------------------- |
| Model Version              | TRN-MultiScale                    |
| Resource                   | 8x PCIE V100                      |
| Upload Date                | 03/25/2022 (mm/dd/yyyy)           |
| MindSpore Version          | 1.5.0                             |
| Dataset                    | 20bn-Jester-v1                    |
| Training Parameters        | epoch=120, batch_size=8, lr=0.001 |
| Optimizer                  | SGD                               |
| Loss Function              | SoftmaxCrossEntropy               |
| Speed                      | 220 ms/step                       |
| Total time                 | 13h 43m 32s                       |

#### Evaluation Performance

| Parameters          | GPU 1p                     |
| ------------------- | -------------------------- |
| Model Version       | TRN-MultiScale             |
| Resource            | 1x PCIE V100               |
| Upload Date         | 03/25/2022 (mm/dd/yyyy)    |
| MindSpore Version   | 1.5.0                      |
| Dataset             | 20bn-Jester-v1             |
| batch_size          | 1                          |
| Accuracy            | Top1: 94.50%, Top5: 99.78% |

## [Description of Random Situation](#contents)

We use random seed in train.py, eval.py and export.py.

## [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
