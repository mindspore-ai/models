
# Contents

[查看中文](./README_CN.md)

<!-- TOC -->

- [Contents](#Contents)

- [3D_DenseNet](#3D_DenseNet)

- [Introduction](#Introduction)

- [DataSet](#DataSet)

- [Environment Requirements](#Environment-Requirements)

  -[Python Package](#Python-Package)

- [Quick Start](#Quick-Start)

  -[Installation](#Installation)

  -[Run on Modelarts](#Run-on-Modelarts)

- [Scripts Description](#Scripts-Description)

  -[Script and Sample Code](#Script-and-Sample-Code)

  -[Script Parameters](#Script-Parameters)

- [Training Process](#Training-Process)

  -[Training](#Training)

  -[Distributed Training](#Distributed-Training)

- [Evaluation Process](#Evaluation-Process)

  -[Evaluation](#Evaluation)

- [Export process](#Export-process)

  -[Export](#Export)

- [Performance](#Performance)

  -[Training performance](#Training-performance)

- [Contribution to guide](#Contribution-to-guide)

- [ModelZoo Homepage](#ModelZoo-Homepage)

# 3D_DenseNet

## Introduction

3D-SkipDenseSeg —— Skip-connected 3D DenseNet for volumetric infant brain MRI segmentation By Toan Duc Bui, Jitae Shin, Taesup Moon
6-month infant brain MRI segmentation aims to segment the brain into: White matter, Gray matter, and Cerebrospinal fluid. It is a difficult task due to larger overlapping
between tissues, low contrast intensity. We treat the problem by using very deep 3D convolution neural network. Our result achieved the top performance in 6 performance metrics.

## DataSet

The DataSet is  MICCAI Grand Challenge on [6-month infant brain MRI segmentation-in conjunction with MICCAI 2017](http://iseg2017.web.unc.edu).
The first year of life is the most dynamic phase of the postnatal human brain development, along with rapid tissue growth and development of a wide range of cognitive and motor
functions. This early period is critical in many neurodevelopmental and neuropsychiatric disorders, such as schizophrenia and autism. More and more attention has been paid to this critical period. For example, Baby Connectome Project (BCP) is recently started and will acquire and release thousands of infant MRI scans, which will greatly prosper the community

After Download the Dataset you will get two zip files, iSeg-2017-Training.zip and iSeg-2017-Testing.zip.Using Prepare_hdf5_cutedge. py in the code store to process the data,then collate the result of the target_PATH output.Specifically, the ninth HDF5 file is used as the validation set.The folder where eval and test are performed is the uncompressed raw data.Labels 1-10 are used for training and validation, and others are for test.

```python
└─data_train_no_cut
  ├── train_iseg_nocut_1.h5  // Training set directory structure
  ├── train_iseg_nocut_2.h5
  ├── train_iseg_nocut_3.h5
  ├── train_iseg_nocut_4.h5
  ├── train_iseg_nocut_5.h5
  ├── train_iseg_nocut_6.h5
  ├── train_iseg_nocut_7.h5
  ├── train_iseg_nocut_8.h5
  ├── train_iseg_nocut_10.h5
```

```python
└─data_val_no_cut
  ├── train_iseg_nocut_9.h5  //Validation set directory structure
```

```python
└─data_val
  ├──subject-9-label.hdr  // Evaluation set directory structure
  ├──subject-9-label.img
  ├──subject-9-T1.hdr
  ├──subject-9-T1.img
  ├──subject-9-T2.hdr
  ├──subject-9-T2.img
```

```python
└─data_test
  ├──subject-11-T1.hdr  // Test set directory structure
  ├──subject-11-T1
  ├──subject-11-T2.hdr
  ├──subject-11-T2
  ······
  ├──subject-23-T2.hdr
  ├──subject-23-T2
```

## Environment Requirements

-Hardware（Ascend or GPU）
  -Prepare hardware environment with Ascend or GPU.

-Framework

  -[MindSpore](https://www.mindspore.cn/install/en)

-For more information, please check the resources below:

  -[MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)

  -[MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

### Python Package

```python
opencv_python==4.5.2.52
numpy==1.19.4
MedPy==0.4.0
h5py==3.4.0
mindspore==1.3.0
PyYAML==5.4.1
SimpleITK==2.1.1
```

## Quick Start

```python
# In the defalut_config.yaml, add the data set path
train_dir: "/cache/data/data_train_nocut"
val_dir: "/cache/data/data_val_nocut"
```

```python
# Ascend/GPU/CPU environment running
# Modify device_target in the configuration file default_config.yaml to run in a different processor environment

# Run training Example
bash run_standalone_train.sh [TRAIN_DIR][VAL_DIR]
# example: bash  run_standalone_train.sh data/data_train_no_cut data/data_val_no_cut

# Run the distributed training example
bash run_distribute_train.sh [RANK_TABLE_FILE] [TRAIN_DIR] [VAL_DIR]
# example: bash run_distribute.sh  ranktable.json data/train_no_cut data/val_no_cut

# Run the evaluation example
bash run_eval.sh  [CHECKPOINT_FILE_PATH] [TEST_DIR]
# example: bash run_eval.sh 3D-DenseSeg-20000_36.ckpt data/data_val

# Run the test sample
bash run_test.sh [CHECKPOINT_FILE_PATH] [TEST_DIR]
# example: bash run_test.sh data/data_test
```

### Installation

- Step 1: Download the source code

```python
cd 3D_DenseNet
```

- Step 2: Download dataset at `http://iseg2017.web.unc.edu/download/` and change the path of the dataset `data_path` and saved path `target_path` in file `prepare_hdf5_cutedge.py`

```python
data_path = '/path/to/your/dataset/'
target_path = '/path/to/your/save/hdf5 folder/'
```

- Step 3: Generate hdf5 dataset

```python
python prepare_hdf5_cutedge.py
```

- Step 4: Run 1p training

```python
bash scripts/run_standalone_train.sh
```

if you get the information like below , it means that you run the script successfully

```python
============== Starting Training ==============
epoch: 1 step: 36, loss is 0.29248548
valid_dice: 0.5158226623757749
epoch time: 119787.393 ms, per step time: 3327.428 ms
epoch: 2 step: 36, loss is 0.4542764
valid_dice: 0.577169897796093
epoch time: 3151.715 ms, per step time: 87.548 ms
epoch: 3 step: 36, loss is 0.45287344
valid_dice: 0.6642792932561518
epoch time: 3145.802 ms, per step time: 87.383 ms
epoch: 4 step: 36, loss is 0.36013693
valid_dice: 0.6175640794605014
epoch time: 3161.118 ms, per step time: 87.809 ms
epoch: 5 step: 36, loss is 0.38933912
valid_dice: 0.6884333695452182
```

Run evaluation result

```python
bash run_eval.sh --checkpoint_file_path --eval_dir
```

After eval you will get the result as below.
Dice Coefficient (DC) for 9th subject (9 subjects for training and 1 subject for validation)

|                   |  CSF       | GM             | WM   | Average
|-------------------|:-------------------:|:---------------------:|:-----:|:--------------:|
|3D-SkipDenseSeg  | 93.66| 90.80 | 90.65 | 91.70 |

Notes: RANK_TABLE_FILE can refer to [Link](https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_ascend.html) , and the device_ip can be got as [Link](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools) For large models like InceptionV4, it's better to export an external environment variable export HCCL_CONNECT_TIMEOUT=600 to extend hccl connection checking time from the default 120 seconds to 600 seconds. Otherwise, the connection could be timeout since compiling time increases with the growth of model size. To avoid ops error，you should change the code like below:

in train.py：

```python
context.set_auto_parallel_context(parallel_mode=parallel_mode,
                                  device_num=rank_size,
                                  gradients_mean=False)### Conv3d op need set gradients_mean==False
```

```python
bash run_distribute_train.sh ranktable.json ../data
```

The results will be presented in the following file organization：

```python
.
└─scripts
  ├── train_parallel0
  │   ├── log.txt                      // Training log
  │   ├── env.txt                      // Environment log
  │   ├── XXX.yaml                     // Configuration during training
  │   ├──.
  │   ├──.
  │   ├──.
  ├── train_parallel1
  ├── .
  ├── .
  ├── train_parallel7
```

### Run on Modelarts

If you want to run in modelarts, please check the official documentation of modelarts, and you can start training and evaluation as follows:

```python
# run distributed training on modelarts example
# (1) First, Perform a or b.
#       a. Set "enable_modelarts=True" on yaml file.
#          Set other parameters on yaml file you need.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add other parameters on the website UI interface.
# (2) Set pip-requirements.txt to code directory
# (3) Set the code directory to "/path/3D_DenseNet" on the website UI interface.
# (4) Set the startup file to "train.py" on the website UI interface.
# (5) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
# (6) Create your job.

# run evaluation on modelarts example
# (1) Copy or upload your trained model to S3 bucket.
# (2) Perform a or b.
#       a. Set "enable_modelarts=True" on yaml file.
#          Set "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" on yaml file.
#          Set "checkpoint_url=/The path of checkpoint in S3/" on yaml file.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" on the website UI interface.
#          Add "checkpoint_url=/The path of checkpoint in S3/" on the website UI interface.
# (3) Set pip-requirements.txt to code directory
# (4) Set the code directory to "/path/3D_DenseNet" on the website UI interface.
# (5) Set the startup file to "eval.py" on the website UI interface.
# (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
# (7) Create your job.

#  **Note** : When you using modelarts,To prevent path conflicts, you are advised to add prefixes before related paths，eg data_path: "/cache/data"  train_dir: "/cache/data/data_train_nocut" val_dir: "/cache/data/data_val_nocut"
```

Note that when you find the package is incompatible ，you can try the code like below:

```python
import os
os.system('pip install --upgrade pip -y')
os.system('pip uninstall numpy -y')
os.system('pip install numpy -y'）
```

## Scripts Description

### Script and Sample Code

```python
.
└─3D_DenseNet
  ├── README.md                      // descriptions about 3D_DenseNet
  ├── scripts
  │   ├──run_distribute_train.sh     // shell script for distributed on Ascend
  │   ├──run_standalone_train.sh     // shell script for standalone on Ascend
  ├── src
  │   ├──common.py                   // common data processing function
  │   ├──dataloader.py               // dataloader for MindSpore
  │   ├──eval_call_back.py           // custom callback function for MindSpore
  │   ├──loss.py                     // loss function
  │   ├──lr_schedule.py              // learning rate functions
  │   ├──metrics.py                  // train and eval mertric function
  │   ├──model.py                    // definition of 3D_DenseNet model
  │   ├──prepare_hdf5_cutedge.py     // Prepare the data set function
  │   ├──var_init.py                 // Network parameters initialize
          ├── model_utils
          │   ├──config.py                    // parameter configuration
          │   ├──device_adapter.py            // device adapter
          │   ├──local_adapter.py             // local adapter
          │   ├──moxing_adapter.py            // moxing adapter
  ├── default_config.yaml             // parameter configuration
  ├── train.py                        // training script
  ├── eval.py                         // evaluation script
  ├── test.py                         // script for test_data_set
  ├──export.py                        // Export the checkpoint file to air/mindir in development
```

### Script Parameters

Training and evaluation parameters can be configured in the config.yaml

```Python
enable_modelarts: False                                         # when you use modelarts cloud environment you need set it True
# Url for modelarts
data_url: ""
train_url: ""
checkpoint_url: ""
# Path for local
run_distribute: False                                           # when you want run distributely ,set it True
enable_profiling: False                                         # whether profiling
data_path: "./data"                                             # local data_path
train_dir: "/data/data_train_nocut"                             # train data path
val_dir: "/data/data_val_nocut"                                 # valid data path  during training
eval_dir : "/data/data_val"                                     # eval data path
test_dir : "/data/iseg-testing"                                 # test data path
output_path: "./saved"                                          # local output_path
load_path: "./checkpoint_path/"                                 # local load_path
device_target: "Ascend"
checkpoint_path: "./checkpoint/" # local checkpoint_path
checkpoint_file_path: "3D-DenseSeg-20000_36.ckpt"    # local checkpoint_file_path

# Training options
lr: 0.002                                                       # learning rate
batch_size: 1                                                   # batch_size
epoch_size: 20000                                               # epoch_size
num_classes: 4                                                  # The number of classes
num_init_features: 32                                           # The model init features
save_checkpoint_steps : 5000                                    # every 5000 step save
keep_checkpoint_max: 16                                         # only keep the last
loss_scale: 256.0                                               # loss scale
drop_rate: 0.2                                                  # drop out rate
```

## Training Process

### Training

```python
 bash  run_standalone_train.sh data/data_train_no_cut data/data_val_no_cut
```

The  commands are run in the background. The results are stored in the scripts/train folder. You can view the results in the train.log file. After the training, you can find the checkpoint file in the default scripts folder.

```python
# grep "loss is " train.log
epoch: 1 step: 36, loss is 0.29248548
valid_dice: 0.5158226623757749
epoch: 2 step: 36, loss is 0.4542764
valid_dice: 0.577169897796093
```

### Distributed Training

```python
bash run_distribute.sh  ranktable.json data/train_no_cut data/val_no_cut
```

The shell script runs the distributed training in the background.
You can view the results through the train_PARALLEL [X]/log file.
The loss value shall be achieved in the following ways:

```python
# grep "result:" train_parallel*/log
train_parallel0/log:epoch: 1 step: 4, loss is 0.8156291
train_parallel0/log:epoch: 2 step: 4, loss is 0.47224823
train_parallel1/log:epoch: 1 step: 4, loss is 0.7149776
train_parallel1/log:epoch: 2 step: 4, loss is 0.47474277
```

## Evaluation Process

### Evaluation

Evaluate the test data set while the Ascend environment is running
Check the checkpoint path used for evaluation before running the following command.
Set the checkpoint path to absolute full path

```python
bash run_eval.sh username/3D_DenseNet/3D-DenseSeg-20000_36.ckpt username/3D_DenseNet/data/data_val
```

The python commands above run in the background, and you can view the results in the eval.log file.After a few minutes, the results are presented in the following table:
|                   |  CSF       | GM             | WM   | Average
|-------------------|:-------------------:|:---------------------:|:-----:|:--------------:|
|3D-SkipDenseSeg  | 93.66| 90.80 | 90.65 | 91.70 |

## Export process

### Export

execute the corresponding export function

```python
python export.py
```

By default, a mindir file is generated in the current code directory

## Performance

### Training performance

|           | Ascend                                                  |
| ------------------- | --------------------------------------------------------- | ---------------------------------------------------- |
|Model       | 3D_DenseNet                                             |
| MindSpore Version | 1.3.0                                    |
| DataSet             | i-seg2017                                      |
| Trainings Params | epoch = 20000,  batch_size = 1                    |
| Optimizer           | SGD                                            |
| Loss Function       | SoftmaxCrossEntropyWithLogits                  |
| Parameters        | 466320                                           |
| Dice    |91.70                                                       |

## Contribution to guide

If you want to be a part of this effort, read on [Contribution Guide](https://gitee.com/mindspore/models/blob/master/CONTRIBUTING_CN.md) and [how_to_contribute](https://gitee.com/mindspore/models/tree/master/how_to_contribute)

## ModelZoo Homepage

Please check the official [Home page](https://gitee.com/mindspore/models)
