# Contents

- [Description](#description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Features](#features)
    - [Mixed Precision](#mixed-precision)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
    - [Requirements Installation](#requirements-installation)
    - [Dataset Preparation](#dataset-preparation)
    - [Model Checkpoints](#model-checkpoints)
    - [Running](#running)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Training](#training)
        - [Distributed Training](#distributed-training)
    - [Evaluation Process](#evaluation-process)
- [Model Description](#model-description)
    - [Performance](#performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [Description](#contents)

There has been remarkable progress on object detection and re-identification in recent years which are the core components for multi-object tracking. However, little attention has been focused on accomplishing the two tasks in a single network to improve the inference speed. The initial attempts along this path ended up with degraded results mainly because the re-identification branch is not appropriately learned. In this work, we study the essential reasons behind the failure, and accordingly present a simple baseline to addresses the problems. It remarkably outperforms the state-of-the-arts on the MOT challenge datasets at 30 FPS. This baseline could inspire and help evaluate new ideas in this field. More detail about this model can be found in:

[Paper](https://arxiv.org/abs/2004.01888): Zhang Y, Wang C, Wang X, et al. FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking. 2020.

This repository contains a Mindspore implementation of FairMot based upon original Pytorch implementation (<https://github.com/ifzhang/FairMOT>). The training and validating scripts are also included, and the evaluation results are shown in the [Performance](#performance) section.

# [Model Architecture](#contents)

The overall network architecture of FairMOT is shown below:

[Link](https://arxiv.org/abs/2004.01888)

# [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

Dataset used: ETH, CalTech, MOT17, CUHK-SYSU, PRW, CityPerson

# [Features](#contents)

## [Mixed Precision](#contents)

The [mixed precision](https://mindspore.cn/tutorials/en/master/advanced/mixed_precision.html) training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data formats, and maintains the network precision achieved by the single-precision training at the same time. Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware. For FP16 operators, if the input data type is FP32, the backend of MindSpore will automatically handle it with reduced precision. Users could check the reduced-precision operators by enabling INFO log and then searching ‘reduce precision’.

# [Environment Requirements](#contents)

To run the python scripts in the repository, you need to prepare the environment as follow:

- Python and dependencies
    - Cython 0.29.23
    - opencv-python 4.5.1.4
    - cython-bbox 0.1.3
    - sympy 1.7.1
    - yacs
    - numba
    - progress
    - motmetrics 1.2.0
    - matplotlib 3.4.1
    - lap 0.4.0
    - openpyxl 3.0.7
    - Pillow 8.1.0
    - tensorboardX 2.2
    - python 3.7
    - mindspore 1.2.0
    - pycocotools 2.0
- For more information, please check the resources below：
    - [MindSpore tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# [Quick Start](#contents)

## [Requirements Installation](#contents)

Some packages in `requirements.txt` need Cython package to be installed first. For this reason, you should use the following commands to install dependencies:

```shell
pip install Cython && pip install -r requirements.txt
```

## [Dataset Preparation](#contents)

FairMot model uses mix dataset to train and validate in this repository. We use the training data as [JDE](https://github.com/Zhongdao/Towards-Realtime-MOT) in this part and we call it "MIX". Please refer to their [DATA ZOO](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md) to download and prepare all the training data including Caltech Pedestrian, CityPersons, CUHK-SYSU, PRW, ETHZ, MOT17 and MOT16.

**Configure path to dataset root** in `data/data.json` file.

## [Model Checkpoints](#contents)

Baseline FairMOT model (DLA-34 backbone) is pretrained on the CrowdHuman for 60 epochs with the self-supervised learning approach before training on the MIX dataset for 30 epochs.

The baseline model can be downloaded here: crowdhuman_dla34.pth [[Google]](https://drive.google.com/file/d/1SFOhg_vos_xSYHLMTDGFVZBYjo8cr2fG/view?usp=sharing) [[Baidu, code: ggzx ]](https://pan.baidu.com/s/1JZMCVDyQnQCa5veO73YaMw).

Then you need to convert this model from `pth` to `ckpt` using script `src/utils/pth2ckpt.py`:

```shell
# in root fairmot directory
python src/utils/pth2ckpt.py
```

Install the `torch` package using the following command to run model convert scripts:

```shell
pip install torch
```

## [Running](#contents)

To train the model, run the shell script `scripts/run_standalone_train_ascend.sh` or `scripts/run_standalone_train_gpu.sh` with the format below:

```shell
# standalone training on Ascend
bash scripts/run_standalone_train_ascend.sh DEVICE_ID DATA_CFG(options) LOAD_PRE_MODEL(options)

# standalone training on GPU
bash scripts/run_standalone_train_gpu.sh [config_file] [pretrained_model]

# distributed training on Ascend
bash scripts/run_distribute_train_ascend.sh RANK_SIZE DATA_CFG(options) LOAD_PRE_MODEL(options)

# distributed training on GPU
bash scripts/run_distribute_train_gpu.sh [DEVICE_NUM] [VISIBLE_DEVICES(0,1,2,3,4,5,6,7)] [config_file] [pretrained_model]
```

To validate the model, run the shell script `scripts/run_eval.sh` with the format below:

```shell
bash scripts/run_eval.sh [device] [config] [load_ckpt] [dataset_dir]
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

The structure of the files in this repository is shown below.

```text
└─fairmot
 ├─scripts
 │ ├─run_eval.sh                    // launch ascend standalone evaluation
 │ ├─run_onnx_eval.sh               // launch GPU standalone onnx model evaluation
 │ ├─run_distribute_train_ascend.sh // launch ascend distributed training
 | ├─run_distribute_train_gpu.sh    // launch gpu distributed training
 │ ├─run_standalone_train_ascend.sh // launch ascend standalone training
 │ └─run_standalone_train_gpu.sh    // launch gpu standalone training
 ├─src
 │ ├─tracker
 │ │ ├─basetrack.py               // basic tracker
 │ │ ├─matching.py                // calculating box distance
 │ │ ├─multitracker_onnx.py       // onnx calculating box distance
 │ │ └─multitracker.py            // JDETracker
 │ ├─tracking_utils
 │ │ ├─evaluation.py              // evaluate tracking results
 │ │ ├─kalman_filter.py           // Kalman filter for tracking bounding boxes
 │ │ ├─log.py                     // logging tools
 │ │ ├─io.py                      //I/o tool
 │ │ ├─timer.py                   // evaluation of time consuming
 │ │ ├─utils.py                   // check that the folder exists
 │ │ └─visualization.py           // display image tool
 │ ├─utils
 │ │ ├─callback.py                // custom callback functions
 │ │ ├─image.py                   // image processing
 │ │ ├─jde.py                     // LoadImage
 │ │ ├─logger.py                  // a summary writer logging
 │ │ ├─lr_schedule.py             // learning ratio generator
 │ │ ├─pth2ckpt.py                // pth transformer
 │ │ └─tools.py                   // image processing tool
 │ ├─fairmot_poase.py             // WithLossCell
 │ ├─losses.py                    // loss
 │ ├─config.py                    // total config
 │ ├─util.py                      // routine operation
 │ ├─infer_net.py                 // infer net
 │ └─backbone_dla_conv.py         // dla34_conv net
 ├─eval.py                        // eval fairmot
 ├─eval_onnx.py                   // eval onnx fairmot
 ├─fairmot_run.py                 // run fairmot
 ├─train.py                       // train fairmot
 ├─fairmot_export.py              // export fairmot
 ├─requirements.txt               // pip requirements
 ├─default_config.yaml            // default model configuration
 └─README.md                      // descriptions about this repository
```

## [Training Process](#contents)

### [Training](#contents)

Run `scripts/run_standalone_train_<device>.sh` to train the model standalone. The usage of the script is:

#### Running on Ascend

```shell
bash scripts/run_standalone_train_ascend.sh DEVICE_ID DATA_CFG LOAD_PRE_MODEL
```

For example, you can run the shell command below to launch the training procedure.

```shell
bash scripts/run_standalone_train_ascend.sh 0 ./dataset/ ./crowdhuman_dla34_ms.ckpt
```

#### Running on GPU

```shell
bash scripts/run_standalone_train_gpu.sh [config_file] [pretrained_model]
```

For example, you can run the shell command below to launch the training procedure:

```shell
bash scripts/run_standalone_train_gpu.sh ./default_config.yaml ./crowdhuman_dla34_ms.ckpt
```

The model checkpoint will be saved into `./train/ckpt`.

### [Distributed Training](#contents)

Run `scripts/run_distribute_train_<device>.sh` to train the model distributed. The usage of the script is:

#### Running on Ascend

```shell
bash scripts/run_distribute_train_ascend.sh RANK_SIZE DATA_CFG LOAD_PRE_MODEL
```

For example, you can run the shell command below to launch the distributed training procedure.

```shell
bash scripts/run_distribute_train_ascend.sh 8 ./data.json ./crowdhuman_dla34_ms.ckpt
```

#### Running on GPU

```shell
bash scripts/run_distribute_train_gpu.sh [DEVICE_NUM] [VISIBLE_DEVICES(0,1,2,3,4,5,6,7)] [config_file] [pretrained_model]
```

For example, you can run the shell command below to launch the distributed training procedure:

```shell
bash scripts/run_distribute_train_gpu.sh 8 0,1,2,3,4,5,6,7 ./default_config.yaml ./crowdhuman_dla34_ms.ckpt
```

The above shell script will run distribute training in the background. You can view the results through the file `train/tran.log`.

The model checkpoint will be saved into `train/ckpt`.

## [Evaluation Process](#contents)

The evaluation data set was [MOT20](https://motchallenge.net/data/MOT20/)

Run `scripts/run_eval.sh` to evaluate the model. The usage of the script is:

```shell
bash scripts/run_eval.sh [device] [config] [load_ckpt] [dataset_dir]
```

For example, you can run the shell command below to launch the validation procedure.

```shell
bash scripts/run_eval.sh GPU ./default_config.yaml ./fairmot-30.ckpt data_path
```

If you want to evaluate the ONNXM model, you need to run fairmot_ export. Py, select file_ format="ONNX"
Run scripts/run_onnx_eval.sh to evaluate the onnx model. The usage of the script is:

```shell
bash scripts/run_onnx_eval.sh [device] [config] [load_ckpt] [dataset_dir]
```

For example, you can run the shell command below to launch the validation procedure.

```shell
bash scripts/run_onnx_eval.sh [device] [config] [load_onnx] [dataset_dir]
```

When I evaluate, it is invalid to run the above described ckpt file directly. You need to train 30 epochs as the pre training weight to obtain a MOTA value of 43.5 on the mot20.

The eval results can be viewed in `eval/eval.log`.

# [Model Description](#contents)

## [Performance](#contents)

### FairMot on MIX dataset with detector

#### Performance parameters

| Parameters          | Ascend Standalone           | Ascend Distributed          | GPU Distributed             |
| ------------------- | --------------------------- | --------------------------- | --------------------------- |
| Model Version       | FairMotNet                  | FairMotNet                  | FairMotNet                  |
| Resource            | Ascend 910                  | 8 Ascend 910 cards          | 8x RTX 3090 24GB            |
| Uploaded Date       | 25/06/2021 (day/month/year) | 25/06/2021 (day/month/year) | 21/02/2021 (day/month/year) |
| MindSpore Version   | 1.2.0                       | 1.2.0                       | 1.5.0                       |
| Training Dataset    | MIX                         | MIX                         | MIX                         |
| Evaluation Dataset  | MOT20                       | MOT20                       | MOT20                       |
| Training Parameters | epoch=30, batch_size=4      | epoch=30, batch_size=4      | epoch=30, batch_size=12     |
| Optimizer           | Adam                        | Adam                        | Adam                        |
| Loss Function       | FocalLoss,RegLoss           | FocalLoss,RegLoss           | FocalLoss,RegLoss           |
| Train Performance   | MOTA:43.8% Prcn:90.9%       | MOTA:42.5% Prcn:91.9%%      | MOTA: 41.2%, Prcn: 90.5%    |
| Speed               | 1pc: 380.528 ms/step        | 8pc: 700.371 ms/step        | 8p: 1047 ms/step            |

# [Description of Random Situation](#contents)

We also use random seed in `src/utils/backbone_dla_conv.py` to initial network weights.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
