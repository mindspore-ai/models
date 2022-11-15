# Contents

- [HED Description](#hed-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Parameter configuration](#parameter-configuration)
    - [Training Process](#training-process)
        - [Training](#training)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
        - [Evaluation Performance](#evaluation-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

## [HED Description](#contents)

HED, a new edge detection algorithm that tackles two important issues in this long-standing vision problem: (1) holistic image training and prediction; and (2) multi-scale and multi-level feature learning. Our proposed method, holistically-nested edge detection (HED), performs image-to-image prediction by means of a deep learning model that leverages fully convolutional neural networks and deeply-supervised nets. HED automatically learns rich hierarchical representations (guided by deep supervision on side responses) that are important in order to approach the human ability resolve the challenging ambiguity in edge and object boundary detection. We significantly advance the state-of-the-art on the BSD500 dataset (ODS F-score of .782) and the NYU Depth dataset (ODS F-score of .746), and do so with an improved speed (0.4 second per image) that is orders of magnitude faster than some recent CNN-based edge detection algorithms.

[Paper](): Saining Xie, Zhuowen Tu. Holistically-Nested Edge Detection. arXiv preprint arXiv:1504.06375, 2015.(https://arxiv.org/abs/1504.06375)

## [Model Architecture](#contents)

HED network is based on vgg19 as the backbone, which is mainly composed of several basic modules (including convolution and pool layer) and deconvolution layer.

## [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

### Dataset used: [BSDS500](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html)

- Dataset size: ~81M, 500 colorful images
    - Train: 200 images
    - Val: 100 images
    - Test: 200 images
    - Data format: RGB images
    - Note: Data will be processed in src/dataset.py

## [Features](#contents)

## [Environment Requirements](#contents)

- Hardware（Ascend）
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

## [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- Running on Ascend

For distributed training, a hccl configuration file with JSON format needs to be created in advance.
Please follow the instructions in the link below:
<https://gitee.com/mindspore/models/tree/master/utils/hccl_tools>

- Running on [ModelArts](https://support.huaweicloud.com/modelarts/)

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```text
├── cv
    ├── hed
        ├── README.md                             // descriptions about hed
        ├── ascend310_infer
        ├── ├── CMakeLists.txt                    // CMake
        ├── ├── build.sh                          // shell file of build
        ├── ├── inc
        ├── ├── ├── utils.h                       // utils.h
        ├── ├── src
        ├── ├── ├── main.cc                       // main file
        ├── ├── ├── utils.cc                      // utils.cc
        ├── scripts
        │   ├── run_infer_310.sh                  // shell script for infer on Ascend 310
            ├── run_single_train.sh               // shell script for training
            ├── run_single_eval.sh                // shell script for eval
        ├── src
        │   ├── model_utils
        │   │   ├── __init__.py                   // init file
        │   │   ├── config.py                     // Parse arguments
        │   │   ├── device_adapter.py             // Device adapter for ModelArts
        │   │   ├── local_adapter.py              // Local adapter
        │   │   ├── moxing_adapter.py             // Moxing adapter for ModelArts
        │   ├── impl
        │   │   ├── __init__.py                   // init file
        │   │   ├── bwmorph_thin.py               // bwmorph thin
        │   │   ├── correspond_pixels.py          // correspond pixels
        │   │   ├── edges_eval_dir.py             // edges eval dir
        │   │   ├── edges_eval_plot.py            // edges eval plot
        │   │   ├── toolbox.py                    // box tool
        │   ├── dataset.py                        // creating dataset
            ├── loss.py                           // loss
            ├── lr_schedule.py                    // learning rate set
        │   ├── eval_edge.py                      // edge eval
        │   ├── model.py                          // hed network define
        │   ├── nms_process.py                    // nms process
        ├── eval.py                               // evaluation script
        ├── export.py                             // export script
        ├── postprocess.py                        // postprocess script
        ├── preprocess.py                         // preprocess script
        ├── default_config.yaml                   // Configurations
        ├── train.py                              // train script
        ├── prepare.py                            // prepare script
        ├── requirements.txt                      // requirements
```

### [Script Parameters](#contents)

#### Training

#### Evaluation

### [Parameter configuration](#contents)

Parameters for both training and evaluation can be set in default_config_910/default_config_gpu.

### [Training Process](#contents)

#### Training

- vgg_ckpt_path

the ckpt used in HED is pretrained vgg16 model provided by HUAWEI.(https://download.mindspore.cn/model_zoo/r1.2/vgg16_ascend_v120_imagenet2012_official_cv_bs32_acc73/)

##### Run HED on Ascend

- Training using single device(1p), using BSDS500 dataset in default

```bash
# when using BSDS500 dataset
python prepare.py --config_path [CONFIG_PATH]
bash scripts/run_single_train.sh [DEVICE_ID] [CONFIG_PATH]
```

- `CONFIG_PATH` is the absolute path of config file
- `DEVICE_ID` is the ID of device

> **Attention** This will bind the processor cores according to the `device_num` and total processor numbers. If you don't expect to run pretraining with binding processor cores, remove the operations about `taskset` in `scripts/run_distribute_train.sh`

```text
# training process
epoch: 497 step: 20, loss is 0.10391282
2022-04-15 20:44:31.396521 end epoch 497 1.7445739e-06 0.10391282
epoch: 498 step: 20, loss is 0.07053816
2022-04-15 20:44:42.065112 end epoch 498 1.6104921e-06 0.07053816
epoch: 499 step: 20, loss is 0.09414602
2022-04-15 20:44:52.756103 end epoch 499 1.5289875e-06 0.09414602
epoch: 500 step: 20, loss is 0.09735751
2022-04-15 20:45:04.091150 end epoch 500 1.5000658e-06 0.09735751
```

##### Run HED on GPU

- Training using single device(1p), using BSDS500 dataset in default

```bash
# when using BSDS500 dataset
bash scripts/run_single_train.sh [DEVICE_ID] [CONFIG_PATH]
```

- `CONFIG_PATH` is the absolute path of config file
- `DEVICE_ID` is the ID of device

> **Attention** This will bind the processor cores according to the `device_num` and total processor numbers. If you don't expect to run pretraining with binding processor cores, remove the operations about `taskset` in `scripts/run_distribute_train.sh`

```bash
# training process
epoch: 2 step: 20, loss is 0.67871356
epoch time: 4363.766, per step time: 218.188
2022-04-20 11:41:23.881890 end epoch 2 0.0015 0.67871356
epoch: 3 step: 20, loss is 0.4893721
epoch time: 4466.146, per step time: 223.307
2022-04-20 11:41:28.348629 end epoch 3 0.0015 0.4893721
epoch: 4 step: 20, loss is 0.46001524
epoch time: 4358.786, per step time: 217.939
2022-04-20 11:41:32.707885 end epoch 4 0.0015 0.46001524
epoch: 5 step: 20, loss is 0.28899446
epoch time: 4635.880, per step time: 231.794
2022-04-20 11:41:37.344164 end epoch 5 0.0015 0.28899446
```

### [Evaluation Process](#contents)

#### Evaluation

- Do eval as follows, need to specify dataset type as "BSDS500"

```bash
# when using BSDS500 dataset
Before doing eval, please make the preparations according to ./scripts/README.md
cd src/cxx/src
source build.sh
cd ..
cd ..
cd ..
python prepare.py --config_path [CONFIG_PATH]
bash scripts/run_single_eval.sh [DEVICE_ID] [CONFIG_PATH]
```

- `CONFIG_PATH` is the absolute path of config file
- `DEVICE_ID` is the ID of device

- The above python command will run in the background, you can view the results through the file `eval.log`. You will get the accuracy as following:

```text
# grep "ODS=" output.eval.log
ODS=0.773, OIS=0.793, AP=0.800, R50=0.906 - HED
```

## Inference Process

### [Export MindIR](#contents)

```shell
python prepare.py --config_path [CONFIG_PATH]
python export.py --config_path [CONFIG_PATH] --file_format [FILE_FORMAT]
```

- `CONFIG_PATH` is the absolute path of config file
- `FILE_FORMAT` is chosen from ["AIR", "MINDIR"]

### Infer on Ascend310

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

Before performing inference, the mindir file must be exported by `export.py` script. We only provide an example of inference using MINDIR model.
Current batch_Size for BSDS500 dataset can only be set to 1.

```shell
Before performing following operations, please refer to /scripts/README.md.
cd src/cxx/src
source build.sh
cd ..
cd ..
cd ..
cd scripts/
bash run_infer_310.sh [MINDIR_PATH] [NEED_PREPROCESS] [DEVICE_ID] [CONFIG_PATH]
```

- `MINDIR_PATH` means the path of mindir file.
- `NEED_PREPROCESS` means whether need preprocess or not, it's value is 'y' or 'n', if you choose y, the dataset will be processed in bin format. Default is 'y'.
- `DEVICE_ID` is optional, default value is 0.

### result

Inference result is saved in current path, you can find result like this in acc.log file.

```text
ODS=0.773, OIS=0.793, AP=0.800, R50=0.906 - HED
```

## [Model Description](#contents)

### [Performance](#contents)

#### Training Performance

| Parameters                 | Hed(Ascend)                                                 |
| -------------------------- | ----------------------------------------------------------- |
| Model Version              | V1                                                          |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8 |
| MindSpore Version          | 1.5.0                                                       |
| Dataset                    | BSDS500                                                     |
| Training Parameters        | epoch=500, batch_size=10                                    |
| Optimizer                  | SGD                                                         |
| Loss Function              | BinaryCrossEntropyLoss                                      |
| Speed                      | 1pc: 810 ms/step                 |

#### Evaluation Performance

| Parameters          | HED(Ascend)               | HED(GPU)                  |
| ------------------- | ------------------------- | ------------------------- |
| Model Version       | HED                       | HED                       |
| Resource            | Ascend                    | GPU                       |
| Uploaded Date       | 30/12/2021                | 15/4/2022                 |
| MindSpore Version   | 1.5.0                     | 1.5.0                     |
| Dataset             | BSDS500, 500 images       | BSDS500, 500 images       |
| batch_size          | 10                        | 10                        |
| outputs             | ODS                       | ODS                       |
| Accuracy            | 77.3%                     | 77.3%                     |

## [Description of Random Situation](#contents)

In dataset.py, we set the seed inside "create_dataset" function. We also use random seed in train.py.

## [ModelZoo Homepage](#contents)  

Please check the official [homepage](https://gitee.com/mindspore/models).  
