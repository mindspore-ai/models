# MindSpore Inference with C++ Deployment Guide

## Overview

This tutorial aims at the edge scenarios inference deployment based on the MindIR model file exported by MindSpore.

When the C++ inference backend is MindSpore, the MindIR model exported from MindSpore can be deployed on Ascend310 and Ascend310P.
When the C++ inference backend is MindSpore Lite, MindIR models can be deployed on Ascend310/310P, Nvidia GPU and CPU.

## Installing MindSpore Inference Environment

MindSpore currently supports two kinds of running environments for inference with C++.

One is to deploy the running environment directly using the installation package of MindSpore310,
and the other is to deploy the inference environment using MindSpore Lite.

The running environment are realized through `MS_LITE_HOME` identify whether to use MindSpore Lite. If setting `MS_LITE_HOME`, it will compile the scripts on MindSpore Lite, otherwise it will compile the scripts on MindSpore310.

### MindSpore310

MindSpore310 support Ascend310 and Ascend310P.

From [MindSpore Install](https://mindspore.cn/versions) download MindSpore whl package or MindSpore 310 whl package to deployment reasoning development environment.

### MindSpore Lite

MindSpore Lite support Ascend310、Ascend310P、GPU、CPU.

From [MindSpore Install](https://mindspore.cn/versions) download MindSpore whl package, and MindSpore Lite tar package. After unzipping, set `MS_LITE_HOME` to the unzipped path, such as:

```bash
export MS_LITE_HOME=$some_path/mindspore-lite-2.0.0-linux-x64
```

## Environment Variables

**In order to ensure that the script running, you need to set the environment variables before executing the inference.**

### Ascend

### Identify the run package version

The run package is divided into two versions according to whether the 'ascend-toolkit' directory exists in the installation path.

If 'ascend-toolkit' directory exists, set the environment variables as follows:

```bash
export ASCEND_HOME=/usr/local/Ascend
export PATH=${ASCEND_HOME}/ascend-toolkit/latest/compiler/bin:${ASCEND_HOME}/ascend-toolkit/latest/compiler/ccec_compiler/bin/:${PATH}
export LD_LIBRARY_PATH=${ASCEND_HOME}/driver/lib64:${ASCEND_HOME}/ascend-toolkit/latest/lib64:${LD_LIBRARY_PATH}
export ASCEND_OPP_PATH=${ASCEND_HOME}/ascend-toolkit/latest/opp
export ASCEND_AICPU_PATH=${ASCEND_HOME}/ascend-toolkit/latest/
export PYTHONPATH=${ASCEND_HOME}/ascend-toolkit/latest/compiler/python/site-packages:${PYTHONPATH}
export TOOLCHAIN_HOME=${ASCEND_HOME}/ascend-toolkit/latest/toolkit
```

Another version:

```bash
export ASCEND_HOME=/usr/local/Ascend
export PATH=${ASCEND_HOME}/latest/compiler/bin:${ASCEND_HOME}/latest/compiler/ccec_compiler/bin:${PATH}
export LD_LIBRARY_PATH=${ASCEND_HOME}/driver/lib64:${ASCEND_HOME}/latest/lib64:${LD_LIBRARY_PATH}
export ASCEND_OPP_PATH=${ASCEND_HOME}/latest/opp
export ASCEND_AICPU_PATH=${ASCEND_HOME}/latest
export PYTHONPATH=${ASCEND_HOME}/latest/compiler/python/site-packages:${PYTHONPATH}
export TOOLCHAIN_HOME=${ASCEND_HOME}/latest/toolkit
```

## Environment variables on Nvidia GPU

When the hardware backend is Nvidia GPU, cuda and TensorRT are depended. You need to install cuda and TensorRT first.

The following uses cuda11.1 and TensorRT8.2.5.1 as examples. You need to set the environment variables based on the actual installation paths.

```bash
export CUDA_HOME=/usr/local/cuda-11.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export TENSORRT_PATH=/usr/local/TensorRT-8.2.5.1
export PATH=$TENSORRT_PATH/bin:$PATH
export LD_LIBRARY_PATH=$TENSORRT_PATH/lib:$LD_LIBRARY_PATH
```

## Set the Host log level

```bash
export GLOG_v=2 # 0-DEBUG, 1-INFO, 2-WARNING, 3-ERROR, 4-CRITICAL, default level is WARNING.
```

## Inference process

A typical inference process includes:

- Export MindIR
- Data pre-processing(optional)
- Inference model compilation and execution
- Inference result post-processing

Please referring to [resnet](https://gitee.com/mindspore/models/tree/master/official/cv/ResNet#%E6%8E%A8%E7%90%86%E8%BF%87%E7%A8%8B)(data processing with C++) and [DBNet](https://gitee.com/mindspore/models/tree/master/official/cv/DBNet#%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86)(data processing with python to bin file).

### Export MindIR

MindSpore provides a unified Intermediate Representation (IR) for cloud side (training) and end side (inference). Models can be saved as MindIR directly by using the export interface.

```python
import mindspore as ms
from src.model_utils.config import config
from src.model_utils.env import init_env
# Environment initialization
init_env(config)
# Inference model
net = Net()
# Load model
ms.load_checkpoint("xxx.ckpt", net)
# Construct the input, only need to set the shape and type of the input
inp = ms.ops.ones((1, 3, 224, 224), ms.float32)
# Export model, file_format support 'MINDIR', 'ONNX' and 'AIR'
ms.export(net, ms.export(net, inp, file_name=config.file_name, file_format=config.file_format))
# When using multi inputs
# inputs = [inp1, inp2, inp3]
# ms.export(net, ms.export(net, *inputs, file_name=config.file_name, file_format=config.file_format))
```

### Data pre-processing(optional)

Some data processing is difficult to implement in C++, sometime saving the data as bin files first.

```python
import os
from src.dataset import create_dataset
from src.model_utils.config import config

dataset = create_dataset(config, is_train=False)
it = dataset.create_dict_iterator(output_numpy=True)
input_dir = config.input_dir
for i,data in enumerate(it):
    input_name = "eval_input_" + str(i+1) + ".bin"
    input_path = os.path.join(input_dir, input_name)
    data['img'].tofile(input_path)
```

The input bin file is generated in the dir directory `config.input_path`.

### Inference model development

This involves the creation and compilation process of the C++ project. Generally, the directory structure of a C++ inference project is as follows:

```text
└─cpp_infer
    ├─build.sh                # C++ compilation script
    ├─CmakeLists.txt          # Cmake configuration file
    ├─main.cc                 # Model execution script
    └─common_inc              # Common header file
```

Generally, it doesn't need to change `build.sh`, `CmakeLists.txt`, `common_inc`, a general script is provided under the 'example' directory.

When developing a new model, you need to copy these files to the execution directory and write the 'main.cc' execution of the corresponding model.

**There is no 'common_inc' under execution directory in some models, you need to copy it to the same directory as 'main.cc' during execution.**

`main.cc` generally including:

- Model loading and construction
- Dataset construction / bin file loading
- Model Inference
- Inference result saving

Please refer to [MindSpore 310 infer](https://www.mindspore.cn/tutorials/experts/en/r1.9/infer/ascend_310_mindir.html), and implemented inference model, for example [resnet C++ inference](https://gitee.com/mindspore/models/blob/master/official/cv/ResNet/cpp_infer/src/main.cc).

### Inference model compilation and execution

Generally, we need to provide a `run_infer_cpp.sh` for connecting the whole inference process. For details, please refer to [resnet](https://gitee.com/mindspore/models/blob/master/official/cv/ResNet/scripts/run_infer_cpp.sh).
