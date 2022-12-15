# MindSpore C++推理部署指南

## 简介

本教程针对MindSpore导出的MindIR使用C++进行推理部署的场景。

当运行环境是MindSpore时，支持部署在Ascend310、Ascend310P环境上；当运行环境是MindSpore Lite时，支持部署在Ascend310、Ascend310P、Nvidia GPU、CPU环境上。

## 环境安装

MindSpore目前支持两种使用C++进行推理的运行环境，一种是直接使用MindSpore310的安装包部署运行环境，一种是使用MindSpore Lite进行推理环境部署。

两种推理环境通过`MS_LITE_HOME`识别是否使用MindSpore Lite，编译脚本如果识别存在环境变量`MS_LITE_HOME`，则会使用MindSpore Lite作为推理后端，否则将使用MindSpore whl包作为推理后端。

### MindSpore310

MindSpore310支持Ascend310、Ascend310P硬件环境。

从[官网](https://mindspore.cn/versions)下载MindSpore三合一或者对应硬件环境whl包，并安装。

### 使用MindSpore Lite作为推理后端

MindSpore Lite推理后端支持Ascend310、Ascend310P、GPU、CPU硬件后端。

从[官网](https://mindspore.cn/versions)下载MindSpore Lite Ascend、GPU、CPU三合一tar包，解压缩后，设置`MS_LITE_HOME`环境变量为解压缩的路径，比如：

```bash
export MS_LITE_HOME=$some_path/mindspore-lite-2.0.0-linux-x64
```

## 环境变量

**为了确保脚本能够正常地运行，需要在执行推理前设置环境变量。**

### Ascend硬件后端环境变量

#### 确认run包安装路径

若使用root用户完成run包安装，默认路径为'/usr/local/Ascend'，非root用户的默认安装路径为'/home/HwHiAiUser/Ascend'。

以root用户的路径为例，设置环境变量如下：

```bash
export ASCEND_HOME=/usr/local/Ascend  # the root directory of run package
```

#### 区分run包版本

run包分为2个版本，用安装目录下是否存在'ascend-toolkit'文件夹进行区分。

如果存在'ascend-toolkit'文件夹，设置环境变量如下：

```bash
export ASCEND_HOME=/usr/local/Ascend
export PATH=${ASCEND_HOME}/ascend-toolkit/latest/compiler/bin:${ASCEND_HOME}/ascend-toolkit/latest/compiler/ccec_compiler/bin/:${PATH}
export LD_LIBRARY_PATH=${ASCEND_HOME}/driver/lib64:${ASCEND_HOME}/ascend-toolkit/latest/lib64:${LD_LIBRARY_PATH}
export ASCEND_OPP_PATH=${ASCEND_HOME}/ascend-toolkit/latest/opp
export ASCEND_AICPU_PATH=${ASCEND_HOME}/ascend-toolkit/latest/
export PYTHONPATH=${ASCEND_HOME}/ascend-toolkit/latest/compiler/python/site-packages:${PYTHONPATH}
export TOOLCHAIN_HOME=${ASCEND_HOME}/ascend-toolkit/latest/toolkit
```

若不存在，设置环境变量为：

```bash
export ASCEND_HOME=/usr/local/Ascend
export PATH=${ASCEND_HOME}/latest/compiler/bin:${ASCEND_HOME}/latest/compiler/ccec_compiler/bin:${PATH}
export LD_LIBRARY_PATH=${ASCEND_HOME}/driver/lib64:${ASCEND_HOME}/latest/lib64:${LD_LIBRARY_PATH}
export ASCEND_OPP_PATH=${ASCEND_HOME}/latest/opp
export ASCEND_AICPU_PATH=${ASCEND_HOME}/latest
export PYTHONPATH=${ASCEND_HOME}/latest/compiler/python/site-packages:${PYTHONPATH}
export TOOLCHAIN_HOME=${ASCEND_HOME}/latest/toolkit
```

### Nvidia GPU硬件后端环境变量

硬件后端为Nvidia GPU时，推理依赖cuda和TensorRT，用户需要先安装cuda和TensorRT。

以下以cuda11.1和TensorRT8.2.5.1为例，用户需要根据实际安装路径设置环境变量。

```bash
export CUDA_HOME=/usr/local/cuda-11.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export TENSORRT_PATH=/usr/local/TensorRT-8.2.5.1
export PATH=$TENSORRT_PATH/bin:$PATH
export LD_LIBRARY_PATH=$TENSORRT_PATH/lib:$LD_LIBRARY_PATH
```

### 设置Host侧日志级别

```bash
export GLOG_v=2 # 0-DEBUG, 1-INFO, 2-WARNING, 3-ERROR, 4-CRITICAL, default level is WARNING.
```

## 推理流程

一个典型的推理流程包括：

- 导出MindIR文件
- 数据前处理（可选）
- 推理模型编译及执行
- 推理结果后处理

整个过程可以参考[resnet](https://gitee.com/mindspore/models/tree/master/official/cv/ResNet#%E6%8E%A8%E7%90%86%E8%BF%87%E7%A8%8B)（使用C++进行数据处理）和
[DBNet](https://gitee.com/mindspore/models/tree/master/official/cv/DBNet#%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86)（使用python进行数据处理并转成bin文件）。

### 导出MindIR文件

MindSpore提供了云侧（训练）和端侧（推理）统一的中间表示（Intermediate Representation，[IR](https://www.mindspore.cn/docs/zh-CN/master/design/mindir.html)）。可使用export接口直接将模型保存为MindIR。

```python
import mindspore as ms
from src.model_utils.config import config
from src.model_utils.env import init_env
# 环境初始化
init_env(config)
# 推理模型
net = Net()
# 加载参数
ms.load_checkpoint("xxx.ckpt", net)
# 构造输入，只需要构造输入的shape和type就行
inp = ms.ops.ones((1, 3, 224, 224), ms.float32)
# 模型导出，file_format支持'MINDIR', 'ONNX' 和 'AIR'
ms.export(net, ms.export(net, inp, file_name=config.file_name, file_format=config.file_format))
# 当模型是多输入时
# inputs = [inp1, inp2, inp3]
# ms.export(net, ms.export(net, *inputs, file_name=config.file_name, file_format=config.file_format))
```

### 数据前处理(可选)

有一些数据处理在C++侧比较难实现，可以先将数据保存成bin文件。

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

将会在`config.input_dir`目录下生成输入的bin文件。

### 推理模型开发

这里涉及C++工程的创建及编译流程。一般一个C++推理工程目录结构如下：

```text
└─cpp_infer
    ├─build.sh                # C++ 编译脚本
    ├─CmakeLists.txt          # Cmake配置文件
    ├─main.cc                 # 模型执行脚本
    └─common_inc              # 公共头文件
```

其中`build.sh`，`CmakeLists.txt`，`common_inc`一般是不用变的，在`example`目录下面提供了一份通用的脚本。

在开发一个新的模型时，需要把这几个文件拷到执行目录下，并编写对应模型的`main.cc`执行。

**有的模型`cpp_infer`下面没有`common_inc`文件夹，在执行的时候需要将这个文件夹拷到`main.cc`同目录下**

`main.cc`里面是做模型推理的脚本文件，一般包括：

- 模型加载及构建
- 数据集构建/bin文件加载
- 模型推理
- 推理结果保存

具体请参考[官网介绍](https://www.mindspore.cn/tutorials/experts/zh-CN/r1.9/infer/ascend_310_mindir.html)，以及已实现的推理模型实现，比如[resnet C++推理](https://gitee.com/mindspore/models/blob/master/official/cv/ResNet/cpp_infer/src/main.cc)。

### 推理模型编译及执行

一般我们会在scripts目录下写一个`run_infer_cpp.sh`的文件将整个推理流程串起来。详情请参考[resnet](https://gitee.com/mindspore/models/blob/master/official/cv/ResNet/scripts/run_infer_cpp.sh)。
