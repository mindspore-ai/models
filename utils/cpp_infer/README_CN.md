## 简介

MindSpore导出的MindIR，当C++推理后端为MindSpore时，支持部署在Ascend310、Ascend310P，当C++推理后端为MindSpore Lite时，支持部署在Ascend310、Ascend310P、Nvidia GPU、CPU。

为了确保脚本能够正常地运行，需要在执行推理前设置环境变量。

## Ascend硬件后端环境变量

### 确认run包安装路径

若使用root用户完成run包安装，默认路径为'/usr/local/Ascend'，非root用户的默认安装路径为'/home/HwHiAiUser/Ascend'。

以root用户的路径为例，设置环境变量如下：

```bash
export ASCEND_HOME=/usr/local/Ascend  # the root directory of run package
```

### 区分run包版本

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

## Nvidia GPU硬件后端环境变量

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

## 设置Host侧日志级别

```bash
export GLOG_v=2 # 0-DEBUG, 1-INFO, 2-WARNING, 3-ERROR, 4-CRITICAL, default level is WARNING.
```

## 推理后端

编译脚本如果识别存在环境变量`MS_LITE_HOME`，则会使用MindSpore Lite作为推理后端，否则将使用MidSpore whl包作为推理后端。

### 使用MindSpore Lite作为推理后端

MindSpore Lite推理后端支持Ascend310、Ascend310P、GPU、CPU硬件后端。

从[官网](https://mindspore.cn/versions)下载MindSpore Lite Ascend、GPU、CPU三合一tar包，解压缩后，设置`MS_LITE_HOME`环境变量为解压缩的路径，比如：

```bash
export MS_LITE_HOME=$some_path/mindpsore-lite-2.0.0-linux-x64
```

### 使用MindSpore作为推理后端

MindSpore推理后端支持Ascend310、Ascend310P硬件后端。

从[官网](https://mindspore.cn/versions)下载MindSpore 三合一或者对应硬件环境whl包，并安装，C++编译脚本会识别MindSpore whl的安装路径作为依赖的头文件和动态链接库所在路径。

## 编译环境和运行推理

在每个模型脚本目录，比如`official/cv/resnet`，一般会存在`cpp_infer`目录，其中`build.sh`用于编译C++推理。C++推理脚本依赖最上层`utils/cpp_infer/common_inc`目录下的头文件，用户如果将模型脚本目录拷贝到其他目录编译，则需要将`utils/cpp_infer/common_inc`目录拷贝到`cpp_infer`目录。

以resnet为例：

```bash
cp -r utils/cpp_infer/common_inc official/cv/resnet/cpp_infer/
cd official/cv/resnet/cpp_infer/src/
bash build.sh
```

在每个模型脚本目录，比如`official/cv/resnet`，一般会存在`scripts`目录，其中`run_infer_cpp.sh`用于推理评估并计算精度。通过设置环境变量`DEVICE_TYPE`设置运行的推理后端，当前支持`"Ascend"`、`"GPU"`、`"CPU"`三个选项。
