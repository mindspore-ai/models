# Cvtmodel说明与要求

## Cvtmodel介绍

Cvtmodel文件夹中包含的模型，均是通过Mindconverter直接转换Pytorch所提供的pth模型，并实现了对该网络的Ascend 310推理。

## Cvtmodel使用流程

### 模型下载

以densenet系列网络为例， 首先需要在pytorch提供网页上找到模型的下载方式，按照指南将pth模型文件下载并转化为onnx模型文件。(参考https://pytorch.org/hub/pytorch_vision_densenet/)

### 模型转换

在成功获得onnx文件以后，使用Mindconverter对onnx进行转化，示例如下。

'''shell
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp
mindconverter --model_file ./onnx/densenet-121.onnx --shape 1,3,224,224 --input_nodes images --output_nodes output --output ./
'''

### 模型推理

Mindconverter转换出ckpt文件后，再使用export.py脚本导出MINDIR文件，根据模型文件夹内readme所写的指令进行Ascend 310推理即可。

## 环境要求

- 硬件（Ascend/GPU）
- 准备Ascend或GPU处理器搭建硬件环境。
- 框架
- [MindSpore](https://www.mindspore.cn/install)
- [MindInsight] (https://www.mindspore.cn/mindinsight/docs/zh-CN/master/mindinsight_install.html)
- 如需查看详情，请参见如下资源：
- [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
- [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)
- [MindConverter教程] (https://www.mindspore.cn/mindinsight/docs/zh-CN/master/migrate_3rd_scripts_mindconverter.html)