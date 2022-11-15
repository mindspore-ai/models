## 目录

- [目录](#目录)
    - [PAGE-Net描述](#PAGE-Net描述)
    - [模型架构](#模型架构)
    - [数据集](#数据集)
        - [数据集配置](#数据集配置)
    - [环境要求](#环境要求)
    - [脚本说明](#脚本说明)
        - [代码文件说明](#代码文件说明)
        - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [训练](#训练)
        - [分布式训练](#分布式训练)
    - [评估过程](#评估过程)
    - [导出过程](#导出过程)
    - [模型描述](#模型描述)
        - [评估性能](#评估性能)
        - [推理性能](#推理性能)
    - [ModelZoo主页](#modelzoo主页)

## PAGE-Net描述

PAGE-Net是通过监督学习解决显著性目标检测问题，它由提取特征的骨干网络模块，金字塔注意力模块和显著性边缘检测模块三部分构成。作者通过融合不同分辨率的显著性信息使得到的特征有更大的感受野和更好的表达能力，同时显著性边缘检测模块获得的边缘信息也能更加精确的分割显著性物体的边缘部分，从而使检测的结果更加精确。与其他19个工作在6个数据集上通过3种评价指标进行评估表明，PAGE-Net有着更加优异的性能和有竞争力的结果。

[PAGE-Net的tensorflow-keras源码](https://github.com/wenguanwang/PAGE-Net)，由论文作者提供。具体包含运行文件、模型文件，此外还有数据集，预训练模型的获取途径。

[论文](https://www.researchgate.net/publication/332751907_Salient_Object_Detection_With_Pyramid_Attention_and_Salient_Edges)：Wang W ,  Zhao S ,  Shen J , et al. Salient Object Detection With Pyramid Attention and Salient Edges[C]// CVPR19. 2019.

## 模型架构

PAGE-Net网络由三个部分组成，提取特征的CNN模块，金字塔注意力模块和边缘检测模块。预处理后的输入图片通过降采样输出特征信息，与此同时，对每一层的特征通过金字塔注意力模块生成更好表达力的特征，然后将边缘信息与不同深度提取出来的多尺度特征进行融合，最终输出了一张融合后的显著性检测图像。

## 数据集

数据集统一放在一个目录

### 数据集配置

数据集目录修改在config.py中，训练集变量为train_dataset_imgs,train_dataset_gts,train_dataset_edges, vgg_init
测试集路径请自行修改
测试集若要使用自己的数据集，请添加数据集路径，并在train.py中添加新增的数据集

- 训练集：

  [THUS10K数据集]([MSRA10K Salient Object Database – 程明明个人主页 (mmcheng.net)](https://mmcheng.net/msra10k/)) , 342MB，共有10000张带有标签的图像

- 测试集：

  [ECSSD数据集](https://gitee.com/link?target=http%3A%2F%2Fwww.cse.cuhk.edu.hk%2Fleojia%2Fprojects%2Fhsaliency%2Fdata%2FECSSD%2Fimages.zip%EF%BC%8Chttp%3A%2F%2Fwww.cse.cuhk.edu.hk%2Fleojia%2Fprojects%2Fhsaliency%2Fdata%2FECSSD%2Fground_truth_mask.zip)，67.2MB，共1000张

  [DUTS-OMRON数据集](https://gitee.com/link?target=http%3A%2F%2Fsaliencydetection.net%2Fdut-omron%2F)，113MB，共5163张

  [HKU-IS数据集](https://gitee.com/link?target=https%3A%2F%2Fi.cs.hku.hk%2F~gbli%2Fdeep_saliency.html)，899MB，共4447张

  [SOD数据集](https://gitee.com/link?target=https%3A%2F%2Fwww.elderlab.yorku.ca%2F%3Fsmd_process_download%3D1%26download_id%3D8285)，6.49MB，共300张

  [DUTS-TE数据集](https://gitee.com/link?target=http%3A%2F%2Fsaliencydetection.net%2Fduts%2Fdownload%2FDUTS-TE.zip)，132MB，共5019张

## 环境要求

- 硬件（CPU/GPU/Ascend）

- 如需查看详情，请参见如下资源：

  [MindSpore教程](https://gitee.com/link?target=https%3A%2F%2Fwww.mindspore.cn%2Ftutorials%2Fzh-CN%2Fmaster%2Findex.html)
  [MindSpore Python API](https://gitee.com/link?target=https%3A%2F%2Fwww.mindspore.cn%2Fdocs%2Fapi%2Fzh-CN%2Fmaster%2Findex.html)

- 需要的包

  Mindspore  1.5.0

## 脚本说明

### 代码文件说明

```markdown
├── model_zoo
    ├── PAGENet
        ├── dataset
        │   ├── train_dataset                #训练集
        │   ├── test_dataset                 #测试集
        ├── README.md                        # README文件
        ├── default_config_ascend.yaml       # 参数配置脚本文件(ascned)
        ├── default_config_gpu.yaml          # 参数配置脚本文件(gpu)
        ├── scripts
        │   ├── run_standalone_train.sh      # 单卡训练脚本文件(ascend & gpu)
        │   ├── run_distribute_train_gpu.sh  # 多卡训练脚本文件(gpu)
        │   ├── run_distribute_train.sh      # 多卡训练脚本文件(ascend)
        │   ├── run_eval.sh                  # 评估脚本文件(ascend & gpu)
        │   ├── run_infer_310.sh             # 评估脚本文件(ascend 310)
        ├── src
        |   ├── model_utils
        |   |   ├── config.py
        |   |   ├── device_adapter.py
        |   |   ├── local_adapter.py
        |   |   ├── moxing_adapter.py
        │   ├── mind_dataloader.py           # 加载数据集并进行预处理
        │   ├── pagenet.py                   # pageNet的网络结构
        │   ├── train_loss.py                # 损失定义
        |   ├── MyTrainOneStep.py            # 定义训练网络封装类
        |   ├── vgg.py                       # 定义vgg
        ├── ascend310_infer                  # 310推理
        ├── train.py                         # 训练脚本
        ├── eval.py                          # 评估脚本
        ├── export.py                        # 模型导出脚本
        ├── requirements.txt                 # 需求文档
        ├── preprocess.py                    # 预处理
        ├── postprocess.py                   # 后处理
```

### 脚本参数

```markdown
device_target: "Ascend"                                    # 运行设备 ["CPU", "GPU", "Ascend"]
batch_size: 8                                           # 训练批次大小
n_ave_grad: 10                                          # 梯度累积step数
epoch_size: 100                                          # 总计训练epoch数
image_height: 224                                      # 输入到模型的图像高度
image_width: 224                                        # 输入到模型的图像宽度
train_path: "dataset/train_dataset"                           # 训练数据集的路径
test_path: "dataset/test_dataset/"                                     # 测试数据集的根目录
model: "output/PAGENET.ckpt"                # 测试时使用的checkpoint文件,分布式训练时保存在./scripts/device/output中
```

## 训练过程

### 训练

```shell
bash scripts/run_standalone_train.sh [DEVICE_ID] [CONFIG_PATH]    #运行单卡训练,config路径默认为ascend
```

### 分布式训练

```shell
bash scripts/run_distribute_gpu.sh [DEVICE_NUM] [CONFIG_PATH]          #运行gpu分布式训练
bash scripts/run_distribute_train.sh [DEVICE_NUM] [RANK_TABLE_FILE] [CONFIG_PATH]   #运行ascend分布式训练，config路径默认为ascend
```

### 云上训练

```markdown
启动文件: train.py
运行参数：
        device_target = Ascend      #指定训练设备
        enable_modelarts = True          #启动云上训练
        train_mode = single/distribute     #单卡训练时为single，八卡训练时为distribute
        config_path = [CONFIG_PATH]       #指定训练用参数config
#数据集位置存储在环境变量data_url中，训练输出路径存储在环境变量train_url中
```

## 评估过程

```markdown
bash scripts/eval.sh [DEVICE_ID] [CONFIG_PATH] #运行推理
```

## 导出过程

```shell
python export.py --config_path=[CONFIG_PATH]  #导出mindir，模型文件路径为config中的ckpt_file
```

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

```text
成功导出模型之后，执行以下指令进行310推理
首先需修改default_config_ascend.yaml文件：
1. 修改test_img_path为推理数据集原图的路径
2. 修改test_gt_path为推理数据集MASK的路径
3. 修改batchsize为1
执行推理指令如下：
```

```bash
bash run_infer_310.sh [MINDIR_PATH] [CONFIG_PATH] [DEVICE_ID]
```

## 模型描述

### 评估性能

  THUS10K上的PAGE-Net（GPU）

| 参数          | GPU(单卡)                         | GPU（8卡）                      |
| ------------- | --------------------------------  | ------------------------------- |
| 模型          | PAGE-Net                          | PAGE-Net                        |
| 上传日期      |  2022.6.20                         |   2022.6.20                     |
| Mindspore版本 | 1.5.0                             | 1.5.0                           |
| 数据集        | THUS10K                           | THUS10K                         |
| 训练参数      | epoch=100,steps=1000,batch_size=10|epoch=200,steps=125,batch_size=10|
| 损失函数      |  MSE&BCE                          |   MSE&BCE                       |
| 优化器        |  Adam                             |      Adam                       |
| 速度          |  52s/step                       |       87s/step                 |
| 总时长        |   7h15m0s                         |      3h28m0s                      |
| 微调检查点    |    390M(.ckpt文件)                |          390M(.ckpt文件)          |

  THUS10K上的PAGE-Net（Ascend）

| 参数          | Ascend(单卡)                         | Ascend（8卡）                      |
| ------------- | --------------------------------  | ------------------------------- |
| 模型          | PAGE-Net                          | PAGE-Net                        |
| 上传日期      |  2022.6.20                         |   2022.6.20                     |
| Mindspore版本 | 1.5.0                             | 1.5.0                           |
| 数据集        | THUS10K                           | THUS10K                         |
| 训练参数      | epoch=100,steps=1250,batch_size=8|epoch=100,steps=1250,batch_size=8|
| 损失函数      |  MSE&BCE                          |   MSE&BCE                       |
| 优化器        |  Adam                             |      Adam                       |
| 速度          |  15.37ms/step                       |       48.8ms/step                 |
| 总时长        |   4h50m                         |      1h54m                     |
| 微调检查点    |    558M(.ckpt文件)                |          558M(.ckpt文件)          |

### 推理性能

显著性目标检测数据集上的PAGE-Net（GPU）

| 参数          | GPU(单卡)             | GPU（8卡）            |
| ------------- | --------------------- | ---------------------|
| 模型          | PAGE-Net              | PAGE-Net             |
| 上传日期      |  2022.6.20            |   2022.6.20          |
| Mindspore版本 | 1.5.0                 | 1.5.0                |
| 数据集        | SOD, 300张图像       | SOD, 300张图像       |
| 评估指标      |  F-score:0.593        | F-score:0.593        |
| 数据集        | ECCSD, 1000张图像     | ECCSD, 1000张图像     |
| 评估指标      | F-score: 0.845        | F-score:0.845        |
| 数据集        | DUTS-OMRON, 5163张图像| DUTS-OMRON, 5163张图像|
| 评估指标      |  F-score: 0.80        | F-score: 0.80        |
| 数据集        | HKU-IS, 4447张图像    | HKU-IS, 4447张图像    |
| 评估指标      |  F-score: 0.842       | F-score: 0.842       |
| 数据集        | DUTS-TE, 5019张图像   | DUTS-TE, 5019张图像   |
| 评估指标      | F-score: 0.778        | F-score: 0.778       |

显著性目标检测数据集上的PAGE-Net（Ascend）

| 参数          | Ascend(单卡)             | Ascend（8卡）            |
| ------------- | --------------------- | ---------------------|
| 模型          | PAGE-Net              | PAGE-Net             |
| 上传日期      |  2022.6.29            |   2022.6.29          |
| Mindspore版本 | 1.5.0                 | 1.5.0                |
| 数据集        | SOD, 300张图像       | SOD, 300张图像       |
| 评估指标      |  F-score:0.659        | F-score:0.659        |
| 数据集        | ECCSD, 1000张图像     | ECCSD, 1000张图像     |
| 评估指标      | F-score: 0.927        | F-score:0.927        |
| 数据集        | DUTS-OMRON, 5163张图像| DUTS-OMRON, 5163张图像|
| 评估指标      |  F-score: 0.874        | F-score: 0.874       |
| 数据集        | HKU-IS, 4447张图像    | HKU-IS, 4447张图像    |
| 评估指标      |  F-score: 0.935       | F-score: 0.935       |
| 数据集        | DUTS-TE, 5019张图像   | DUTS-TE, 5019张图像   |
| 评估指标      | F-score: 0.779        | F-score: 0.779       |

## ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)。
