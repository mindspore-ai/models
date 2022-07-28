# 目录(WIP)

- [Yolact++](#Yolact++)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
    - [脚本和样例代码](#脚本和样例代码)
- [训练](#训练)
- [分布式训练](#分布式训练)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)

- [ModelZoo Homepage](#modelzoo-homepage)

# [Yolact++](#目录)

YOLACT提出的实时实例分割算法在2020年被作者扩展为YOLACT++：更好的实时实例分割。在COCO的test dev数据集上达到34.1mAP。YOLACT++在保证实时性(大于或等于30fps)的前提下，对原版的YOLACT做出几点改进，大幅提升了mAP。

在YOLACT++ 中，将ResNet的C3-C5中的各个标准3x3卷积换成3x3可变性卷积，但没有使用堆叠的可变形卷积模块，因为延迟太高。优化了Prediction Head分支，由于YOLACT是anchor-based的，所以对anchor设计进行优化。YOLACT++受MS R-CNN的启发，高质量的mask并不一定就对应着高的分类置信度，所以在模型后添加了Mask Re-Scoring分支，该分支使用YOLACT生成的裁剪后的原型mask(未作阈值化)作为输入，输出对应每个类别的GT-mask的IoU。

[论文](https://arxiv.org/abs/1912.06218): Bolya D,  Zhou C,  Xiao F, et al. YOLACT++: Better Real-time Instance Segmentation[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2020

# [数据集](#目录)

使用的数据集：[COCO2017](https://cocodataset.org/#download)

- [COCO2017](https://cocodataset.org/)是一个广泛应用的数据集，带有边框和像素级背景注释。这些注释可用于场景理解任务，如语义分割，目标检测和图像字幕制作。训练和评估的图像大小为118K和5K。
- 数据集大小：19G
    - 训练：18G，118,000个图像
    - 评估：1G，5000个图像
    - 注释：241M；包括实例、字幕、人物关键点等
- 数据格式：图像及JSON文件
    - 注：数据在dataset.py中处理。

# [环境要求](#目录)

- 硬件 Ascend
    - 使用Ascend处理器准备硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 更多关于MindSpore的信息，请查看以下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# [脚本说明](#目录)

## [脚本和样例代码](#目录)

```markdown
├── Yolact++
    ├── train.py                              // 训练脚本
    ├── eval.py                               // 评估脚本
    ├── export.py                             // 导出模型的脚本
    ├── creat_mindrecord.py                   // 云上训练先生成mindrecord，再执行train效率会有提升
    ├── README.md
    ├── requirements.txt
    ├── scripts
        ├── run_standalone_train.sh           // 单卡脚本
        ├── run_distribute_train.sh           // 多卡脚本
    ├── src
        ├── config.py                         // 参数配置文件
        ├── loss_monitor.py                   // 监视对loss训练是否正常
        ├── lr_schedule.py                    // 学习率
        ├── dataset.py                        // 创建数据集
        ├── network_define.py                 // yolact 训练网络封装器
        ├── yolact
            ├── yolactpp.py                   // 模型架构
            ├── layers
                ├── backbone_dcnV2.py         // Backbone
                ├── fpn.py                    // FPN
                ├── protonet.py               // Protonet
                ├── functions
                    ├── detection.py
                ├── modules
                    ├── loss.py           // 损失函数计算
                    ├── match.py          // 训练时计算Iou与标签匹配
            ├── utils
                ├── functions.py               // 生成网络
                ├── interpolate.py
                ├── ms_box_utils.py           // 验证时计算Box Iou与编码和解码

```

# [训练](#目录)

- 运行`run_standalone_train.sh`开始Yolact++模型的非分布式训练。

```bash
bash run_standalone_train.sh DEVICE_ID
```

# [分布式训练](#目录)

- 运行`run_distribute_train.sh`开始Yolact++模型的分布式训练。

```bash
bash run_distribute_train.sh RANK_TABLE_FILE DEVICE_NUMS
```

# [Model Description](#目录)

## [Performance](#目录)

### Training Performance

| Parameters          |                                                              |
| ------------------- | ------------------------------------------------------------ |
| Model Version       | Yolact++                                                     |
| Resource            | CentOs 8.2; Ascend 910; CPU 2.60GHz, 192cores; Memory 755G   |
| MindSpore Version   | 1.3.0                                                        |
| Dataset             | COCO2017                                                     |
| Training Parameters | epoch = 300,  batch_size = 8                                 |
| Optimizer           | Momentum                                                     |
| Loss Function       | semantic_segmentation_loss, ohem_conf_loss, mask_iou_loss, lincomb_mask_loss |
| outputs             | super-resolution pictures                                    |
| Accuracy            | 0                                                            |
| Speed               | 1pc(Ascend): 10000 ms/step                                   |
| Total time          | 1pc: 270days                                                 |
| Scripts             | [Yolact++ script](https://gitee.com/mindspore/models/tree/master/research/cv/Yolact++) |

# [ModelZoo Homepage](#目录)

Please check the official [homepage](https://gitee.com/mindspore/models).

