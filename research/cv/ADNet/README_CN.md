# 目录

<!-- TOC -->

- [目录](#目录)
- [ADNet描述](#ADNet描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [训练](#训练)
    - [评估过程](#评估过程)
        - [评估](#评估)
    - [导出mindir模型](#导出mindir模型)
    - [推理过程](#推理过程)
        - [用法](#用法)
        - [结果](#结果)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

## ADNet描述

ADNet是2017年提出的视频目标跟踪算法,该论文发表在CVPR2017上面,相比传统的视频目标跟踪算法做到了更快,以监督学习为模型主要训练方式,并结合强化学习进行模型finetune,平均提升2个点的精度.

[论文](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yun_Action-Decision_Networks_for_CVPR_2017_paper.pdf)：Sangdoo Yun（Seoul National University, South Korea）. "Action-Decision Networks for Visual Tracking with Deep Reinforcement
Learning'. *Presented at CVPR 2017*.

## 模型架构

ADNet模型由vgg-m提供视频帧的特征提取,满足模型轻量化需求,结合历史行为信息动态预测模型的下个行为信息以及当前行为的置信度.

## 数据集

使用的数据集：[VOT2013]、[VOT2014]、[VOT2015]、[OTB100] </br>

## 环境要求

- 硬件（Ascend/ModelArts）
    - 准备Ascend或ModelArts处理器搭建硬件环境
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

## 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：</br>
train时注意dataset_path路径是包含vot13~15的根目录,test的时候dataset_path是包含basketball..等的目录 </br>
vggm.pth 预训练vggm参数 (https://data.lip6.fr/cadene/pretrainedmodels/vggm-786f2434.pth) </br>
将获取到的vggm.ckpt放在src/models/ 下,vggm.pth需要转换成vggm.ckpt

```python
# 转换vggm.pth脚本，会在运行目录下生成一个vggm.ckpt
python pth2ckpt.py --pth_path /path/pth
# 进入脚本目录，需要指定device_id,该步骤默认需要进行Reinforcement LeLearning的微调,可在SL训练后手动终断
python train.py --target_device device_id --dataset_path /path/dataset/
# 进入脚本目录，根据权重文件生成预测框文件,需要指定训练ckpt,device_id
python eval.py --dataset_path /path/otb --weight_file /path/to/weight_file --target_device device_id
# 进入脚本目录，根据生成的预测文件进行最终精度评估,bboxes_folder为上一行命令生成的预测文件夹名,一般为results_on_test_images_part2/weight_file
python create_plots.py --bboxes_folder results_on_test_images_part2/weight_file
#Ascend多卡训练
bash scripts/run_distributed_train.sh RANK_TABLE_FILE RANT_SIZE 0 /path/dataset
#Ascend单卡训练
bash scripts/run_standalone_train.sh /path/dataset DEVICE_ID
#Ascend多卡测试,需要指定weight_file
bash scripts/run_distributed_test.sh RANK_TABLE_FILE RANT_SIZE 0 weight_file /path/dataset
```

Ascend训练：生成[RANK_TABLE_FILE](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)

## 脚本说明

### 脚本及样例代码

```text
├── ADNet
    ├── scripts
    │   ├──run_distribute_train.sh          // 在Ascend中多卡训练
    │   ├──run_distribute_test.sh           // 在Ascend中多卡测试
    │   ├──run_standalone_train.sh          // 在Ascend中单卡训练
    ├── src             //源码
    │   │   ├── datasets
    │   │   │   ├──get_train_dbs.py
    │   │   │   ├──online_adaptation_dataset.py
    │   │   │   ├──rl_dataset.py
    │   │   │   ├──sl_dataset.py
    │   │   │   ├── data
    │   │   │   │   ├──otb        //OTB100数据集
    │   │   │   │   ├──vot13      //vot2013数据集
    │   │   │   │   ├──vot14      //vot2014数据集
    │   │   │   │   ├──vot15      //vot2015数据集
    │   │   ├── models
    │   │   │   ├──ADNet.py                //ADNet主干网络模型
    │   │   │   ├──CustomizedCell.py       //自定义网络结构
    │   │   │   ├──vggm.py                 //vggm网络模型
    │   │   │   ├──vggm.ckpt               //vggm预训练网络模型
    │   │   ├── options
    │   │   │   ├──general.py            //模型相关配置
    │   │   ├── trainers
    │   │   │   ├──adnet_test.py            //测试主文件
    │   │   │   ├──adnet_train_rl.py        //强化学习主文件
    │   │   │   ├──adnet_train_sl.py        //监督学习主文件
    │   │   │   ├──RL_tools.py              //强化学习环境
    │   │   ├── utils
    │   │   │   ├──augmentations.py
    │   │   │   ├──display.py
    │   │   │   ├──do_action.py
    │   │   │   ├──draw_box_from_npy.py
    │   │   │   ├──gen_action_labels.py
    │   │   │   ├──gen_samples.py
    │   │   │   ├──get_action_history_onehot.py
    │   │   │   ├──get_benchmark_info.py
    │   │   │   ├──get_benchmark_path.py         //数据集位置描述文件
    │   │   │   ├──get_train_videos.py
    │   │   │   ├──get_video_infos.py
    │   │   │   ├──my_math.py
    │   │   │   ├──overlap_ratio.py
    │   │   │   ├──precision_plot.py
    │   │   │   ├── videolist           //定义追踪文件夹
    │   │   │   │   ├── vot13-otb.txt
    │   │   │   │   ├── vot14-otb.txt
    │   │   │   │   ├── vot15-otb.txt
    ├── README_CN.md                    // ADNet相关说明
    ├── train.py                        // 训练入口
    ├── eval.py                         // 评估入口
    ├── create_plots.py                 // 精度生成脚本

```

### 脚本参数

```text
train.py
--data_url: obs桶数据集位置,vot13,vot14,vot15
--train_url: 输出文件路径
--start_iter: 起始iteration数
--save_folder: 权重文件保存的相对路径
--device_target: 实现代码的设备,值为'Ascend'
--target_device: 使用的物理卡号
--resume: 恢复训练保存ckpt的路径
--run_supervised: 是否启用SL,或直接启用RL,需传入resume的ckpt路径
--distributed: 多卡运行
--run_online: ModelArts上运行,默认为False
eval.py
--weight_file: 权重文件路径
--save_result_npy: 保存所有预测文件的相对路径的根目录
--device_target: 实现代码的设备，值为'Ascend'
--target_device: 使用的物理卡号
--data_url: obs桶数据集位置
--train_url: 输出文件路径
create_plots.py
--bboxes_folder 运行eval.py所指定的save_result_npy中对应权重文件目录
```

### 训练过程

#### 训练

- Ascend处理器环境运行

  ```python
  python train.py --target_device device_id
  # 或进入脚本目录，执行脚本 单卡训练时间过长,不建议使用单卡训练,8卡监督训练时间大概需要80h(30epoch),强化学习部分不建议在进行训练,精度也可达标
  bash scripts/run_standalone_train.sh DEVICE_ID
  ```

  经过训练后，损失值如下：

  ```text
  # grep "Loss:" log
  iter 970 || Loss: 2.4038 || Timer: 2.5797 sec.
  iter 980 || Loss: 2.2499 || Timer: 2.4897 sec.
  iter 990 || Loss: 2.4569 || Timer: 2.4808 sec.
  iter 1000 || Loss: 2.5012 || Timer: 2.4311 sec.
  iter 1010 || Loss: 2.3282 || Timer: 2.5438 sec.
  iter 1020 || Loss: 2.0806 || Timer: 2.4931 sec.
  iter 1030 || Loss: 2.3262 || Timer: 2.6490 sec.
  iter 1040 || Loss: 2.2101 || Timer: 2.4939 sec.
  iter 1050 || Loss: 2.3560 || Timer: 2.4301 sec.
  iter 1060 || Loss: 0.8712 || Timer: 2.5953 sec.
  iter 1070 || Loss: 2.3375 || Timer: 2.4974 sec.
  iter 1080 || Loss: 1.3731 || Timer: 2.4519 sec
  ...
  ```

  模型检查点保存在weights目录下,多卡训练时仅rank为0的卡保存检查点

### 评估过程

#### 评估

在运行以下命令之前，请检查用于评估的检查点路径

- Ascend处理器环境运行

  ```python
  # 进入脚本目录，根据OTB100数据集online finetune and test生成预测文件,该步骤单卡情况下大约要执行17个小时左右
    python eval.py --weight_file /path/weight_file
  # 进入脚本目录，根据生成的预测文件绘制distance等metrics图,该步骤执行会生成对应的精度
    python create_plots.pt --bboxes_folder /path
  ```

-
  测试数据集的准确率如下：
作者目标仓库精度75.3%,70.7%,69.0%,68.7%,75.5%,69.4%,avg precision=71.3%
实际测试的SL精度为73.6%左右

# 推理

本模型支持导出静态mindir，但静态推理效果无法接受，故暂不提供推理流程

## [导出mindir模型](#contents)

```python
python export.py --device_id [DEVICE_ID] --ckpt_file [CKPT_PATH]
```

## [推理过程](#contents)

### 用法

mindir文件必须由export.py导出，输入文件必须为bin格式

### 结果

## 模型描述

### 性能

#### 评估性能

| 参数 | ModelArts
| -------------------------- | -----------------------------------------------------------
| 资源 | Ascend 910；CPU 2.60GHz, 192核；内存：755G
| 上传日期 | 2021-08-12
| MindSpore版本 | 1.3.0
| 数据集 | VOT2013,VOT2014,VOT2015
| 训练参数 | epoch=100, batch_size=8, lr=0.0001
| 优化器 | SGD
| 损失函数 | SoftmaxCrossEntropyWithLogits
| 损失 | 0.03
| 速度 | 200毫秒/步
| 总时间 | 10分钟
| 微调检查点 | 大约40M （.ckpt文件）

## 随机情况说明

train.py中设置了随机种子

## ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)