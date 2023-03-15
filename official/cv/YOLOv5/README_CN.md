# 目录

- [目录](#目录)
- [YOLOv5说明](#yolov5说明)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [训练](#训练)
        - [分布式训练](#分布式训练)
    - [推理过程](#推理过程)
        - [导出MindIR](#导出mindir)
        - [在Ascend 310上进行推理](#在ascend-310上进行推理)
        - [结果](#结果)
        - [导出ONNX](#导出onnx)
        - [运行ONNX评估](#运行onnx评估)
        - [结果](#结果-1)
- [模型说明](#模型说明)
- [性能](#性能)
    - [评估性能](#评估性能)
    - [推理性能](#推理性能)
    - [迁移学习](#迁移学习)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

# [YOLOv5描述](#目录)

YOLOv5发布于2020年4月，利用COCO数据集实现了最先进的目标检测性能。这是继YoloV3之后的一个重要改进，在网络**骨干**部署新架构，并且网络**颈部**的修改使**mAP**（平均精度均值）提高了**10%**，**FPS**（每秒帧数）提高了**12%**。

[代码](https://github.com/ultralytics/yolov5)

# [模型架构](#目录)

YOLOv5主要组成：CSP结构和Focus结构作为骨干、空间金字塔池化（SPP）作为附加模块、PANet路径聚合作为颈部、YOLOv3作为头部。[CSP](https://arxiv.org/abs/1911.11929)是一个新的骨干网络，可以增强CNN的学习能力。在CSP上添加[空间金字塔池化](https://arxiv.org/abs/1406.4729)模块来增加更多可接受空间，并分离出最重要的上下文特征。不同级别检测器使用PANet聚合参数，而非在YOLOv3中使用的用于对象检测的特征金字塔网络（FPN）。具体来说，CSPDarknet53包含5个CSP模块，这些模块使用的卷积C的内核大小k=3x3，步长s=2x2；在PANet和SPP中，使用的最大池化层为1x1、5x5、9x9、13x13。

# [数据集](#目录)

使用的数据集：[COCOCO2017](<https://cocodataset.org/#download>)

注：您可以使用**COCO2017**或与MS COCO标注格式相同的数据集运行脚本。但建议您使用MS COCO数据集来运行我们的模型。

# [快速入门](#目录)

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

```bash
#通过Python在Ascend或GPU上进行训练（单卡）
python train.py \
    --device_target="Ascend" \ # Ascend或GPU
    --data_dir=xxx/dataset \
    --is_distributed=0 \
    --yolov5_version='yolov5s' \
    --lr=0.01 \
    --max_epoch=320 \
    --warmup_epochs=4 > log.txt 2>&1 &
```

```bash
# 若使用shell脚本单卡运行，请更改配置文件中的`device_target`为在Ascend/GPU上运行，并参考注释的内容，更改`T_max`、`max_epoch`、`warmup_epochs`。
bash run_standalone_train.sh [DATASET_PATH]

# 在Ascend中运行shell脚本进行分布式训练示例（8卡）
bash run_distribute_train.sh [DATASET_PATH] [RANK_TABLE_FILE]

# 在GPU运行shell脚本进行分布式训练示例（8卡）
bash run_distribute_train_gpu.sh [DATASET_PATH] [RANK_SIZE]
```

```bash
# 通过Python命令在Ascend或GPU上运行评估
python eval.py \
    --device_target="Ascend" \ # Ascend或GPU
    --data_dir=xxx/dataset \
    --yolov5_version='yolov5s' \
    --pretrained="***/*.ckpt" \
    --eval_shape=640 > log.txt 2>&1 &
```

```bash
# 通过shell脚本运行评估，请将配置文件中的`device_target`更改为在Ascend或GPU上运行。
bash run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH] [DEVICE_ID]
```

请注意，default_config.yaml是8卡上yolov5s的默认参数。Ascend和GPU上的`batchsize`和`lr`不同，请参考`scripts/run_distribute_train.sh`或`scripts/run_distribute_train_gpu.sh`中的设置。

# [脚本说明](#目录)

## [脚本及样例代码](#目录)

```text
├── model_zoo
    ├── README.md                              // 所有模型相关说明
    ├── yolov5
        ├── README.md                          // yolov5的相关说明
        ├── scripts
        │   ├──docker_start.sh                 // 运行shell脚本启动docker
        │   ├──run_distribute_train.sh         // 在Ascend中进行分布式训练（8卡）
        │   ├──run_distribute_train_gpu.sh     // 在GPU中进行分布式训练（8卡）
        │   ├──run_standalone_train.sh         // 进行单卡训练
        │   ├──run_infer_310.sh                // 在310上评估的shell脚本
        │   ├──run_eval.sh                     // 用于评估的shell脚本
        │   ├──run_eval_onnx.sh                // 用于onnx评估的shell脚本
        ├──model_utils
        │   ├──config.py                       // 参数配置
        │   ├──device_adapter.py               // 获取设备信息
        │   ├──local_adapter.py                // 获取设备信息
        │   ├──moxing_adapter.py               // 装饰器
        ├── src
        │   ├──backbone.py                     // 骨干网络
        │   ├──distributed_sampler.py          // 数据集迭代
        │   ├──initializer.py                  // 参数初始化
        │   ├──logger.py                       // 日志函数
        │   ├──loss.py                         // 损失函数
        │   ├──lr_scheduler.py                 // 生成学习率
        │   ├──transforms.py                   // 预处理数据
        │   ├──util.py                         // Util函数
        │   ├──yolo.py                         // YOLOv5网络
        │   ├──yolo_dataset.py                 // 创建YOLOv5数据集
        ├── default_config.yaml                // 参数配置（YOLOv5s 8卡）
        ├── train.py                           // 训练脚本
        ├── eval.py                            // 评估脚本
        ├── eval_onnx.py                       // ONNX评估脚本
        ├── export.py                          // 导出脚本
```

## [脚本参数](#目录)

```text
train.py中主要的参数有：

可选参数：

  --device_target       实现代码的设备。默认值：Ascend
  --data_dir            训练数据集目录
  --per_batch_size      训练的批处理大小。默认值：32（单卡），16（Ascend 8卡）或32（GPU 8卡）
  --resume_yolov5       用于微调的YoLOv5的CKPT文件。默认值：""。
  --lr_scheduler        学习率调度器。可选值：exponential或cosine_annealing
                        默认值：cosine_annealing
  --lr                  学习率。默认值：0.01（单卡），0.02（Ascend 8卡）或0.025（GPU 8卡）
  --lr_epochs           学习率变化轮次，用英文逗号（,）分割。默认值为'220,250'。
  --lr_gamma            指数级lr_scheduler系数降低学习率。默认值为0.1。
  --eta_min             cosine_annealing调度器中的eta_min。默认值为0。
  --t_max               在cosine_annealing调度器中的T-max。默认值为300（8卡）。
  --max_epoch           模型训练最大轮次。默认值为300（8卡）。
  --warmup_epochs       热身总轮次。默认值为20（8卡）。
  --weight_decay        权重衰减因子。默认值为0.0005。
  --momentum            动量参数。默认值为0.9。
  --loss_scale          静态损失缩放。默认值为64。
  --label_smooth        是否在CE中使用标签平滑。默认值为0。
  --label_smooth_factor 初始one-hot编码的平滑强度。默认值为0.1。
  --log_interval        日志记录间隔步骤。默认值为100。
  --ckpt_path           CKPT文件保存位置。默认值为outputs/。
  --is_distributed      是否进行分布式训练，1表示是，0表示否。默认值为0。
  --rank                分布式训练的本地序号。默认值为0。
  --group_size          设备的全局大小。默认值为1。
  --need_profiler       是否使用Profiler，0表示否，1表示是。默认值为0。
  --training_shape     设置固定训练shape。默认值为""。
  --resize_rate         调整多尺度训练率。默认值为10。
  --bind_cpu            分布式训练时是否绑定cpu。默认值为True。
  --device_num          每台服务器的设备数量。默认值为8。
```

## [训练过程](#目录)

### 训练

在Ascend上开始单机训练

```shell
#使用python命令进行训练（单卡）
python train.py \
    --data_dir=xxx/dataset \
    --yolov5_version='yolov5s' \
    --is_distributed=0 \
    --lr=0.01 \
    --T_max=320
    --max_epoch=320 \
    --warmup_epochs=4 \
    --per_batch_size=32 \
    --lr_scheduler=cosine_annealing > log.txt 2>&1 &
```

在GPU上进行单卡训练时，应微调参数。

上述python命令将在后台运行，您可以通过`log.txt`文件查看结果。

训练结束后，您可在默认**outputs**文件夹下找到checkpoint文件。得到如下损失值：

```text
# grep "loss:" log.txt
2021-08-06 15:30:15,798:INFO:epoch[0], iter[600], loss:296.308071, fps:44.44 imgs/sec, lr:0.00010661844862625003
2021-08-06 15:31:21,119:INFO:epoch[0], iter[700], loss:276.071959, fps:48.99 imgs/sec, lr:0.00012435863027349114
2021-08-06 15:32:26,185:INFO:epoch[0], iter[800], loss:266.955208, fps:49.18 imgs/sec, lr:0.00014209879736881703
2021-08-06 15:33:30,507:INFO:epoch[0], iter[900], loss:252.610914, fps:49.75 imgs/sec, lr:0.00015983897901605815
2021-08-06 15:34:42,176:INFO:epoch[0], iter[1000], loss:243.106683, fps:44.65 imgs/sec, lr:0.00017757914611138403
2021-08-06 15:35:47,429:INFO:epoch[0], iter[1100], loss:240.498834, fps:49.04 imgs/sec, lr:0.00019531932775862515
2021-08-06 15:36:48,945:INFO:epoch[0], iter[1200], loss:245.711473, fps:52.02 imgs/sec, lr:0.00021305949485395104
2021-08-06 15:37:51,293:INFO:epoch[0], iter[1300], loss:231.388255, fps:51.33 imgs/sec, lr:0.00023079967650119215
2021-08-06 15:38:55,680:INFO:epoch[0], iter[1400], loss:238.904242, fps:49.70 imgs/sec, lr:0.00024853984359651804
2021-08-06 15:39:57,419:INFO:epoch[0], iter[1500], loss:232.161600, fps:51.83 imgs/sec, lr:0.00026628002524375916
2021-08-06 15:41:03,808:INFO:epoch[0], iter[1600], loss:227.844698, fps:48.20 imgs/sec, lr:0.00028402020689100027
2021-08-06 15:42:06,155:INFO:epoch[0], iter[1700], loss:226.668858, fps:51.33 imgs/sec, lr:0.00030176035943441093
...
```

### 分布式训练

运行shell脚本进行分布式训练示例（8卡）

```bash
# 在Ascend环境中运行shell脚本进行分布式训练示例（8卡）
bash run_distribute_train.sh [DATASET_PATH] [RANK_TABLE_FILE]

# 在GPU运行shell脚本进行分布式训练示例（8卡）
bash run_distribute_train_gpu.sh [DATASET_PATH] [RANK_SIZE]
```

上述shell脚本将在后台运行分布式训练。您可以通过文件train_parallel[X]/log.txt(Ascend)或distribute_train/nohup.out(GPU)查看结果 得到如下损失值：

```text
# 分布式训练结果（8卡，动态shape）
...
2021-08-05 16:01:34,116:INFO:epoch[0], iter[200], loss:415.453676, fps:580.07 imgs/sec, lr:0.0002742903889156878
2021-08-05 16:01:57,588:INFO:epoch[0], iter[300], loss:273.358383, fps:545.96 imgs/sec, lr:0.00041075327317230403
2021-08-05 16:02:26,247:INFO:epoch[0], iter[400], loss:244.621502, fps:446.64 imgs/sec, lr:0.0005472161574289203
2021-08-05 16:02:55,532:INFO:epoch[0], iter[500], loss:234.524876, fps:437.10 imgs/sec, lr:0.000683679012581706
2021-08-05 16:03:25,046:INFO:epoch[0], iter[600], loss:235.185213, fps:434.08 imgs/sec, lr:0.0008201419259421527
2021-08-05 16:03:54,585:INFO:epoch[0], iter[700], loss:228.878598, fps:433.48 imgs/sec, lr:0.0009566047810949385
2021-08-05 16:04:23,932:INFO:epoch[0], iter[800], loss:219.259134, fps:436.29 imgs/sec, lr:0.0010930676944553852
2021-08-05 16:04:52,707:INFO:epoch[0], iter[900], loss:225.741833, fps:444.84 imgs/sec, lr:0.001229530549608171
2021-08-05 16:05:21,872:INFO:epoch[1], iter[1000], loss:218.811336, fps:438.91 imgs/sec, lr:0.0013659934047609568
2021-08-05 16:05:51,216:INFO:epoch[1], iter[1100], loss:219.491889, fps:436.50 imgs/sec, lr:0.0015024563763290644
2021-08-05 16:06:20,546:INFO:epoch[1], iter[1200], loss:219.895906, fps:436.57 imgs/sec, lr:0.0016389192314818501
2021-08-05 16:06:49,521:INFO:epoch[1], iter[1300], loss:218.516680, fps:441.79 imgs/sec, lr:0.001775382086634636
2021-08-05 16:07:18,303:INFO:epoch[1], iter[1400], loss:209.922935, fps:444.79 imgs/sec, lr:0.0019118449417874217
2021-08-05 16:07:47,702:INFO:epoch[1], iter[1500], loss:210.997816, fps:435.60 imgs/sec, lr:0.0020483077969402075
2021-08-05 16:08:16,482:INFO:epoch[1], iter[1600], loss:210.678421, fps:444.88 imgs/sec, lr:0.002184770768508315
2021-08-05 16:08:45,568:INFO:epoch[1], iter[1700], loss:203.285874, fps:440.07 imgs/sec, lr:0.0023212337400764227
2021-08-05 16:09:13,947:INFO:epoch[1], iter[1800], loss:203.014775, fps:451.11 imgs/sec, lr:0.0024576964788138866
2021-08-05 16:09:42,954:INFO:epoch[2], iter[1900], loss:194.683969, fps:441.28 imgs/sec, lr:0.0025941594503819942
...
```

## [评估过程](#目录)

### 评估

在运行以下命令之前，请检查用于评估的检查点路径。以下脚本中使用的文件**yolov5.ckpt**是最后保存的检查点文件。

```shell
# 使用python命令进行评估
python eval.py \
    --data_dir=xxx/dataset \
    --pretrained=xxx/yolov5.ckpt \
    --eval_shape=640 > log.txt 2>&1 &
或
# 运行shell脚本进行评估
bash run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH] [DEVICE_ID]
```

上述python命令将在后台运行。您可以通过"log.txt"文件查看结果。测试数据集的mAP如下：

```text
# log.txt
=============coco eval reulst=========
Average Precision (AP) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.369
Average Precision (AP) @[ IoU=0.50 | area= all | maxDets=100 ] = 0.573
Average Precision (AP) @[ IoU=0.75 | area= all | maxDets=100 ] = 0.395
Average Precision (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.218
Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.418
Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.482
Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets= 1 ] = 0.298
Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets= 10 ] = 0.501
Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.557
Average Recall (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.395
Average Recall (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.619
Average Recall (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.677
2020-12-21 17:16:40,322:INFO:testing cost time 0.35h
```

## 推理过程

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

### [导出MindIR](#目录)

```shell
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

必须设置ckpt_file参数。
`file_format`的值为AIR或MINDIR

### 在Ascend 310上进行推理

在进行推理之前，必须通过`export.py`脚本导出MindIR文件。下方为使用MindIR模型进行推理的例子。
注意当前batch_Size只能设置为1。

```shell
# Ascend 310推理
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [ANN_FILE] [DVPP] [DEVICE_ID]
```

- `DVPP`必填，可选值为DVPP或CPU，不区分大小写。DVPP硬件要求宽度为16对齐和高度为偶对齐。因此，网络需要使用CPU算子来处理图像。
- `DATA_PATH`为必填项，包含图像数据集的路径。
- `ANN_FILE`为必填项，标注文件的路径。
- `DEVICE_ID`是可选参数，默认值为0。

### 结果

推理结果保存在当前路径中，您可以在acc.log文件中找到类似如下结果。

```text
Average Precision (AP) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.369
Average Precision (AP) @[ IoU=0.50 | area= all | maxDets=100 ] = 0.573
Average Precision (AP) @[ IoU=0.75 | area= all | maxDets=100 ] = 0.395
Average Precision (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.218
Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.418
Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.482
Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets= 1 ] = 0.298
Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets= 10 ] = 0.501
Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.557
Average Recall (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.395
Average Recall (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.619
Average Recall (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.677
```

### [导出ONNX](#目录)

- 导出ONNX模型

  ```shell
  python export.py --ckpt_file /path/to/yolov5.ckpt --file_name /path/to/yolov5.onnx --file_format ONNX
  ```

### 运行ONNX评估

- 从YOLOv5目录运行ONNX评估：

  ```shell
  bash scripts/run_eval_onnx.sh <DATA_DIR> <ONNX_MODEL_PATH> [<DEVICE_TARGET>]
  ```

### 结果

- 您可以通过文件eval.log查看结果。验证数据集的mAP如下所示：

  ```text
  =============coco eval reulst=========
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.366
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.569
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.397
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.213
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.415
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.474
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.299
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.501
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.557
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.399
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.611
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.677
  ```

# [模型说明](#目录)

## [性能](#目录)

### 评估性能

YOLOv5应用于118000张图像上（标注和数据格式必须与COCO 2017相同）

| 参数                | YOLOv5s                                                     | YOLOv5s                                                     |
| -------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 资源                  | Ascend 910；CPU 2.60GHz，192核；内存755GB              | GPU NV SMX2 V100-32G                                        |
| 上传日期             | 7/12/2021                                  | 9/15/2021                                  |
| MindSpore版本         | 1.2.0                                                       | 1.3.0                                                       |
| 数据集                   | 118000张图                                                 | 118000张图                                                 |
| 训练参数       | epoch=300, batch_size=8, lr=0.02,momentum=0.9,warmup_epoch=20| epoch=300, batch_size=32, lr=0.025, warmup_epoch=20, 8p     |
| 优化器                 | 动量                                                    | 动量                                                    |
| 损失函数             | Sigmoid Cross Entropy with logits, Giou Loss                | Sigmoid Cross Entropy with logits, Giou Loss                |
| 输出                   | 框和标签                                             | 框和标签                                             |
| 损失                      | 111.970097                                                  | 85                                                          |
| 速度                     | 8卡，约450FPS                                            | 8卡，约290FPS                                            |
| 总时长                | 8卡，21小时28分钟                                                 | 8卡，35小时                                                      |
| 微调检查点| 53.62MB（.ckpt文件）                                         | 58.87MB（.ckpt文件）                                         |
| 脚本                   | https://gitee.com/mindspore/models/tree/master/official/cv/YOLOv5| https://gitee.com/mindspore/models/tree/master/official/cv/YOLOv5|

### 推理性能

| 参数         | YOLOv5s                                       | YOLOv5s                                     |
| ------------------- | -----------------------------------------------| ---------------------------------------------|
| 资源           | Ascend 910；CPU 2.60GHz，192核；内存755GB| GPU NV SMX2 V100-32G                        |
| 上传日期      | 7/12/2021                    | 9/15/2021                  |
| MindSpore版本  | 1.2.0                                         | 1.3.0                                       |
| 数据集            | 20000张图                                    | 20000张图                                  |
| batch_size         | 1                                             | 1                                           |
| 输出            | 边框位置和分数，以及概率      | 边框位置和分数，以及概率    |
| 准确率           | mAP >= 36.7%（shape=640）                       | mAP >= 36.7%（shape=640）                     |
| 推理模型| 56.67MB（.ckpt文件）                           | 58.87MB（.ckpt文件）                         |

### 迁移学习

# [随机情况说明](#目录)

在dataset.py中，我们设置了“create_dataset”函数内的种子。我们还在train.py中使用随机种子。

# [ModelZoo主页](#目录)

 请查看官方[主页](https://gitee.com/mindspore/models)。
