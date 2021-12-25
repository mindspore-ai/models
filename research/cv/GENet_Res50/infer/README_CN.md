# GENet MindX推理及mx Base推理

<!-- TOC -->

- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [数据集准备](#数据集准备)
    - [模型转换](#模型转换)
    - [MindX SDK 启动流程](#mindx-sdk-启动流程)
    - [mx Base 推理流程](#mx-base-推理流程)

<!-- /TOC -->

## 脚本说明

### 脚本及样例代码

```tex
GENet_Res50
├── infer                      # MindX高性能预训练模型新增
│   ├── docker_start.sh        # 启动docker环境脚本
│   ├── convert                # 转换om模型脚本
│   │   ├── aipp.cfg           #  pipline 配置
│   │   └── convert_om.sh
│   ├── data                   # 包括模型文件、推理输出结果
│   │   ├── models             # 存放AIR、OM模型文件
│   │   │   ├── GENet_Res50_plus.air
│   │   │   └── GENet_Res50_plus.om
│   │   ├── input              # 存放用于mxbase推理的图片
│   │   │   └──ILSVRC2012_val_00000001.JPEG
│   │   └── config
│   │   │   └──GENet.cfg
│   ├── mxbase                 # 基于mxbase推理
│   │   ├── src
│   │   │   ├── GENet.cpp
│   │   │   ├── GENet.h
│   │   │   └── main.cpp
│   │   ├── CMakeLists.txt
│   │   └── build.sh
│   └── sdk                    # 基于sdk.run包推理
│   │   ├── main.py            # 推理脚本
│   │   ├── classification_task_metric.py      # 验证推理精度
│   │   └── run.sh             # 调用推理的python脚本
├── modelarts                  # MindX高性能预训练模型适配Modelarts新增
│   └───train_start.py         # Modelarts启动文件
```

### 数据集准备

由于后续推理均在容器中进行，因此需要把用于推理的模型文件、代码等均放在同一数据路径中，后续示例将以“/home/HwHiAiUser“为例。示例的推理数据集为Imagenet2012。

1、在 infer 目录下或其他文件夹下创建数据集文件夹（记为“/home/data/imagenet_val/imagenet/"），并将 imagenet数据集的val部分上传到该文件夹下。该文件夹下都是图片。并且准备好imagenet的val文件，在本例中为“/home/data/imagenet_val/imagenet/val_lable.txt”。

2、MindX SDK 直接使用原始数据集，上述数据集准备好后即可执行 “MindX SDK 启动流程” 步骤。

### 模型转换

1、首先执行 r2plus1d 根目录下的 export.py 将准备好的权重文件转换为 air 模型文件。

```python
# 此处简要举例
python export.py \
--formate_type="AIR" \
```

2、然后执行 convert 目录下的 ATC_AIR_2_OM.sh 将刚刚转换好的 air 模型文件转换为 om 模型文件以备后续使用。

```bash
cd infer/convert
bash convert_om.sh ../data/models/GENet_Res50_plus.air aipp.cfg ../data/models/GENet_Res50_plus.om

#转换后的模型结果为GENet_Res50_plus.om，将放在infer/data/models/目录下
```

### MindX SDK 启动流程

```bash
# 通过 bash 脚本启动 MindX SDK 推理
cd infer/sdk
bash run.sh [IMG_DIR] [RESULTS_PATH]
# 例如：
bash run.sh /home/data/imagenet_val/imagenet/ results

# 通过 python 命令启动 MindX SDK 推理结果的精度校验。
# 其中RESULTS_PATH表示sdk推理出的所有txt格式的结果的路径，LABEL_PATH表示标签文件的路径，如val_label.txt，OUTPUT_PATH表示精度文件的保存路径，OUTPUT_FILE_NAME表示精度文件的名字。
python classification_task_metric.py [RESULTS_PATH] [LABEL_PATH] [OUTPUT_PATH] [OUTPUT_FILE_NAME]

# 例如如下代码，将results路径下的推理结果与label文件/home/data/imagenet_val/val_lable.txt进行对比，结果保存在./results.json中。
python3 classification_task_metric.py results /home/data/imagenet_val/val_lable.txt ./ results.json

cat results.json
```

推理结果示例：

```bash
cd results
cat ILSVRC2012_val_00037500_1.txt
# 98 97 99 136 127
```

推理精度示例：

```bash
{"title": "Overall statistical evaluation",
 "value": [{"key": "Number of images", "value": "50000"}, {"key": "Number of classes",  "value": "5"}, {"key": "Top1 accuracy", "value": "78.2%"}, {"key": "Top2 accuracy", "value": "87.95%"}, {"key": "Top3 accuracy", "value": "91.25%"}, {"key": "Top4 accuracy", "value": "92.99%"}, {"key": "Top5 accuracy", "value": "94.13%"}]}
```

### mx Base 推理流程

1、编译 mx Base

```bash
cd infer/mxbase/
bash build.sh
```

**说明：**
> 编译成功后将在该路径（“infer/mxbase”)下生成“build”目录，其中包含了编译好的可执行文件“GENet”。

2、执行 mx Base 推理

   ```bash
   ./build/GENet image_path
   ```

   | 参数       | 说明                                                   |
   | ---------- | ------------------------------------------------------ |
   | image_path | mxbase推理图片所在文件的路径。如：“infer/data/input”。 |

3、观察结果。

图像分类的top5预测结果会以*.txt的格式保存在../mxbase/infer_results文件夹中。

```bash
WARNING: Logging before InitGoogleLogging() is written to STDERR
I0110 07:10:15.160562 26116 ModelInferenceProcessor.cpp:22] Begin to ModelInferenceProcessor init
I0110 07:10:15.797708 26116 ModelInferenceProcessor.cpp:69] End to ModelInferenceProcessor init
I0110 07:10:15.798082 26116 Resnet50PostProcess.cpp:57] Start to Init Resnet50PostProcess.
I0110 07:10:15.798097 26116 PostProcessBase.cpp:69] Start to LoadConfigDataAndLabelMap in  PostProcessBase.
I0110 07:10:15.801332 26116 Resnet50PostProcess.cpp:66] End to Init Resnet50PostProcess.
I0110 07:10:15.806140 26116 GENet.cpp:92] image size after resize224 224
I0110 07:10:15.878696 26116 Resnet50PostProcess.cpp:79] Start to Process Resnet50PostProcess.
I0110 07:10:15.879489 26116 Resnet50PostProcess.cpp:126] End to Process Resnet50PostProcess.
I0110 07:10:15.879504 26116 GENet.cpp:147] image path../data/input//ILSVRC2012_val_00049999.JPEG
I0110 07:10:15.879520 26116 GENet.cpp:151] file path for saving resultinfer_results/ILSVRC2012_val_00049999_1.txt
I0110 07:10:15.879688 26116 GENet.cpp:161]  className:groom, bridegroom confidence:10.6797 classIndex:982
I0110 07:10:15.879734 26116 GENet.cpp:161]  className:bow tie, bow-tie, bowtie confidence:5.42188 classIndex:457
I0110 07:10:15.879748 26116 GENet.cpp:161]  className:gown confidence:4.64453 classIndex:578
I0110 07:10:15.879788 26116 GENet.cpp:161]  className:bathtub, bathing tub, bath, tub confidence:3.91797 classIndex:435
I0110 07:10:15.879815 26116 GENet.cpp:161]  className:tub, vat confidence:3.70312 classIndex:876
I0110 07:10:15.884963 26116 GENet.cpp:92] image size after resize224 224
I0110 07:10:15.956809 26116 Resnet50PostProcess.cpp:79] Start to Process Resnet50PostProcess.
I0110 07:10:15.957525 26116 Resnet50PostProcess.cpp:126] End to Process Resnet50PostProcess.
I0110 07:10:15.957543 26116 GENet.cpp:147] image path../data/input//ILSVRC2012_val_00041670.JPEG
I0110 07:10:15.957556 26116 GENet.cpp:151] file path for saving resultinfer_results/ILSVRC2012_val_00041670_1.txt
I0110 07:10:15.957723 26116 GENet.cpp:161]  className:wooden spoon confidence:8.80469 classIndex:910
I0110 07:10:15.957751 26116 GENet.cpp:161]  className:paddle, boat paddle confidence:4.83203 classIndex:693
I0110 07:10:15.957759 26116 GENet.cpp:161]  className:ladle confidence:3.99609 classIndex:618
I0110 07:10:15.957801 26116 GENet.cpp:161]  className:paintbrush confidence:3.74609 classIndex:696
I0110 07:10:15.957885 26116 GENet.cpp:161]  className:spatula confidence:3.63281 classIndex:813
I0110 07:10:15.963654 26116 GENet.cpp:92] image size after resize224 224
I0110 07:10:16.035523 26116 Resnet50PostProcess.cpp:79] Start to Process Resnet50PostProcess.
I0110 07:10:16.036218 26116 Resnet50PostProcess.cpp:126] End to Process Resnet50PostProcess.
I0110 07:10:16.036232 26116 GENet.cpp:147] image path../data/input//ILSVRC2012_val_00033336.JPEG
I0110 07:10:16.036240 26116 GENet.cpp:151] file path for saving resultinfer_results/ILSVRC2012_val_00033336_1.txt
I0110 07:10:16.036347 26116 GENet.cpp:161]  className:carbonara confidence:8.71094 classIndex:959
I0110 07:10:16.036366 26116 GENet.cpp:161]  className:wok confidence:4.38281 classIndex:909
I0110 07:10:16.036370 26116 GENet.cpp:161]  className:spaghetti squash confidence:4.15625 classIndex:940
I0110 07:10:16.036386 26116 GENet.cpp:161]  className:caldron, cauldron confidence:4.07812 classIndex:469
I0110 07:10:16.036401 26116 GENet.cpp:161]  className:head cabbage confidence:3.65039 classIndex:936
I0110 07:10:16.041656 26116 GENet.cpp:92] image size after resize224 224
I0110 07:10:16.113504 26116 Resnet50PostProcess.cpp:79] Start to Process Resnet50PostProcess.
I0110 07:10:16.114178 26116 Resnet50PostProcess.cpp:126] End to Process Resnet50PostProcess.
I0110 07:10:16.114199 26116 GENet.cpp:147] image path../data/input//ILSVRC2012_val_00050000.JPEG
I0110 07:10:16.114212 26116 GENet.cpp:151] file path for saving resultinfer_results/ILSVRC2012_val_00050000_1.txt
I0110 07:10:16.114312 26116 GENet.cpp:161]  className:llama confidence:8.92188 classIndex:355
I0110 07:10:16.114336 26116 GENet.cpp:161]  className:Arabian camel, dromedary, Camelus dromedarius confidence:4.08594 classIndex:354
I0110 07:10:16.114346 26116 GENet.cpp:161]  className:koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus confidence:3.17188 classIndex:105
I0110 07:10:16.114360 26116 GENet.cpp:161]  className:ostrich, Struthio camelus confidence:2.38672 classIndex:9
I0110 07:10:16.114372 26116 GENet.cpp:161]  className:ram, tup confidence:2.37109 classIndex:348
I0110 07:10:16.119334 26116 GENet.cpp:92] image size after resize224 224
I0110 07:10:16.191089 26116 Resnet50PostProcess.cpp:79] Start to Process Resnet50PostProcess.
I0110 07:10:16.191761 26116 Resnet50PostProcess.cpp:126] End to Process Resnet50PostProcess.
I0110 07:10:16.191781 26116 GENet.cpp:147] image path../data/input//ILSVRC2012_val_00025002.JPEG
I0110 07:10:16.191793 26116 GENet.cpp:151] file path for saving resultinfer_results/ILSVRC2012_val_00025002_1.txt
I0110 07:10:16.191905 26116 GENet.cpp:161]  className:baseball confidence:8.21094 classIndex:429
I0110 07:10:16.191927 26116 GENet.cpp:161]  className:ballplayer, baseball player confidence:7.94141 classIndex:981
I0110 07:10:16.191937 26116 GENet.cpp:161]  className:chainlink fence confidence:3.41602 classIndex:489
I0110 07:10:16.191948 26116 GENet.cpp:161]  className:racket, racquet confidence:2.29492 classIndex:752
I0110 07:10:16.191958 26116 GENet.cpp:161]  className:sandbar, sand bar confidence:2.12891 classIndex:977
I0110 07:10:16.403167 26116 DeviceManager.cpp:83] DestroyDevices begin
I0110 07:10:16.403223 26116 DeviceManager.cpp:85] destroy device:0
I0110 07:10:17.514941 26116 DeviceManager.cpp:91] aclrtDestroyContext successfully!
I0110 07:10:19.585104 26116 DeviceManager.cpp:99] DestroyDevices successfully
I0110 07:10:19.585137 26116 main.cpp:83]  ms    fps: 6.97829 imgs/sec
```
