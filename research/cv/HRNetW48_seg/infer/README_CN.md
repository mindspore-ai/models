# HRNetW48_seg MindX推理及MxBase推理

<!-- TOC -->

- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [模型转换](#模型转换)
    - [SDK启动流程](#sdk启动流程)
    - [MxBase推理流程](#mxbase推理流程)

<!-- /TOC -->

## 脚本说明

### 脚本及样例代码

```tex
├─ HRNetW48_seg
│   ├─ infer                                # MindX推理相关脚本
│   │   ├─ convert                          # om模型转换相关脚本
│   │   │   ├─ convert.sh                   # om模型转换执行脚本
│   │   │   └─ hrnetw48seg_aipp.cfg         # om模型转换配置信息
│   │   ├─ data                             # 推理过程所需数据信息
│   │   │   └─ config
│   │   │       ├─ hrnetw48seg.cfg
│   │   │       ├─ hrnetw48seg.names
│   │   │       └─ hrnetw48seg.pipeline
│   │   ├─ mxbase                           # MxBase推理相关脚本
│   │   │   ├─ src
│   │   │   │   ├─ hrnetw48seg.cpp
│   │   │   │   ├─ hrnetw48seg.h
│   │   │   │   └─ main.cpp
│   │   │   ├─ build.sh                     # MxBase推理执行脚本
│   │   │   └─ CMakeLists.txt
│   │   ├─ sdk                              # SDK推理相关脚本
│   │   │   ├─ cityscapes.py
│   │   │   ├─ do_infer.sh                  # SDK推理执行脚本
│   │   │   └─ main.py
│   │   └─ docker_start.sh                  # 镜像启动
│   ├─ modelarts                            # ModelArts训练相关脚本
│   │   └─ start.py                         # ModelArts训练启动脚本
│   ├─ ...                                  # 其他代码文件
```

### 模型转换

1、首先执行 HRNetW48_seg 目录下的 export.py 将准备好的权重文件转换为 air 模型文件。

```bash
python export.py --device_id [DEVICE_ID] --checkpoint_file [CKPT_PATH] --file_name [FILE_NAME] --file_format AIR --device_target Ascend --dataset [DATASET]
```

2、然后执行 convert 目录下的 convert.sh 将刚刚转换好的 air 模型文件转换为 om 模型文件以备后续使用。

```bash
cd ./infer/convert/
bash convert.sh [AIR_MODEL_PATH] [AIPP_CONFIG_PATH] [OM_MODEL_NAME]

# Example
bash convert.sh ./hrnetw48seg.air ./hrnetw48seg_aipp.cfg ../data/model/hrnetw48seg
```

### SDK启动流程

```bash
# 执行SDK推理
cd ./infer/sdk/
bash do_infer.sh [DATA_PATH] [DATA_LST_PATH]

# Example
bash do_infer.sh ../data/input/cityscapes ../data/input/cityscapes/val.lst
```

推理得到测试集全部图片的语义分割效果图，结果存储于 `./inferResults/` 目录。精度结果在推理结束后打印在执行窗口，与910推理的精度差异可控制在0.5%以内。

### MxBase推理流程

```python
# 执行MxBase推理
cd ./infer/mxbase/
bash build.sh                               # 编译
build/hrnetw48seg [TEST_IMAGE_PATH]         # 推理

# Example
build/hrnetw48seg ./test.png
```

推理结果为输入图片的语义分割效果图，输出位置与输入图片相同，命名以 `_infer` 结尾。
