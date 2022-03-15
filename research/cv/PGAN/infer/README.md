# PGAN MindX推理及mx Base推理

[脚本说明](#脚本说明)

- [脚本及样例代码](#脚本及样例代码)
- [模型转换](#模型转换)
- [MindX SDK 启动流程](#mindx-sdk-启动流程)
- [mx Base 推理流程](#mx-base-推理流程)

## 脚本说明

### 脚本及样例代码

 ```text
 ├── PGAN
 │   └── README_CN.md              # PGAN 的说明文件
 ├── infer               # MindX高性能预训练模型新增  
 │   └── README.md       # 离线推理文档
 │   └── docker_start_infer.sh
 │   ├── convert
 │   │   └──convert.sh   # 转换om模型命令
 │   ├── mxBase          # 基于mxbase推理
 │   │   ├── src
 │   │   │   └── Pgan.cpp
 │   │   │   └── Pgan.h
 │   │   │   └── main.cpp
 │   │   └── CMakeLists.txt
 │   │   └── build.sh
 │   ├── sdk             # 基于sdk.run包推理；如果是C++实现，存放路径一样
 │   │   ├── pipeline
 │   │   │   └── pgan.pipeline
 │   │   ├── python
 │   │   │   ├── api
 │   │   │   │   └── infer.py  # sdk推理步骤所用方法集合类
 │   │   │   ├── config
 │   │   │   │   └── config.py # 一些常用常量定义
 │   │   │   ├── src
 │   │   │   │   └── image_transform.py # 输出数据处理成RGB图片数据
 │   │   │   └── main.py
 │   │   │   └── run.sh
 ├── ......                 # 其他代码文件
 ```

### 模型转换

1. 首先将ModelArts平台 得到的 air 模型文件，放在convert目录下。

2. 然后执行 convert 目录下的 convert.sh 将刚刚转换好的 air 模型文件转换为 om 模型文件以备后续使用。

 ```bash
 cd /home/PGAN/infer/convert
 bash convert.sh [air_path] [om_path]
 例如：
 bash  convert.sh ./PGAN.air ../data/model/PGAN
 # 转换后的模型结果为PGAN.om，将放在/home/PGAN/infer/data/model/目录下
 ```

### MindX SDK 启动流程

 1. 通过 bash 脚本启动 MindX SDK 推理

 ```bash

 cd /home/PGAN/infer/sdk/python
 bash run.sh [pipeline] [output_directory] [batch_size]
 ```

 例如：

```bash

 bash run.sh ../pipeline/pgan.pipeline ./output 1
 # 注意:../pipeline/pgan.pipeline 中默认 MindX SDK 推理所需的模型文件为 "pgan.om"，且放在 ../data/model/ 目录下，具体可以修改该文件中的 "modelPath" 属性进行配置。
```

 执行成功后在./output目录可以看到随机生成的一张人脸图片

 2. 通过 python 命令启动 MindX SDK 推理

```bash
 python3.7 main.py \
 --pipeline_path=pipeline \
 --infer_result_dir=[output_directory] \
 --batch_size=[batch_size]
```

 例如：

 ```bash
 python3.7 main.py \
 --pipeline_path=../pipeline/pgan.pipeline \
 --infer_result_dir=./output \
 --batch_size=1
 # 注意： ../data/config/pgan.pipeline 中默认 MindX SDK 推理所需的模型文件为 "pgan.om"，且放在 ../data/model/ 目录下，具体可以修改该文件中的 "modelPath" 属性进行配置。
 ```

 执行成功后在./output目录可以看到随机生成的一张人脸图片

| 参数             | 说明                     |
| ---------------- | ------------------------ |
| pipeline         | pipeline路径             |
| output_directory | 输出图片存放路径         |
| batch_size       | 用户希望生成的人脸图片数 |

### mx Base 推理流程

1. 编译 mx Base

 ```bash
 cd /home/pgan/infer/mxBase
 bash build.sh
 ```

> ![输入图片说明](https://images.gitee.com/uploads/images/2021/0719/172222_3c2963f4_923381.gif "icon-note.gif") **说明**：编译成功后将在该路径（“/home/pgan/infer/mxbase”)下生成“build”目录，其中包含了编译好的可执行文件“Pgan”。“/home/pgan/infer/mxBase”目录下的“Pgan”也可直接运行，二者是一致的。
>

2. 执行 mx Base 推理

 ```bash
 ./Pgan [om_path] [output_directory] [batch_size]
 # 按顺序传入模型路径、输出图像存放路径（不需要提前创建）、希望生成的人脸图片数
 例如：
 ./Pgan ../convert/PGAN.om ./output 1
 # 执行成功后，输出目录有一张生成人脸图片
 ```

| 参数             | 说明                     |
| ---------------- | ------------------------ |
| om_path          | om模型路径               |
| output_directory | 输出图片存放路径         |
| batch_size       | 用户希望生成的人脸图片数 |

> ![输入图片说明](https://images.gitee.com/uploads/images/2021/0719/172222_3c2963f4_923381.gif "icon-note.gif") **说明：**视频行为识别的结果将保存在output_directory参数指定的目录下，不论指定生成多少张图片，都会组合成一张图片命名为result.jpg。