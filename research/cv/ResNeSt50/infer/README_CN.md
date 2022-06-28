# ResNeSt50 MindX推理及mx Base推理

## 源码介绍

```tex
ResNeSt50_for_mindspore_{version}_code    #代码示例中不体现具体版本号，用{version}代替
├── infer                # MindX高性能预训练模型新增  
│   └── README_CN.md        # 离线推理文档
│   ├── convert          # 转换om模型命令
│   │   └── atc.sh
│   ├── data             # 包括模型文件、模型输入数据集、模型相关配置文件
│   │   ├── input
│   │   │   └── imaganet # imagenet2012(5000 pieces of validation images)
│   │   └── config
│   │   │   ├── imagenet1000_clsidx_to_labels.names
│   │   │   ├── ResNeSt50.cfg
│   │   │   └── ResNeSt50.pipeline
│   │   └── model
│   │   │   └── resnest50.air
│   ├── mxbase           # 基于mxbase推理
│   │   ├── src
│   │   │   ├── ResNeSt50Classify.cpp
│   │   │   ├── ResNeSt50Classify.h
│   │   │   └── main.cpp
│   │   ├── CMakeLists.txt
│   │   └── build.sh
│   ├── util  
│   │   ├── task_metric.py  # 验证精度
│   │   ├── acc5.json       # 用来存放sdk精度结果
│   │   ├── accmx5.json     # 用来存放mxbase精度结果
│   │   └── val_label.txt
│   └── sdk                 # 基于sdk推理
│       ├── result.txt      # 用来存放后处理结果
│       └── main.py
├── scripts
│   ├─run_train.sh
│   ├─run_eval.sh
│   ├─run_distribute_train.sh              # 启动Ascend分布式训练（8卡）
│   ├─run_distribute_eval.sh               # 启动Ascend分布式评估（8卡）
│   └─run_infer_310.sh                     # 启动310推理
├── src
│   ├─datasets
│     ├─autoaug.py                  # 随机数据增强方法
│     ├─dataset.py                  # 数据集处理
│   ├─models
│     ├─resnest.py                  # ResNeSt50网络定义
│     ├─resnet.py                   # 主干网络
│     ├─splat.py                    # split-attention
│     ├─utils.py                    # 工具函数：网络获取、加载权重等
│   ├─config.py                     # 参数配置
│   ├─crossentropy.py               # 交叉熵损失函数
│   ├─eval_callback.py              # 推理信息打印
│   ├─logging.py                    # 日志记录
├──eval.py                          # 评估网络
├──train.py                         # 训练网络
├──export.py                        # 导出Mindir接口
├──create_imagenet2012_label.py     # 创建数据集标签用于310推理精度验证
├──postprocess.py                   # 后处理
└── README.md                       # README文件
```

## 准备推理数据

准备模型转换和模型推理所需目录及数据。

1. “下载模型脚本”和“下载模型”，下载所需软件包。

2. 将源码上传至推理服务器任意目录并解压（如：“/home/HwHiAiUser/ResNeSt50_for_mindspore_{version}_code/ResNeSt50“）。

   ```shell
   cd /home/HwHiAiUser/ResNeSt50_for_mindspore_{version}_code/ResNeSt50/infer && dos2unix `find .`
   ```

3. 进入容器。（根据实际情况修改命令）

   ```shell
   bash docker_start_infer.sh docker_image:tag model_dir
   ```

   **表 2**  参数说明

   | 参数         | 说明                                  |
   | ------------ | ------------------------------------- |
   | docker_image | 推理镜像名称，根据实际写入。          |
   | tag          | 镜像tag，请根据实际配置，如：21.0.4。 |
   | model_dir    | 推理代码路径。                        |

   启动容器时会将推理芯片和数据路径挂载到容器中。可根据需要通过修改**docker\_start\_infer.sh**的device来指定挂载的推理芯片。

   ```shell
   docker run -it \
     --device=/dev/davinci0 \        # 可根据需要修改挂载的npu设备
     --device=/dev/davinci_manager \
   ```

4. 准备数据。

   将下载好的`ImageNet`中的`valid`数据集放到上面目录的`imagenet`目录下，也就是说`imagenet`目录下直接是`ILSVRC2012_val_00000001.JPEG`...到`ILSVRC2012_val_00050000.JPEG`的50000图片。

## 模型转换

1. 准备模型文件。

   进入`infer/data/model`目录，将训练所得AIR模型文件上传到该目录下。

2. 模型转换。

   进入`infer/convert`目录进行模型转换，执行以下命令

   ```shell
bash atc.sh  [input_path] [output_path]
   e.g.
bash atc.sh ../data/model/resnest50.air ../data/model/resnest50
   ```

   在`infer/data/model`目录下就会生成一个`resnest50.om`文件

## mxBase推理

### 前提条件

* 已进入推理容器环境。

* 确保`imagenet1000_clsidx_to_labels.names`文件在上述目录的`infer/data/config`下。
* 确保`val_label.txt`文件在上述目录的`infer/util`下。
* 在`infer/util`目录下创建一个`accmx5.json`的文件。

### 操作步骤

进入infer/mxbase目录。  

1. 编译工程。

   运行命令

   ```shell
   bash build.sh
   ```

   其中，`main.cpp`文件中，模型路径参数需要根据实际情况修改，此处模型的目录请写具体的目录。

   ```c++
   initParam.modelPath = "../../data/model/resnest50.om";
   ```

   编译生成**`resnest50`**可执行文件至`build.sh`同目录文件**`build`**中

2. 运行推理服务。
   运行命令

   ```shell
   cd build/
   ./resnest50 [input_data_path] [dataset_size]
   e.g.
./resnest50 ../../data/input/imagenet 50000
   ```

3. 观察结果。

   在`infer/mxbase/build`目录下会生成一个`mx_pred_result.txt`，模型后处理结果存放在里面。

   然后转到`infer/util`目录下，执行下面语句：

   ```shell
   python3 task_metric.py [predict_result_path] [label_file] [result_file] [Top_one_to_top_five]
   e.g.
python3 task_metric.py --prediction ../mxbase/build/mx_pred_result.txt --gt ./val_label.txt --result_json ./accmx5.json --top_k 5
   ```

   那么`accmx5.json`文件中开头的`accuracy`的第一个值是`TOP1`，第五个值是`TOP5`。

## MindX SDK推理

### 前提条件

* 已进入推理容器环境。
* 确保`imagenet1000_clsidx_to_labels.names`文件在上述目录的`infer/data/config`下。
* 确保`ResNeSt50.cfg`文件在上述目录的`infer/data/config`下。
* 确保`val_label.txt`文件在上述目录的`infer/util`下。
* 在`infer/sdk`目录下创建一个`result.txt`文件
* 在`infer/util`目录下创建一个`acc5.json`的文件。

### 操作步骤  

**进入infer/sdk目录。**  

1. 修改配置文件。

实际情况修改当前目录下的`pipeline`文件，模型推理路径需要根据实际情况修改。此处模型路径应该填写具体路径，如："`../data/model/resnest50.om`"。

   ```tex
   {
       "ResNeSt50": {
           "stream_config": {
                   "deviceId": "0"
           },
           "appsrc0": {
                   "props": {
                           "blocksize": "409600"
                   },
                   "factory": "appsrc",
                   "next": "mxpi_tensorinfer0"
           },
           "mxpi_tensorinfer0": {
                   "props": {
                           "dataSource": "appsrc0",
                           "modelPath": "../data/model/resnest50.om",
                           "waitingTime": "2000",
                           "outputDeviceId": "-1"
                   },
                   "factory": "mxpi_tensorinfer",
                   "next": "mxpi_classpostprocessor0"
           },
           "mxpi_classpostprocessor0": {
                   "props": {
                           "dataSource": "mxpi_tensorinfer0",
                           "postProcessConfigPath": "../data/config/ResNeSt50.cfg",
                           "labelPath": "../data/config/imagenet1000_clsidx_to_labels.names",
                           "postProcessLibPath": "libresnet50postprocess.so"
                   },
                   "factory": "mxpi_classpostprocessor",
                   "next": "mxpi_dataserialize0"
           },
           "mxpi_dataserialize0": {
                   "props": {
                           "outputDataKeys": "mxpi_classpostprocessor0"
                   },
                   "factory": "mxpi_dataserialize",
                   "next": "appsink0"
           },
           "appsink0": {
                   "props": {
                           "blocksize": "4096000"
                   },
                   "factory": "appsink"
           }
       }
   }
   ```

2. 性能测试。

   执行以下命令：

   ```shell
   vi /usr/local/sdk_home/mxManufacture/config/sdk.conf
   # 打开性能统计开关。将“enable_ps”参数设置为true，“ps_interval_time”参数设置为6。
   ```

3. 当前`infer/sdk`目录下，运行推理服务。 运行命令：

   ```shell
   bash run.sh [input_data_path] [result_file]
   e.g.
   bash run.sh --glob ../data/input/imagenet --result_file ./result.txt
   ```

   模型后处理结果存放在`result.txt`里面。

4. 将在日志目录"`/usr/local/sdk_home/mxManufacture/logs`"查看性能统计结果，输入以下命令：

   ```shell
   cd /usr/local/sdk_home/mxManufacture/logs
   ```

   `performance—statistics.log.e2e.xxx`

   `performance—statistics.log.plugin.xxx`

   `performance—statistics.log.tpr.xxx`

   其中`e2e`日志统计端到端时间，`plugin`日志统计单插件时间。可以输入**`ll`**查看

5. 输入以下命令查看性能：(挑一个最近时间的)

   ```shell
   cat 'performance-statistics.log.plugin.2022-06-20 16:44:41' | grep mxpi_tensorinfer0
   ```

   取几组`average`时间的平均值，`sdk`推理性能=1000000/平均值（`imgs/sec`）。

   得到`ResNeSt50`推理性能为31.8 `images/sec`。

6. 然后转到`infer/util`目录下，执行下面语句：

   ```shell
   python3 task_metric.py [predict_result_path] [label_file] [result_file] [Top_one_to_top_five]
   e.g.
   python3 task_metric.py --prediction ../sdk/result.txt --gt ./val_label.txt --result_json ./acc5.json --top_k 5
   ```

   那么`acc5.json`文件中开头的`accuracy`的第一个值是`TOP1`，第五个值是`TOP5`。
