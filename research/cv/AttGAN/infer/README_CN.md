# AttGAN MindX推理及mx Base推理

## 源码介绍

```tex
AttGAN_for_mindspore_{version}_code    #代码示例中不体现具体版本号，用{version}代替
├── infer                # MindX高性能预训练模型新增  
│   └── README_CN.md        # 离线推理文档
│   ├── convert          # 转换om模型命令
│   │   └── air2om.sh
│   ├── data  
│   │   ├──dataset     # 数据集存放
│   │   │   └── image   # 图片存放地址
│   │   ├── model       # 模型存放
│   │   ├── mxbase_result   # mxbase结果保存
│   │   ├── sdk_result      # sdk结果保存
│   ├── mxbase           # 基于mxbase推理
│   │   ├── src
│   │   │   └── Attgan.cpp
│   │   │   └── Attgan.h
│   │   │   └── main.cpp
│   │   ├── CMakeLists.txt
│   │   └── build.sh
│   └── sdk                 # 基于sdk推理
│       ├── config      # 存放运行需要python文件
│       │   │   │   └── attgan.pipeline  #pipeline文件
│       │   │   │   └── config.py      #运行函数文件
│       ├── api.py
│       └── main.py
├── scripts
│   ├─run_single_train.sh
│   ├──run_single_train_gpu.sh
│   ├─run_eval.sh
│   ├─run_eval_gpu.sh
│   ├─run_distribute_eval_gpu.sh
│   ├─run_distribute_train.sh              # 启动Ascend分布式训练（8卡）
│   ├─run_distribute_train_gpu.sh
│   ├─run_distribute_eval.sh               # 启动Ascend分布式评估（8卡）
│   └─run_infer_310.sh                     # 启动310推理
├── src
│   ├─attgan.py
│   ├─block.py
│   ├─cell.py
│   ├─data.py
│   ├─helpers.py
│   ├─loss.py
│   ├─utils.py                    # 工具函数：网络获取、加载权重等
├──eval.py
├──train.py                         # 训练网络
├──export.py                        # 导出Mindir接口
├──preprocess.pu                    # 预处理
├──postprocess.py                   # 后处理
└── README.md                       # README文件
```

## 准备推理数据

准备模型转换和模型推理所需目录及数据。

1. “下载模型脚本”和“下载模型”，下载所需软件包。

2. 将源码上传至推理服务器任意目录并解压（如：“/home/HwHiAiUser/AttGAN_for_mindspore_{version}_code/AttGAN“）。

   ```shell
   cd /home/HwHiAiUser/AttGAN_for_mindspore_{version}_code/AttGAN/infer && dos2unix `find .`
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

   将下载好的`celeba`中的`img`数据集中的图片放到上面目录的`data/dataset/image`目录下，同时将图片对应的`list_attr_celeba.txt`标签文件放在`data/dataset`。

## 模型转换

1. 准备模型文件。

   进入`infer/data/model`目录，将训练所得AIR模型文件上传到该目录下。

2. 模型转换。

   进入`infer/convert`目录进行模型转换，执行以下命令

   ```shell
   bash atc.sh  [input_path] [output_path]
   e.g.
   bash air2om.sh ../data/model/attgan.air ../data/model/attgan
   ```

   在`infer/data/model`目录下就会生成一个`attgan.om`文件

## mxBase推理

### 前提条件

* 已进入推理容器环境。
* cd到AttGAN/infer目录下，创建data/dataset目录，将下载好的CelebA数据集中的标签文list_attr_celeba.txt放到dataset文件夹下
* 在data/dataset文件夹下创建image文件夹，放入图片文件
* 在data/dataseet文件夹下创建存放数据结果的mxbase_result文件夹

### 操作步骤

进入infer/mxbase目录。  

1. 编译工程。

   运行命令

   ```shell
   bash build.sh
   ```

   其中，`main.cpp`文件中，模型路径参数需要根据实际情况修改，此处模型的目录请写具体的目录。

   ```c++
   OM_MODEL_PATH = "../../data/model/attgan.om";
   savePath = "../../data/mxbase_result"
   ```

   编译生成**`attgan`**可执行文件至`build.sh`同目录文件**`build`**中

2. 运行推理服务。
   运行命令

   ```shell
   cd build/
   ./attgan [input_data_path]
   e.g.
   ./attgan ../../data/dataset
   ```

## MindX SDK推理

### 前提条件

* 已进入推理容器环境
* cd到AttGAN/infer目录下，创建data/dataset目录，将下载好的CelebA数据集中的标签文list_attr_celeba.txt放到dataset文件夹下
* 在data/dataset文件夹下创建image文件夹，放入图片文件
* 在data/dataseet文件夹下创建存放数据结果的sdk_result文件夹

### 操作步骤  

**进入infer/sdk目录。**  

1. 修改配置文件。

实际情况修改当前目录下的`pipeline`文件，模型推理路径需要根据实际情况修改。此处模型路径应该填写具体路径，如："`../data/model/attgan.om`"。

   ```tex
       {
        "detection": {
            "stream_config": {
                "deviceId": "0"
            },
            "appsrc1": {
                "props": {
                    "blocksize": "409600"
                },
                "factory": "appsrc",
                "next": "mxpi_tensorinfer0:1"
            },
            "appsrc0": {
                "props": {
                    "blocksize": "409600"
                },
                "factory": "appsrc",
                "next": "mxpi_tensorinfer0:0"
            },
            "mxpi_tensorinfer0": {
                "props": {
                    "dataSource": "appsrc0,appsrc1",
                    "modelPath": "../model/attgan.om"
                },
                "factory": "mxpi_tensorinfer",
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

2. 当前`infer/sdk`目录下，运行推理服务。 运行命令：

   ```shell
   python3 main.py [input_data_path]
   e.g.
   python3 main.py  ../data/dataset
   ```

   模型后处理结果存放在`../../data/sdk_result`里面。