# RelationNet Mindx高性能模型

## 概述

### 简述

RelationNet

- 参考论文：[paper](https://arxiv.org/abs/1711.11575): **[arXiv:1711.11575](https://arxiv.org/abs/1711.11575) [cs.CV]**

### 默认配置

- 训练数据集（omniglot_resized）

- 测试数据集（omniglot_resized）

### 推理环境准备

- 硬件环境、开发环境和运行环境准备请参见[《CANN 软件安装指南》](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=installation-upgrade)。
- 宿主机上需要安装Docker并登录[Ascend Hub中心](https://ascendhub.huawei.com/#/home)获取镜像。

### 源码介绍

```bash
├── infer                      # MindX高性能预训练模型新增  
│   ├── convert                # 转换om模型脚本
│   │   └── convert.sh
│   ├── data                   # 包括模型文件、推理输出结果
│   │   ├── model              # 存放AIR、OM模型文件
│   │   │   ├── relationnet.om
│   │   │   └── relationnet.air
│   │   ├── input         # 存放预处理后数据，需要自行创建
│   │   └── label         # 存放预处理后的标签，需要自行创建
│   ├── mxbase                 # 基于mxbase推理
│   │   ├── src
│   │   │   ├── Relationnet.cpp
│   │   │   ├── Relationnet.h
│   │   │   └── main.cpp
│   │   ├── CMakeLists.txt
│   │   └── build.sh
│   └── sdk                    # 基于sdk.run包推理
│   │   ├── main.py
│   │   └── run.sh
│   ├── preprocess_mindx.py    # MindX高性能预训练模型数据预处理
├── modelarts                  # MindX高性能预训练模型适配Modelarts新增
│   └───train_start.py         # Modelarts启动文件
```

## 推理

### 准备推理数据

准备模型转换和模型推理所需目录及数据。

1. 下载源码包。

2. 将源码上传至推理服务器任意目录并解压（如：“/home/HwHiAiUser“）。

   ```bash
   # 在环境上执行
   unzip Relationnet_MindSpore_{version}_code.zip
   ```

3. 准备数据。

   将数据集放入data/dataset目录下

   ```bash
   ├── infer                      # MindX高性能预训练模型新增  
   │   ├── convert                # 转换om模型脚本
   │   │   └── convert.sh
   │   ├── data                   # 包括模型文件、推理输出结果
   │   │   ├── model              # 存放AIR、OM模型文件
   │   │   │   ├── relationnet.om
   │   │   │   └── relationnet.air
   │   │   ├── input         # 存放预处理后数据，代码自动创建
   │   │   ├── label         # 存放预处理后的标签，代码自动创建
   │   │   └── dataset       # 存放预处理后的标签，代码自动创建
   │   ├── mxbase                 # 基于mxbase推理
   │   │   ├── src
   │   │   │   ├── Relationnet.cpp
   │   │   │   ├── Relationnet.h
   │   │   │   └── main.cpp
   │   │   ├── CMakeLists.txt
   │   │   └── build.sh
   │   └── sdk                    # 基于sdk.run包推理
   │   │   ├── main.py
   │   │   └── run.sh
   │   ├── preprocess_mindx.py    # MindX高性能预训练模型数据预处理
   ├── modelarts                  # MindX高性能预训练模型适配Modelarts新增
   │   └───train_start.py         # Modelarts启动文件
   ```

   AIR模型可通过“模型训练”后转换生成或通过“下载模型”获取。请将AIR文件放到`data/model`文件夹中。

4. 启动容器。

   进入"infer"目录，执行以下命令，启动容器。

   ```bash
   bash docker_start_infer.sh docker_image:tag model_dir
   ```

   | 参数         | 说明                              |
   | ------------ | --------------------------------- |
   | dokcer_image | 推理镜像名称，根据实际写入        |
   | tag          | 镜像tag，请根据实际配置，如21.0.3 |
   | model_dir    | 推理代码路径                      |

   启动容器时会将推理芯片和数据路径挂载到容器中。可根据需要通过docker_start_infer.sh的device来指定挂载的推理芯片。

   ```bash
   docker run -it \
    --device=/dev/davinci0 \  #可根据需要修改挂载的npu设备
    --device=/dev/davinci_manage \
   ```

   > ![181445_0077d606_8725359](figures/181445_0077d606_8725359.gif) **说明：**
   > MindX SDK开发套件（mxManufacture)已安装在基础镜像中，安装路径：“/usr/local/sdk\_home“。

5. 进行数据预处理。（以数据集为omniglot_resized为例)

   进入data目录创建文件夹

   ```bash
   mkdir input
   mkdir label
   ```

   进入infer目录

   ```bash
   python3 preprocess_mindx.py --label_output_path data/label --data_path ./data/dataset/omniglot_resized --data_output_path data/input
   ```

   预处理得到的结果将保存在./data/input 和 ./data/label路径下

### 模型转换

1. 准备模型文件。

2. 模型转换。

    进入“infer/convert“目录进行模型转换，转换详细信息可查看转换脚本，可在**convert.sh**脚本文件中，查看相关参数。

    ```bash
    if [ $# -ne 2 ]
    then
       echo "Wrong parameter format."
       echo "Usage:"
       echo "         bash $0 [INPUT_AIR_PATH] [AIPP_PATH] [OUTPUT_OM_PATH_NAME]"
       echo "Example: "
       echo "         bash convert_om.sh  xxx.air ./aipp.cfg xx"
       exit 1
    fi

    input_air_path=$1
    output_om_path=$2

    export install_path=/usr/local/Ascend/ascend-toolkit/latest
    export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
    export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
    export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
    export ASCEND_OPP_PATH=${install_path}/opp
    export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest
    echo "Input AIR file path: ${input_air_path}"
    echo "Output OM file path: ${output_om_path}"

    atc --input_format=NCHW \
        --framework=1 \
        --model="${input_air_path}" \
        --input_shape="x:10,1,28,28"  \
        --output="${output_om_path}" \
        --enable_small_channel=0 \
        --log=debug \
        --soc_version=Ascend310 \
        --op_select_implmode=high_precision

    ```

    转换命令如下。

    例如：

    ```bash
       bash convert.sh ./data/model/relationnet.air ./data/model/relationnet
    ```

 （注：本脚本在Ascend-cann-toolkit_5.0.3.alpha005_linux-x86_64.run/Ascend-cann-toolkit_5.0.3.alpha005_linux-aarch64.run 包下完成测试）

### mxBase推理

在该步运行前，请完成数据预处理，或保证其路径在./data/input  和./data/label下

1. 配置相关环境变量

    ```bash
          export ASCEND_VERSION=nnrt/latest
          export ARCH_PATTERN=.
          export MXSDK_OPENSOURCE_DIR=${MX_SDK_HOME}/opensource
          export LD_LIBRARY_PATH="${MX_SDK_HOME}/lib/plugins:${MX_SDK_HOME}/opensource/lib64:${MX_SDK_HOME}/lib:${MX_SDK_HOME}/lib/modelpostprocessors:${MX_SDK_HOME}/opensource/lib:${LD_LIBRARY_PATH}"
    ```

2. 修改配置文件。

   可根据实际情况修改，配置文件位于`infer/mxbase/src/main.cpp`中，可修改参数如下。

   ```c
   void InitRelationnetParam(InitParam* initParam) {
       initParam->deviceId = 0;
       initParam->modelPath = "../data/model/relationnet.om";
   }
   ```

3. 编译工程

   ```bash
   cd infer/mxbase/
   bash build.sh
   ```

4. 运行推理服务

   ```bash
   ./relationnet
   ```

   （请保证数据集被正确的放在./data/input  ./data/label)

5. 观察结果。

   推理图像结果会以.bin的格式保存在./result文件夹中。

   回到relationnet目录下

   ```bash
   cd ../../
   ```

   执行

   ```bash
   python3 postprocess.py --result_path ./infer/mxbase/result --label_path ./infer/data/label
   ```

   ```bash
   aver_accuracy : 0.9918
   ```

   (注：数据的预处理有随机性，准确率会在99%上下浮动)

### MindX SDK推理

1. 修改配置文件。

   1. 可根据实际情况修改pipeline文件。

      ```json
      {
      "relationnet": {
          "stream_config": {
                  "deviceId": "0"
              },
          "appsrc0": {
              "props": {
                      "blocksize": "409600"
                  },
              "factory": "appsrc",
              "next": "tensorinfer0"
          },
          "tensorinfer0": {
              "props": {
              "modelPath": "../data/model/relationnet.om",
              "dataSource": "appsrc0",
              "waitingTime": "2000",
              "outputDeviceId": "-1",
              "singleBatchInfer": "1"
          },
              "factory": "mxpi_tensorinfer",
          "next": "dataserialize"
          },
          "dataserialize": {
              "props": {
                  "outputDataKeys": "tensorinfer0"
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

      2、打开性能统计开关，执行推理

2. 模型推理。

   创建结果文件夹

   ```bash
   mkdir result
   ```

   执行推理命令：

   ```bash
   python3 main.py
   ```

   图像结果会以.bin的格式保存在./result文件夹中。

3. 精度和性能测试

   ```bash
    cd ../../
    python3 postprocess.py --label_path ./infer/data/label/ --result_path ./infer/sdk/result/
   ```

   最后会输出sdk推理的精度

   ```bash
   aver_accuracy : 0.9918
   ```

   (注：数据的预处理有随机性，准确率会在99%上下浮动)
