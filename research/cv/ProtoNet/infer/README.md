# 目录

- [源码介绍](#源码介绍)
- [准备推理数据](#准备推理数据)
- [模型转换](#模型转换)
- [mxBase推理](#mxbase推理)
- [MindX SDK推理](#mindx-sdk推理)

## 源码介绍

```tex
/home/HwHiAiUser/protonet_for_mindspore_{version}_code
├── infer              # MindX高性能预训练模型新增
│   └── README.md
│   ├── convert
│   │   ├── convert.sh       # 转换om模型脚本
│   │   └── dataprocess.sh   # 数据预处理脚本(将图片处理为bin文件，需要使用Mindspore函数库)
│   ├── data                # 模型数据集、模型配置文件、模型文件
│   │   ├── input
│   │   │      ├── dataset         # 数据集
│   │   │      │     ├── data      # 数据集内容
│   │   │      │     ├── omniglot
│   │   │      │     ├── raw
│   │   │      │     └── splits    # 该文件夹内容包含对数据集的说明，由此区分训练集、验证集、测试集
│   │   │      └── ...
│   │   └── config
│   │       └── protonet.pipeline
│   ├── mxbase           # 基于mxbase推理
│   │   ├── src
│   │   │   ├── Protonet.cpp
│   │   │   ├── Protonet.h
│   │   │   └── main.cpp
│   │   ├── CMakeLists.txt
│   │   └── build.sh
│   └── sdk           # 基于sdk.run包推理
│   │   ├── main.py
│   │   └── run.sh
│   └── docker_start_infer.sh     # 启动容器脚本
```

## 准备推理数据

准备模型转换和模型推理所需目录及数据。

1. 下载源码包。

2. 将源码上传至推理服务器任意目录并解压（如：“/home/HwHiAiUser“）。

3. 准备数据。

    （_准备用于推理的图片、数据集、模型文件、代码等，放在同一数据路径中，如：“/home/HwHiAiUser“。_）

    由于后续推理均在容器中进行，因此需要把用于推理的图片、数据集、模型文件、代码等均放在同一数据路径中，后续示例将以“/home/HwHiAiUser“为例。

    ```
    ..
    ├── infer            # MindX高性能预训练模型新增
    │   └── README.md
    │   ├── convert
    │   │   ├──dataprocess.sh  # 数据预处理脚本
    │   │   └──convert.sh     # 转换om模型脚本
    │   ├── data     # 模型数据集、模型配置文件、模型文件
    │   │   ├── input
    │   │   ├── model       # air、om模型文件
    │   │   └── config      # 推理所需的配置文件
    │   ├── mxbase
    │   └── sdk
    │   └──docker_start_infer.sh     # 启动容器脚本
    ```

    AIR模型可通过“模型训练”后转换生成或通过“下载模型”获取。

    将Omniglot数据集放到“infer/data/input”目录下

4. 启动容器。

    进入“infer“目录，执行以下命令，启动容器。

    **bash docker\_start\_infer.sh** _docker\_image:tag_ _model\_dir_

    **表 2**  参数说明

    <a name="table8122633182517"></a>
    <table><thead align="left"><tr id="row16122113320259"><th class="cellrowborder" valign="top" width="40%" id="mcps1.2.3.1.1"><p id="p16122163382512"><a name="p16122163382512"></a><a name="p16122163382512"></a>参数</p>
    </th>
    <th class="cellrowborder" valign="top" width="60%" id="mcps1.2.3.1.2"><p id="p8122103342518"><a name="p8122103342518"></a><a name="p8122103342518"></a>说明</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row11225332251"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p712210339252"><a name="p712210339252"></a><a name="p712210339252"></a><em id="i121225338257"><a name="i121225338257"></a><a name="i121225338257"></a>docker_image</em></p>
    </td>
    <td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p0122733152514"><a name="p0122733152514"></a><a name="p0122733152514"></a>推理镜像名称，根据实际写入。</p>
    </td>
    </tr>
    <tr id="row052611279127"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p2526192714127"><a name="p2526192714127"></a><a name="p2526192714127"></a><em id="i12120733191212"><a name="i12120733191212"></a><a name="i12120733191212"></a>tag</em></p>
    </td>
    <td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p16526142731219"><a name="p16526142731219"></a><a name="p16526142731219"></a>镜像tag，请根据实际配置，如：21.0.2。</p>
    </td>
    </tr>
    <tr id="row5835194195611"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p59018537424"><a name="p59018537424"></a><a name="p59018537424"></a>model_dir</p>
    </td>
    <td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1390135374214"><a name="p1390135374214"></a><a name="p1390135374214"></a>推理代码路径。</p>
    </td>
    </tr>
    </tbody>
    </table>

    启动容器时会将推理芯片和数据路径挂载到容器中。可根据需要通过修改**docker\_start\_infer.sh**的device来指定挂载的推理芯片。

    ```
    docker run -it \
      --device=/dev/davinci0 \         # 可根据需要修改挂载的npu设备
      --device=/dev/davinci_manager \
      --device=/dev/devmm_svm \
      --device=/dev/hisi_hdc \
      -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
      -v ${data_path}:${data_path} \
      ${docker_image} \
      /bin/bash
    ```

    > ![输入图片说明](https://images.gitee.com/uploads/images/2021/0926/181445_0077d606_8725359.gif "icon-note.gif") **说明：**
    > MindX SDK开发套件（mxManufacture）已安装在基础镜像中，安装路径：“/usr/local/sdk\_home“。

## 模型转换

在容器内进行模型转换。

1. 准备模型文件。

2. 模型转换。

    进入“infer/convert“目录进行模型转换，转换详细信息可查看转换脚本，**在convert.sh**脚本文件中，配置相关参数。

    ```
    air_path=$1
    om_path=$2
    /usr/local/Ascend/atc/bin/atc \
      --model=${air_path} \                     # 待转换的air模型
      --framework=1 \                           # 1代表MindSpore。
      --output=${om_path} \                     # 转换后输出的om模型。
      --input_shape="input:100,1,28,28" \       # 输入数据的shape。input取值根据实际使用场景确定。
      --soc_version=Ascend310 \                 # 模型转换时指定芯片版本
    exit 0
    ```

    模型转换命令如下（在infer/data/目录下创建model目录）。

    **bash convert.sh** _air\_path_ _om\_path_

    air\_path：AIR文件路径

    om\_path：生成om文件路径，转换脚本会在此基础上添加om后缀。

    示例：

    bash convert.sh ./protonet.air ../data/model/

3. 数据预处理,处理后的数据集放在infer/data/input/目录下。

    ```
    cd infer/convert
    bash dataprocess.sh
    ```

## mxBase推理

在容器内用mxBase进行推理。

1. 配置相关环境变量

    ```bash
          export ASCEND_VERSION=nnrt/latest
          export ARCH_PATTERN=.
          export MXSDK_OPENSOURCE_DIR=${MX_SDK_HOME}/opensource
          export LD_LIBRARY_PATH="${MX_SDK_HOME}/lib/plugins:${MX_SDK_HOME}/opensource/lib64:${MX_SDK_HOME}/lib:${MX_SDK_HOME}/lib/modelpostprocessors:${MX_SDK_HOME}/opensource/lib:${LD_LIBRARY_PATH}"
    ```

2. 修改配置文件。
3. 编译工程。

    ```
    cd mxbase
    bash build.sh
    ```

4. 运行推理服务。

    ```
    ./protonet
    ```

5. 观察结果，结果会保存在result目录中。

## MindX SDK推理

1. 修改配置文件。
    可根据实际情况修改pipeline文件。

    ```
      cd infer/sdk
      vim ../data/config/protonet.pipeline
      ```

      以protonet.pipeline文件为例，作简要说明。

    ```

    {
    "protonet": {
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
            "modelPath": "../data/model/protonet.om",
            "dataSource": "appsrc0",
            "waitingTime": "2000",
            "outputDeviceId": "-1"
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

2. 模型推理。

    1. 执行推理

    ```
    cd sdk
    bash run.sh ../data/config/protonet.pipeline \               # pipeline文件路径
                ../data/input/data_preprocess_Result/ \          # 预处理数据集
                > infer_result.txt                               # 保存推理结果
    ```

    2. 查看推理结果及性能。

    ```
    cat infer_result.txt
    ```

3. 执行精度测试。

    ```

    python postprocess.py --result_path=./infer/sdk/result \
                        --label_classes_path=./infer/data/input/label_classes_preprocess_Result \
                          > infer/sdk/infer_accuracy.txt
    ```

4. 查看精度结果。

    ```
    cd sdk
    cat infer_accuracy.txt
    ```
