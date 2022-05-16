# TextCNN模型推理任务交付件

**发布者（Publisher**：Huawei

**应用领域（Application Domain）**：Natural Language Processing

**版本（Version）**：0.1

**修改时间（Modified）**：2021.09.27

**大小（Size）**：3.28 MB \(air\)/10.00 MB \(ckpt\)/3.14 MB \(om\)

**框架（Framework）**：MindSpore\_1.3.0

**模型格式（Model Format）**：air/ckpt/om

**精度（Precision）**：Mixed/FP16

**处理器（Processor）**：昇腾910/昇腾310

**应用级别（Categories）**：Official

**描述（Description）**：基于MindSpore框架的TextCNN的语言分类网络模型训练并保存模型，通过ATC工具转换，可在昇腾AI设备上运行推理

## 概述

### 简述

TextCNN是一种使用卷积神经网络对文本进行分类的算法，广泛应用于文本分类的各种任务。文本神经网络的各个模块可以独立完成文本分类任务，便于分布式配置和并行执行，非常适合对短文本进行语义分析，比如微博/新闻/电子商务评论/视频弹幕评论等。

- 参考论文：[Kim Y. Convolutional neural networks for sentence classification[J]. arXiv preprint arXiv:1408.5882, 2014.](https://arxiv.org/abs/1408.5582)
- 参考实现：[https://gitee.com/mindspore/models/tree/master/official/nlp/textcnn](https://gitee.com/mindspore/models/tree/master/official/nlp/textcnn)

通过Git获取对应commit_id的代码方法如下：

```bash
git clone {repository_url}        # 克隆仓库的代码
cd {repository_name}              # 切换到模型的代码仓目录
git checkout {branch}             # 切换到对应分支
git reset --hard {commit_id}      # 代码设置到对应的commit_id
cd {code_path}                    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
```

### 默认配置

1、数据集

数据集基于原始论文中使用的数据集[Movie Review Data](http://www.cs.cornell.edu/people/pabo/movie-review-data/)，可以从链接网站下载MR数据压缩包，共1.18M，text文本格式，包括5331条正面评价语句和5331条负面评价语句。

2、训练/评估/推理超参

训练、评估和推理操作涉及的主要超参如下，更为全面的参数配置请参考textcnn_for_mindspore_{version}_code 目录下的 *.yaml 文件内容。

```yaml
'pre_trained': 'False'    # whether training based on the pre-trained model
'nump_classes': 2         # the number of classes in the dataset
'batch_size': 64          # training batch size
'epoch_size': 4           # total training epochs
'weight_decay': 3e-5      # weight decay value
'data_path': './data/'    # absolute full path to the train and evaluation datasets
'device_target': 'Ascend' # device running the program
'device_id': 0            # device ID used to train or evaluate the dataset. Ignore it when you use run_train.sh for distributed training
'keep_checkpoint_max': 1  # only keep the last keep_checkpoint_max checkpoint
'checkpoint_path': './train_textcnn.ckpt'  # the absolute full path to save the checkpoint file
'word_len': 51            # The length of the word
'vec_length': 40          # The length of the vector
'base_lr': 1e-3          # The base learning rate
```

## 准备工作

### 推理环境准备

- 硬件环境、开发环境和运行环境准备请参见[《CANN 软件安装指南](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=installation-upgrade)》。
- 宿主机上需要安装Docker并登录[Ascend Hub中心](https://ascendhub.huawei.com/#/home)获取镜像。

当前模型支持的镜像列表如下表所示。

**表 1**  镜像列表

| 镜像名称           | 镜像版本                | 配套CANN版本            |
| -------------- | -------------------------------------------- |------------|
| ARM/x86架构：[infer-modelzoo](https://ascendhub.huawei.com/#/detail/infer-modelzoo) | 21.0.2 | [5.0.2](https://www.hiascend.com/software/cann/commercial)    |

### 源码介绍

源码目录结构如下图所示：

```bash
/home/HwHiAiUser/textcnn_for_mindspore_{version}_code
|-- infer
|   |-- Dockerfile
|   |-- README.md        # 代码说明
|   |-- convert          # AIR to OM 转换命令
|   |   `-- convert.sh
|   |-- data             # 数据/参数管理目录
|   |   `-- config           # 配置文件目录
|   |       `-- textcnn.pipeline
|   |-- docker_start_infer.sh     # 启动容器脚本
|   |-- mxbase           # mxbase推理目录
|   |   |-- CMakeLists.txt
|   |   |-- build.sh
|   |   `-- src              # C++及头文件源码目录
|   |       |-- TextCnnBase.cpp
|   |       |-- TextCnnBase.h
|   |       `-- main.cpp
|   `-- sdk              # 基于sdk.run包推理
|       |-- main.py              # sdk推理脚本
|       `-- predata.py           # 语料预处理脚本
```

## 推理

### 准备推理数据

准备模型转换和模型推理所需目录及数据。

1、下载源码包。（例如：单击“下载模型脚本”和“下载模型”，下载所需软件包。）

2、将源代码(infer)上传至服务器任意目录（如：/home/HwHiAiUser/textcnn_for_mindspore_{version}_code）,并进入该目录。

3、准备数据

由于后续推理均在容器中进行，因此需要把用于推理的图片、数据集、模型文件、代码等均放在同一数据路径中，后续示例将以“/home/HwHiAiUser/textcnn_for_mindspore_{version}_code“为例。

```bash
infer
|-- READMD.md        # 代码说明
|-- Dockerfile
|-- docker_start_infer.sh
|-- convert          # AIR to OM 转换脚本目录
|   |-- convert.sh
|   `-- textcnn.air
|-- data             # 数据/参数管理目录
|   |-- config           # 配置文件目录
|   |   |-- infer_label.txt  #用于推理的分类标签文件
|   |   `-- textcnn.pipeline
|   |-- ids              # 预处理操作生成的ids数据目录
|   |-- input            # MR原始数据，作为推理数据预处理的输入
|   |   |-- rt-polarity.neg
|   |   `-- rt-polarity.pos
|   |-- labels           # 预处理操作生成的lables目标数据
|   `-- model            # OM模型文件
|       `-- textcnn.om
|-- mxbase           # mxbase推理目录
|   |-- CMakeLists.txt
|   |-- build.sh
|   `-- src              # C++及头文件源码目录
|       |-- TextcnnBase.cpp
|       |-- TextcnnBase.h
|       `-- main.cpp
|-- output
|   |-- mxbase_result.txt    # mxbase推理结果
|   `-- sdk_result           # sdk推理结果
`-- sdk
    |-- main.py              # sdk推理脚本
    `-- predata.py           # 语料预处理脚本
```

执行推理之前，需进行数据集语料预处理。数据集使用[Movie Review Data](http://www.cs.cornell.edu/people/pabo/movie-review-data/)，可以从链接网站下载MR数据压缩包，解压后将rt-polarity.neg 和 rt-polarity.pos文件保存在 ./infer/data/input目录下（若上述目录不存在需手工创建）。

在shell中执行：

```bash
cd sdk
python3 predata.py
```

预处理后的语料数据保存于 ./infer/data/ids 和 ./infer/data/labels 目录下。

准备infer_label.txt标签文件，保存于./infer/data/config目录下，若该目录不存在请手动创建。

此环节所准备数据适用于 SDK 和 mxbase 两种推理方式。

4、启动容器

执行以下命令，启动容器实例。

```bash
bash docker_start_infer.sh docker_image:tag model_dir
```

**表 3**  容器启动参数说明

| 参数           | 说明                                         |
| -------------- | -------------------------------------------- |
| *docker_image* | 推理镜像名称，推理镜像请从Ascend Hub上下载。 |
| *tag*          | 镜像tag，请根据实际配置，如：21.0.1。        |
| *model_dir*    | 推理容器挂载路径，本例中为/home/data         |

启动容器时会将推理芯片和数据路径挂载到容器中。

其中docker_start_infer.sh(/infer/docker_start_infer.sh)内容如下。

docker_start_infer.sh文件内容

```shell
#!/bin/bash
docker_image=$1
model_dir=$2

if [ -z "${docker_image}" ]; then
    echo "please input docker_image"
    exit 1
fi

if [ ! -d "${model_dir}" ]; then
    echo "please input model_dir"
    exit 1
fi

docker run -it \
           --device=/dev/davinci0 \  #请根据芯片的情况更改
           --device=/dev/davinci_manager \
           --device=/dev/devmm_svm \
           --device=/dev/hisi_hdc \
           -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
           -v ${model_dir}:${model_dir} \
           ${docker_image} \
           /bin/bash
```

### 模型转换

1、准备模型文件

将模型训练生成的checkpoint文件执行export冻结操作后输出的textcnn.air文件 ./infer/convert 目录下。

2、模型转换

进入 ./infer/convert 目录进行模型转换，转换详细信息可查看转换脚本; 在**atc.sh**脚本文件中，配置相关参数。

执行转换脚本 bash convert.sh textcnn.air textcnn。使用mkdir命令创建 ./infer/data/model 目录，将转换生成的OM模型文件textcnn.om放入model目录下。

convert.sh文件内容

```bash
#!/bin/bash
model_path=$1
output_model_name=$2
/usr/local/Ascend/atc/bin/atc \
  --model=${model_path} \          # 待转换的air模型，模型可以通过训练生成或通过“下载模型”获得。
  --framework=1 \                  # 1代表MindSpore。
  --soc_version=Ascend310 \        # 模型转换时指定芯片版本。
  --output=${output_model_name}    # 转换后输出的om模型。
exit 0
```

转换命令如下：

**bash atc.sh** _model\_path_ _output\_model\_name_

**表 4** 模型转换参数说明

| 参数 | 说明 |
| --------- | --------- |
| model_path | AIR文件路径 |
| output_model_name | 生成的OM文件名，转换脚本会在此基础上添加.om后缀 |

### mxBase推理

在容器内用mxBase进行推理，并参考以下步骤执行：

1、添加环境变量。

通过**vi ~/.bashrc**命令打开~/.bashrc文件，将下面的环境变量添加进当前环境，添加好环境变量以后退出文件编辑，执行**source ~/.bashrc**使环境变量生效。

```bash
export ASCEND_HOME="/usr/local/Ascend"
export ASCEND_VERSION="nnrt/latest"
export ARCH_PATTERN="."
export LD_LIBRARY_PATH="${MX_SDK_HOME}/lib/modelpostprocessors:${LD_LIBRARY_PATH}"
export MXSDK_OPENSOURCE_DIR="${MX_SDK_HOME}/opensource"
```

2、准备标签文件

执行推理之前，用户需事先准备匹配训练集的分类标签文件，如本示例中的infer_label.txt，保存于./infer/data/config目录下，若目录不存在请手动创建。

3、编译工程

进入 ./infer/mxbase 目录，执行指令**bash build.sh**

shell脚本build.sh内容如下：

```bash
path_cur=$(dirname $0)
echo "current dir is: ${path_cur}"
ASCEND_HOME=$LOCAL_ASCEND

function check_env()
{
    # set ASCEND_VERSION to ascend-toolkit/latest when it was not specified by user
    if [ ! "${ASCEND_VERSION}" ]; then
        export ASCEND_VERSION=ascend-toolkit/latest
        echo "Set ASCEND_VERSION to the default value: ${ASCEND_VERSION}"
    else
        echo "ASCEND_VERSION is set to ${ASCEND_VERSION} by user"
    fi

    if [ ! "${ARCH_PATTERN}" ]; then
        # set ARCH_PATTERN to ./ when it was not specified by user
        export ARCH_PATTERN=./
        echo "ARCH_PATTERN is set to the default value: ${ARCH_PATTERN}"
    else
        echo "ARCH_PATTERN is set to ${ARCH_PATTERN} by user"
    fi
}

function build_textcnn()
{
    cd $path_cur
    old_mxbaseResult="../output/mxbase_result.txt"
    if [ -f "$old_mxbaseResult" ]; then
        rm -rf $old_mxbaseResult
        echo "delete $old_mxbaseResult success! "
    fi
    rm -rf build
    mkdir -p build
    cd build
    cmake ..
    make
    ret=$?
    if [ ${ret} -ne 0 ]; then
        echo "Failed to build textcnn."
        exit ${ret}
    fi
    make install
}

check_env
build_textcnn
```

4、运行推理服务

在 mxbase目录下执行 **./textcnn ../data/ 1**，推理结果保存在 ./infer/output/目录下的mxbase_result.txt文件。

5、观察结果

关于推理精度及性能统计结果，用户可以直接观察屏幕打印输出。结果如下示例：

```bash
I0909 08:38:57.929585 81602 main.cpp:106] ==============================================================
I0909 08:38:57.929594 81602 main.cpp:108] Accuracy: 0.774345
I0909 08:38:57.929630 81602 main.cpp:110] Precision: 0.773832
I0909 08:38:57.929636 81602 main.cpp:112] Recall: 0.775281
I0909 08:38:57.929641 81602 main.cpp:113] F1 Score: 0.774556
I0909 08:38:57.929654 81602 main.cpp:114] ==============================================================
I0909 08:38:57.934760 81602 DeviceManager.cpp:90] DestroyDevices begin
I0909 08:38:57.934777 81602 DeviceManager.cpp:92] destroy device:0
I0909 08:38:58.554832 81602 DeviceManager.cpp:98] aclrtDestroyContext successfully!
I0909 08:38:59.007510 81602 DeviceManager.cpp:106] DestroyDevices successfully
I0909 08:38:59.007552 81602 main.cpp:122] Infer images sum 1068, cost total time: 718.826 ms.
I0909 08:38:59.007570 81602 main.cpp:123] The throughput: 1485.76 bin/sec.
```

### MindX SDK推理

在容器内执行SDK推理，并参考以下步骤执行：

1、准备配置文件

使用mkdir命令创建 ./infer/data/config 目录，预先准备MindX SDK编排配置文件 textcnn.pipeline 存放于 config 目录下。 textcnn.pipeline 文件内容如下：

```json
{
    "im_bertbase": {
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
                "modelPath": "../data/model/textcnn.om"   //示例文件，用户可根据实际情况替换
            },
            "factory": "mxpi_tensorinfer",
            "next": "mxpi_dataserialize0"
        },
        "mxpi_dataserialize0": {
            "props": {
                "outputDataKeys": "mxpi_tensorinfer0"
            },
            "factory": "mxpi_dataserialize",
            "next": "appsink0"
        },
        "appsink0": {
            "factory": "appsink"
        }
    }
}
```

2、性能统计开关

SDK推理服务本身具备推理性能统计功能，用户打开性能统计开关可使用该功能。正确安装SDK推理服务软件包之后
，将sdk.conf配置文件中“enable_ps”参数设置为true，“ps_interval_time”参数设置为6。操作示例如下：

```bash
vim /home/HwHiAiUser/mxManufacture/config/sdk.conf
```

```json
# MindX SDK configuration file

# whether to enable performance statistics, default is false [dynamic config]
enable_ps = true
...
ps_interval_time = 1
...
```

3、运行推理服务

进入 ./infer/sdk 子目录，运行SDK推理脚本开启推理过程。

SDK推理命令如下：

```bash
cd sdk
python3 main.py
```

SDK推理结果保存于 ./infer/output/sdk_result 目录下。

4、观察运行结果和性能统计

 关于推理精度统计结果，用户可以直接观察屏幕打印输出。如下示例：

```bash
==============================================================
Total ids input is: 1068,    NegNum: 534,    PosNum: 534
TP=414,    FP=121,    FN= 120
Accuracy:  0.774345
Precision:  0.773832
Recall:  0.775281
F1:  0.774556
==============================================================
Infer images sum: 1068, cost total time: 1.325859 sec.
The throughput: 805.515766 bin/sec.
```

若用户在前述步骤2中开启了SDK推理性能统计功能，可在日志目录“/home/HwHiAiUser/mxManufacture/logs/”查看性能统计结果。

```bash
performance—statistics.log.e2e.xxx
performance—statistics.log.plugin.xxx
performance—statistics.log.tpr.xxx
```

其中e2e日志统计端到端时间，plugin日志统计单插件时间。

## 在ModelArts上应用

### 创建OBS桶

ModelArts环境下执行模型训练及冻结操作时，首先应在华为云对象存储服务OBS下创建桶，其后在桶内创建code、data等文件夹用户输入输出数据存储。

1. 创建桶

    登录[OBS管理控制台](https://storage.huaweicloud.com/obs)，创建对象存储服务OBS桶。具体请参见[创建桶](https://support.huaweicloud.com/usermanual-obs/obs_03_0306.html)介绍。例如，本示例创建了名称为“textcnn4modelarts”的OBS桶。创建桶的区域需要与ModelArts所在的区域一致，本例ModelArts在华北-北京四区域，在创建桶时也选择了华北-北京四。

2. 创建文件夹存放数据

    创建用于存放数据的文件夹，具体请参见[新建文件夹](https://support.huaweicloud.com/usermanual-obs/obs_03_0316.html)章节。本示例在已创建的OBS桶“textcnn4modelarts”中创建如下模型目录。

    ![obs_folders.jpg](./img/obs_folders.jpg)

    目录结构示例：

    - code：存放训练脚本及参数配置目录
    - data：存放训练数据集目录
    - logs：存放训练日志目录
    - output：训练生成ckpt和air模型目录

    将TextCNN模型的代码文件夹直接上传至OBS桶code目录。数据集使用[Movie Review Data](http://www.cs.cornell.edu/people/pabo/movie-review-data/)，可以从链接网站下载MR数据压缩包，解压后将rt-polarity.neg 和 rt-polarity.pos文件保存在OBS桶data目录下。由于文本编码格式兼容性问题，部分用户运行模型训练脚本时出现MR数据集文本文件打开错误提示，并导致模型训练终止。遇到此种情况，用户可以编辑OBS桶code目录下 ./src/dataset.py 脚本文件，将第132行内容：

    ```python
              with codecs.open(filename, 'r') as f:
    ```

    修订为：

    ```python
              with codecs.open(filename, 'r', encoding='ISO-8859-1') as f:
    ```

    此外，为方便用户，本项目在./modelarts目录下同时放置了修订后的dataset.py文件，运行模型训练脚本前用此文件覆盖./src目录下的同名文件，也能解决上述问题。

### 创建算法

1. 使用华为云帐号登录[ModelArts管理控制台](https://console.huaweicloud.com/modelarts)，在左侧导航栏中选择“算法管理”。
2. 在“我的算法管理”界面，单击左上角“创建”，进入“创建算法”页面。
3. 在“创建算法”页面，填写相关参数，然后单击“提交”。
    1. 设置算法基本信息。

    2. 设置“创建方式”为“自定义脚本”。

        用户需根据实际算法代码情况设置“AI引擎”、“代码目录”和“启动文件”。选择的AI引擎和编写算法代码时选择的框架必须一致。例如编写算法代码使用的是MindSpore，则在创建算法时也要选择MindSpore。

        ![create_algorithm.jpg](./img/create_algorithm.jpg)

        **表 5**  参数说明

        | 参数 <div style="width:100px">   | 说明                                                         |
        | ------------- | ------------------------------------------------------------ |
        | *AI引擎* | Ascend-Powered-Engine，mindspore_1.3.0-cann_5.0.2                                 |
        | *代码目录*  | 算法代码存储的OBS路径。上传训练脚本，如：/obs桶/textcnn4modelarts/code |
        | *启动文件*         | 启动文件：启动训练的python脚本，如：/obs桶/textcnn4modelarts/code/train_start.py |
        | *输入数据配置*       | 代码路径参数：data_url |
        | *输出数据配置*       | 代码路径参数：train_url |

    3. 设置超参数

        ModelArts模型训练超参数设置方式比较灵活，可以选择在线配置，也可以选择脚本代码读取yaml配置文件方式。本项目采用后者，脚本参数从桶内code目录下 mr_config.yaml 文件读取。用户如需设定或更改脚本参数，请在脚本运行前修订mr_config.yaml相关字段。

        本示例中在 ./modelarts 目录放置有修订后的mr_config.yaml样本文件，在ModelArts云平台使用时可直接替换/home/HwHiAiUser/textcnn_for_mindspore_{version}_code目录下的同名文件。

        mr_config.yaml 样本文件主要内容如下：

        ```yaml
        # Builtin Configurations(DO NOT CHANGE THESE CONFIGURATIONS unless you know exactly what you are doing)
        enable_modelarts: False
        # Url for modelarts
        data_url: ""
        train_url: ""
        checkpoint_url: ""
        # Path for local
        data_path: "/cache/data"
        output_path: "/cache/train"
        load_path: "/cache/checkpoint_path/"
        device_target: 'Ascend'
        device_id: 0
        enable_profiling: False

        # ==============================================================================
        # Training options
        dataset: 'MR'
        pre_trained: False
        num_classes: 2
        batch_size: 1
        epoch_size: 1
        weight_decay: 3e-5
        keep_checkpoint_max: 1
        checkpoint_path: "./checkpoint/"
        word_len: 51
        vec_length: 40
        base_lr: 1e-3
        label_dir: ""
        result_dir: ""
        result_path: "./preprocess_Result/"

        # Export options
        file_name: "textcnn"
        file_format: "MINDIR"
        ```

### 创建训练作业

1. 使用华为云帐号登录[ModelArts管理控制台](https://console.huaweicloud.com/modelarts)，在左侧导航栏中选择“训练管理 > 训练作业”，默认进入“训练作业”列表。

2. 单击“创建训练作业”，进入“创建训练作业”页面，在该页面填写训练作业相关参数。

3. 在创建训练作业页面，填写训练作业相关参数，然后单击“下一步”。

   本步骤只提供训练任务部分参数配置说明，其他参数配置详情请参见《[ModelArts AI 工程师用户指南](https://support.huaweicloud.com/modelarts/index.html)》中“使用自定义镜像训练模型”章节。

   a. 填写基本信息

   ​   基本信息包含“名称”和“描述”。

   b. 填写作业参数

      包含数据来源、算法来源等关键信息。本步骤只提供训练任务部分参数配置说明，其他参数配置详情请参见[《ModelArts AI 工程师用户指南](https://support.huaweicloud.com/modelarts/index.html)》中“训练模型（new）”。

      **MindSpore创建训练作业步骤**

      ![train_task.jpg](./img/train_task.jpg)

      **表 6**  参数说明

     | 参数名称<div style="width:100px"> | 子参数<div style="width:100px">  | 说明 |
     | ------- | ------- | --- |
     | *算法* | *我的算法*     | 选择“我的算法”页签，勾选上文中创建的算法。如果没有创建算法，请单击“创建”进入创建算法页面，详细操作指导参见“创建算法”。 |
     | *训练输入*  | *数据来源*      | 选择OBS上数据集存放的目录，如：/obs桶/textcnn4modelarts/data |
     | *训练输出*  | *模型输出*      | 选择OBS上训练结果的存储位置，如：/obs桶/textcnn4modelarts/output |
     | *规格*      | *-*      | Ascend: 1*Ascend 910(32GB) | ARM: 24 核 96GB |
     | *作业日志路径*  | *-*      | 设置训练日志存放的目录，如：/obs桶/textcnn4modelarts/logs，请注意选择的OBS目录有读写权限。 |

     本项目中算法训练涉及的超参数采用yaml配置文件方式导入，在训练作业的超参设置输入窗口仅需指定训练输入data_url和训练输出train_url对应的OBS目录，其它超参毋须重新设置。

4. 单击“提交”，完成训练作业的创建。

    训练作业一般需要运行一段时间，根据您选择的数据量和资源不同，训练时间将耗时几分钟到几十分钟不等。

### 查看训练任务日志

1. 在ModelArts管理控制台，在左侧导航栏中选择“训练管理 \> 训练作业（New）”，默认进入“训练作业”列表。
2. 在训练作业列表中，您可以单击作业名称，查看该作业的详情。

    详情中包含作业的基本信息、训练参数、日志详情和资源占用情况，如下图所示：

    ![train_logs.jpg](./img/train_logs.jpg)

### 查看训练模型

训练完成后，模型输出checkpoint文件保存于OBS桶output目录；其后脚本自动执行模型冻结操作，生成AIR模型文件。下图示例中的模型文件保存于obs://textcnn4modelarts/output/0/checkpoint目录下。

![ckpt_air_model.jpg](./img/ckpt_air_model.jpg)