# MCNN模型交付件-众智

## 交付件基本信息

**发布者（Publisher）**：Huawei

**应用领域（Application Domain）**：Crowd Counting

**版本（Version）**：1.1

**修改时间（Modified）**：2020.12.4

**大小（Size）**：501 KB \(ckpt\)/541 KB \(air\)/916 KB \(om\)

**框架（Framework）**：MindSpore

**模型格式（Model Format）**：ckpt/air/om

**精度（Precision）**：Mixed/FP16

**处理器（Processor）**：昇腾910/昇腾310

**应用级别（Categories）**：Released

Released（发行版模型）：昇腾推荐使用，支持训练推理端到端流程。

【A类Research; B类性能<1.8倍用Official，\>1.8倍用Benchmark; C类Released】

**描述（Description）**：MCNN是一种多列卷积神经网络，可以从几乎任何角度准确估计单个图像中的人群数量。

## 概述

### 简述

MCNN包含三个平行CNN，其滤波器具有不同大小的局部感受野。为了简化，除了过滤器的大小和数量之外，我们对所有列（即conv–pooling–conv–pooling）使用相同的网络结构。每个2×2区域采用最大池，由于校正线性单元（ReLU）对CNN具有良好的性能，因此采用ReLU作为激活函数。

- 参考论文：

    [Yingying Zhang, Desen Zhou, Siqin Chen, Shenghua Gao, Yi Ma. Single-Image Crowd Counting via Multi-Column Convolutional Neural Network](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper.pdf)

- 参考实现：[https://github.com/svishwa/crowdcount-mcnn](https://github.com/svishwa/crowdcount-mcnn)

通过Git获取对应commit\_id的代码方法如下：

``` python
git clone {repository_url}    # 克隆仓库的代码
cd {repository_name}    # 切换到模型的代码仓目录
git checkout  {branch}    # 切换到对应分支
git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
```

### 默认配置

- 训练集预处理 :

    原始数据集的每张图片尺寸各不相同

    预处理做的事情是将这些图片的长宽都调整到256

- 测试集预处理 :

    原始测试集的每张图片尺寸各不相同

    预处理做的事情是将这些图片的长宽都调整到1024（为了和om模型对应）

- 训练超参 :

    Batch size : 1

    Learning rate : 0.000028

    Momentum : 0.0

    Epoch_size : 800

    Buffer_size : 1000

    Save_checkpoint_steps : 1

    Keep_checkpoint_max : 10

    Air_name : "mcnn"

### 支持特性

支持的特性包括：1、分布式并行训练。

### 分布式训练

MindSpore支持数据并行及自动并行。自动并行是MindSpore融合了数据并行、模型并行及混合并行的一种分布式并行模式，可以自动建立代价模型，为用户选择一种并行模式。相关代码示例。

``` python
context.set_auto_parallel_context(device_num=args_opt.device_num, parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
```

## 准备工作

### 推理环境准备

- 硬件环境、开发环境和运行环境准备请参见[《CANN 软件安装指南](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=installation-upgrade)》。
- 宿主机上需要安装Docker并登录[Ascend Hub中心](https://ascendhub.huawei.com/#/home)获取镜像。

    当前模型支持的镜像列表如下表所示。

    **表 1**  镜像列表

    <a name="zh-cn_topic_0000001205858411_table1519011227314"></a>
    <table><thead align="left"><tr id="zh-cn_topic_0000001205858411_row0190152218319"><th class="cellrowborder" valign="top" width="55.00000000000001%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001205858411_p1419132211315"><a name="zh-cn_topic_0000001205858411_p1419132211315"></a><a name="zh-cn_topic_0000001205858411_p1419132211315"></a>镜像名称</p>
    </th>
    <th class="cellrowborder" valign="top" width="20%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001205858411_p75071327115313"><a name="zh-cn_topic_0000001205858411_p75071327115313"></a><a name="zh-cn_topic_0000001205858411_p75071327115313"></a>镜像版本</p>
    </th>
    <th class="cellrowborder" valign="top" width="25%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001205858411_p1024411406234"><a name="zh-cn_topic_0000001205858411_p1024411406234"></a><a name="zh-cn_topic_0000001205858411_p1024411406234"></a>配套CANN版本</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="zh-cn_topic_0000001205858411_row71915221134"><td class="cellrowborder" valign="top" width="55.00000000000001%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001205858411_p58911145153514"><a name="zh-cn_topic_0000001205858411_p58911145153514"></a><a name="zh-cn_topic_0000001205858411_p58911145153514"></a>ARM/x86架构：<a href="https://ascendhub.huawei.com/#/detail/infer-modelzoo" target="_blank" rel="noopener noreferrer">infer-modelzoo</a></p>
    </td>
    <td class="cellrowborder" valign="top" width="20%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001205858411_p14648161414516"><a name="zh-cn_topic_0000001205858411_p14648161414516"></a><a name="zh-cn_topic_0000001205858411_p14648161414516"></a>21.0.2</p>
    </td>
    <td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001205858411_p1264815147514"><a name="zh-cn_topic_0000001205858411_p1264815147514"></a><a name="zh-cn_topic_0000001205858411_p1264815147514"></a><a href="https://www.hiascend.com/software/cann/commercial" target="_blank" rel="noopener noreferrer">5.0.2</a></p>
    </td>
    </tr>
    </tbody>
    </table>

### 源码介绍

脚本和示例代码

``` python
/MCNN
├── infer                # MindX高性能预训练模型新增  
│   ├── convert          # 转换om模型命令
│   │   └──convert_om.sh
│   ├── model            # 存放模型
│   ├── test_data        # 存放数据集
│   ├── mxbase           # 基于mxbase推理
│   │   ├── src
│   │   │   ├── Mcnn.cpp
│   │   │   ├── Mcnn.h
│   │   │   └── main.cpp
│   │   ├── output       # 存放结果路径
│   │   ├── CMakeLists.txt
│   │   └── build.sh
│   └── sdk               # 基于sdk.run包推理；如果是C++实现，存放路径一样
│   │   ├── out           # 存放结果路径
│   │   ├── main.py
│   │   ├── mcnn.pipeline
│   │   └── run.sh
│   └──docker_start_infer.sh     # 启动容器脚本
```

## 推理

- **[准备推理数据](#准备推理数据.md)**  

- **[模型转换](#模型转换.md)**  

- **[mxBase推理](#mxBase推理.md)**  

- **[MindX SDK推理](#MindX-SDK推理.md)**  

### 准备推理数据

准备模型转换和模型推理所需目录及数据。

1. 下载源码包。

    单击“下载模型脚本”和“下载模型”，下载所需软件包。

2. 将源码上传至推理服务器任意目录并解压（如：“/home/MCNN“）。【【【添加以下命令，最终下载模型脚本"步骤得到的文件需要进行格式转换】】】

    ```
    #在环境上执行
    unzip MCNN_Mindspore_{version}_code.zip
    cd {code_unzip_path}/MCNN_MindSpore_{version}_code/infer && dos2unix `find .`
    ```

3. 准备数据。

    由于后续推理均在容器中进行，因此需要把用于推理的图片、数据集、模型文件、代码等均放在同一数据路径中，后续示例将以“/home/MCNN“为例。

    ```
    ..
    /MCNN
    ├── infer                # MindX高性能预训练模型新增  
    │   ├── convert          # 转换om模型命令
    │   │   └──convert_om.sh
    │   ├── model            # 存放模型
    │   ├── test_data        # 存放数据集
    │   ├── mxbase           # 基于mxbase推理
    │   │   ├── src
    │   │   │   ├── Mcnn.cpp
    │   │   │   ├── Mcnn.h
    │   │   │   └── main.cpp
    │   │   ├── output       # 存放结果路径
    │   │   ├── CMakeLists.txt
    │   │   └── build.sh
    │   └── sdk               # 基于sdk.run包推理；如果是C++实现，存放路径一样
    │   │   ├── out           # 存放结果路径
    │   │   ├── main.py
    │   │   ├── mcnn.pipeline
    │   │   └── run.sh
    │   └──docker_start_infer.sh     # 启动容器脚本
    ```

    AIR模型可通过“模型训练”后转换生成或通过“下载模型”获取。

    将shanghaitechA数据集test_data放到“/MCNN/infer/”目录下。

    数据集链接: https://pan.baidu.com/s/185jBeL91R85OUcbeARP9Sg 提取码: 5q9v

4. 准备mxbase的输出目录/output。该目录下存储推理的中间结果。

    ```
    #在环境上执行
    cd MCNN_Mindspore_{version}_code/infer/mxbase/
    mkdir output
    ```

5. 启动容器。

    进入“infer“目录，执行以下命令，启动容器。

    bash docker_start_infer.sh docker_image data_path

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
    <tr id="row5835194195611"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p59018537424"><a name="p59018537424"></a><a name="p59018537424"></a>data_path</p>
    </td>
    <td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1390135374214"><a name="p1390135374214"></a><a name="p1390135374214"></a>数据集路径。</p>
    </td>
    </tr>
    </tbody>
    </table>

    启动容器时会将推理芯片和数据路径挂载到容器中。可根据需要通过修改**docker\_start\_infer.sh**的device来指定挂载的推理芯片。

    ```
    docker run -it \
      --device=/dev/davinci0 \        # 可根据需要修改挂载的npu设备
      --device=/dev/davinci_manager \

    # 说明：MindX SDK开发套件（mxManufacture）已安装在基础镜像中，安装路径：“/usr/local/sdk\_home“。
    ```

### 模型转换

1. 准备模型文件。

    在infer目录下创建model目录，将mcnn.air放至此目录下。

    ```
    cd MCNN_Mindspore_{version}_code/infer/
    mkdir model
    ```

    将mcnn.air放至infer/model/下。

2. 模型转换。

    进入“infer/mcnn/convert“目录进行模型转换，**convert_om.sh**脚本文件中，配置相关参数。

    ```
    atc \
                    --model=${model} \
                    --output=${output} \
                    --soc_version=${soc_version} \
                    --input_shape=${input_shape} \
                    --framework=1 \
                    --input_format=NCHW
    ```

    转换命令如下。

    ```
    bash convert_om.sh --model=[model_path] --output=[output_model_name]
    # model_path为air模型路径 output_model_name为新模型的名字
    # 例子如下：
    bash convert_om.sh --model=../model/mcnn.air --output=../model/mcnn
    ```

    **表 1**  参数说明

    <a name="table15982121511203"></a>
    <table><thead align="left"><tr id="row1598241522017"><th class="cellrowborder" valign="top" width="40%" id="mcps1.2.3.1.1"><p id="p189821115192014"><a name="p189821115192014"></a><a name="p189821115192014"></a>参数</p>
    </th>
    <th class="cellrowborder" valign="top" width="60%" id="mcps1.2.3.1.2"><p id="p1982161512206"><a name="p1982161512206"></a><a name="p1982161512206"></a>说明</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row0982101592015"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1598231542020"><a name="p1598231542020"></a><a name="p1598231542020"></a>model_path</p>
    </td>
    <td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p598231511200"><a name="p598231511200"></a><a name="p598231511200"></a>AIR文件路径。</p>
    </td>
    </tr>
    <tr id="row109831315132011"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p598319158204"><a name="p598319158204"></a><a name="p598319158204"></a>output_model_name</p>
    </td>
    <td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1898316155207"><a name="p1898316155207"></a><a name="p1898316155207"></a>生成的OM文件名，转换脚本会在此基础上添加.om后缀。</p>
    </td>
    </tr>
    </tbody>
    </table>

### mxBase推理

在容器内用mxBase进行推理。

1. 编译工程。

    进入/mxbase路径下

    ```
    bash build.sh
    ```

2. 运行推理服务。

    ```
    ./Mcnn [model_path] [data_path] [label_path] [output_path]
    # model_path为om模型路径 data_path为数据集数据部分路径 label_path为数据集标签部分路径 output_path为输出路径
    # 例子如下：
    ./Mcnn ../model/mcnn.om ../test_data/images/ ../test_data/ground_truth_csv/ ./output
    ```

3. 观察结果。

    推理结果的参数解释：

    datasize: 输入数据大小

    output_size: 输出数据大小

    output0_datatype: 输出数据类型

    output0_shape: 输出数据的形状

    output0_bytesize: 输出数据字节类型

    接下来的是测试图片的id、预测值、真实值。

    最后输出的是整个测试集的精度。

### MindX SDK推理

1. 检查环境。

    确保mcnn.om模型在/MCNN/infer/model下。

2. 修改配置文件。

    1. 修改pipeline文件。

    ```
    {
    "mcnn_opencv": {
        "appsrc0": {
            "factory": "appsrc",
            "next": "mxpi_tensorinfer0"
        },
        "mxpi_tensorinfer0": {
            "props": {
                "dataSource": "appsrc0",
                "modelPath": "../model/mcnn.om", //此处是你的模型存放路径
                "waitingTime": "2000"
            },
            "factory": "mxpi_tensorinfer",
            "next": "appsink0"
        },
        "appsink0": {
            "factory": "appsink"
        }
    }
    }
    ```

3. 打开性能统计开关。将"enable_ps"参数设置为true，"ps_interval_time"参数设置为6。

    vim ${MX_SDK_HOME}/config/sdk.conf

    ```
    # Mindx SDK configuration file
    # whether to enable performance statistics，default is false [dynamic config]
    enable_ps=true
    ...
    ps_interval_time=6
    ...
    ```

4. 运行推理服务。

    1. 执行推理。

        进入/sdk后，运行以下命令：

        ```
        bash run.sh [input_dir]  [gt_dir]
        # input_dir为数据集数据部分路径 gt_dir为数据集标签部分路径
        # 例如：
        bash run.sh ../test_data/images/ ../test_data/ground_truth_csv/
        ```

    2. 查看推理结果.

5. 执行精度和性能测试。

6. 在日志目录“$ {MX_SDK_HOME}/logs/”查看性能统计结果。

    ```
    performance-statistics.log.e2e.xxx
    performance-statistics.log.plugin.xxx
    performance-statistics.log.tpr.xxx
    ```

    其中e2e日志统计端到端时间,plugin日志统计单插件时间。

## 在ModelArts上应用

- **[创建OBS桶](#创建OBS桶.md)**  

- **[创建算法（适用于MindSpore和TensorFlow）](#创建算法（适用于MindSpore和TensorFlow）.md)**  

- **[创建训练作业](#创建训练作业.md)**  

- **[查看训练任务日志](#查看训练任务日志.md)**  

- **[迁移学习](#迁移学习.md)**  

### 创建OBS桶

1.登录[OBS管理控制台](https://storage.huaweicloud.com/obs)，创建OBS桶。具体请参见[创建桶](https://support.huaweicloud.com/usermanual-obs/obs_03_0306.html)章节。例如，创建名称为“MCNN”的OBS桶。

创建桶的区域需要与ModelArts所在的区域一致。例如：当前ModelArts在华北-北京四区域，在对象存储服务创建桶时，请选择华北-北京四。

创建用于存放数据的文件夹，具体请参见[新建文件夹](https://support.huaweicloud.com/usermanual-obs/obs_03_0316.html)章节。例如，在已创建的OBS桶中MCNN项目里创建data、LOG、train_output

目录结构说明：

- MCNN：存放训练脚本目录
- data：存放训练数据集目录
- LOG：存放训练日志目录
- train_output：存放训练ckpt文件和冻结的AIR模型（output中result文件夹中）

数据集shanghaitechA传至“data”目录。

数据集链接: https://pan.baidu.com/s/185jBeL91R85OUcbeARP9Sg 提取码: 5q9v

注意：需要将modlearts下的start_train.py移至根目录下进行训练。

### 创建算法（适用于MindSpore和TensorFlow）

1. 使用华为云帐号登录[ModelArts管理控制台](https://console.huaweicloud.com/modelarts)，在左侧导航栏中选择“算法管理”。
2. 在“我的算法管理”界面，单击左上角“创建”，进入“创建算法”页面。
3. 在“创建算法”页面，填写相关参数，然后单击“提交”。
    1. 设置算法基本信息。

    2. 设置“创建方式”为“自定义脚本”。

        用户需根据实际算法代码情况设置“AI引擎”、“代码目录”和“启动文件”。选择的AI引擎和编写算法代码时选择的框架必须一致。例如编写算法代码使用的是MindSpore，则在创建算法时也要选择MindSpore。

        **表 1**

        <a name="table09972489125"></a>
        <table><thead align="left"><tr id="row139978484125"><th class="cellrowborder" valign="top" width="29.470000000000002%" id="mcps1.2.3.1.1"><p id="p16997114831219"><a name="p16997114831219"></a><a name="p16997114831219"></a><em id="i1199720484127"><a name="i1199720484127"></a><a name="i1199720484127"></a>参数名称</em></p>
        </th>
        <th class="cellrowborder" valign="top" width="70.53%" id="mcps1.2.3.1.2"><p id="p199976489122"><a name="p199976489122"></a><a name="p199976489122"></a><em id="i9997154816124"><a name="i9997154816124"></a><a name="i9997154816124"></a>说明</em></p>
        </th>
        </tr>
        </thead>
        <tbody><tr id="row11997124871210"><td class="cellrowborder" valign="top" width="29.470000000000002%" headers="mcps1.2.3.1.1 "><p id="p1299734820121"><a name="p1299734820121"></a><a name="p1299734820121"></a><em id="i199764819121"><a name="i199764819121"></a><a name="i199764819121"></a>AI引擎</em></p>
        </td>
        <td class="cellrowborder" valign="top" width="70.53%" headers="mcps1.2.3.1.2 "><p id="p1899720481122"><a name="p1899720481122"></a><a name="p1899720481122"></a><em id="i9997848191217"><a name="i9997848191217"></a><a name="i9997848191217"></a>Ascend-Powered-Engine，mindspore_1.3.0-cann_5.0.2</em></p>
        </td>
        </tr>
        <tr id="row5997348121218"><td class="cellrowborder" valign="top" width="29.470000000000002%" headers="mcps1.2.3.1.1 "><p id="p139971748141218"><a name="p139971748141218"></a><a name="p139971748141218"></a><em id="i1199784811220"><a name="i1199784811220"></a><a name="i1199784811220"></a>代码目录</em></p>
        </td>
        <td class="cellrowborder" valign="top" width="70.53%" headers="mcps1.2.3.1.2 "><p id="p2099724810127"><a name="p2099724810127"></a><a name="p2099724810127"></a><em id="i17997144871212"><a name="i17997144871212"></a><a name="i17997144871212"></a>算法代码存储的OBS路径。上传训练脚本，如：/mcnn/MCNN/</em></p>
        </td>
        </tr>
        <tr id="row899794811124"><td class="cellrowborder" valign="top" width="29.470000000000002%" headers="mcps1.2.3.1.1 "><p id="p799714482129"><a name="p799714482129"></a><a name="p799714482129"></a><em id="i399704871210"><a name="i399704871210"></a><a name="i399704871210"></a>启动文件</em></p>
        </td>
        <td class="cellrowborder" valign="top" width="70.53%" headers="mcps1.2.3.1.2 "><p id="p13997154831215"><a name="p13997154831215"></a><a name="p13997154831215"></a><em id="i11997648161214"><a name="i11997648161214"></a><a name="i11997648161214"></a>启动文件：启动训练的python脚本，如：/mcnn/MCNN/start_train.py</em></p>
        <div class="notice" id="note1799734891214"><a name="note1799734891214"></a><a name="note1799734891214"></a><span class="noticetitle"> 须知： </span><div class="noticebody"><p id="p7998194814127"><a name="p7998194814127"></a><a name="p7998194814127"></a><em id="i199987481127"><a name="i199987481127"></a><a name="i199987481127"></a>需要把modelArts/目录下的start.py启动脚本拷贝到根目录下。</em></p>
        </div></div>
        </td>
        </tr>
        <tr id="row59981448101210"><td class="cellrowborder" valign="top" width="29.470000000000002%" headers="mcps1.2.3.1.1 "><p id="p19998124812123"><a name="p19998124812123"></a><a name="p19998124812123"></a><em id="i1399864831211"><a name="i1399864831211"></a><a name="i1399864831211"></a>输入数据配置</em></p>
        </td>
        <td class="cellrowborder" valign="top" width="70.53%" headers="mcps1.2.3.1.2 "><p id="p139982484129"><a name="p139982484129"></a><a name="p139982484129"></a><em id="i299816484122"><a name="i299816484122"></a><a name="i299816484122"></a>代码路径参数：data_url</em></p>
        </td>
        </tr>
        <tr id="row179981948151214"><td class="cellrowborder" valign="top" width="29.470000000000002%" headers="mcps1.2.3.1.1 "><p id="p89981948191220"><a name="p89981948191220"></a><a name="p89981948191220"></a><em id="i599844831217"><a name="i599844831217"></a><a name="i599844831217"></a>输出数据配置</em></p>
        </td>
        <td class="cellrowborder" valign="top" width="70.53%" headers="mcps1.2.3.1.2 "><p id="p599814485120"><a name="p599814485120"></a><a name="p599814485120"></a><em id="i189981748171218"><a name="i189981748171218"></a><a name="i189981748171218"></a>代码路径参数：train_url</em></p>
        </td>
        </tr>
        </tbody>
        </table>

    3. 填写超参数。

        单击“添加超参”，手动添加超参。配置代码中的命令行参数值，请根据您编写的算法代码逻辑进行填写，确保参数名称和代码的参数名称保持一致，可填写多个参数。

        **表 2** _超参说明_

        <a name="table29981482127"></a>
        <table><thead align="left"><tr id="row1599894881216"><th class="cellrowborder" valign="top" width="25%" id="mcps1.2.6.1.1"><p id="p89988484121"><a name="p89988484121"></a><a name="p89988484121"></a><em id="i89985485123"><a name="i89985485123"></a><a name="i89985485123"></a>参数名称</em></p>
        </th>
        <th class="cellrowborder" valign="top" width="15%" id="mcps1.2.6.1.2"><p id="p1999114814121"><a name="p1999114814121"></a><a name="p1999114814121"></a><em id="i7999448181212"><a name="i7999448181212"></a><a name="i7999448181212"></a>类型</em></p>
        </th>
        <th class="cellrowborder" valign="top" width="17%" id="mcps1.2.6.1.3"><p id="p6999124810126"><a name="p6999124810126"></a><a name="p6999124810126"></a><em id="i17999144818126"><a name="i17999144818126"></a><a name="i17999144818126"></a>默认值</em></p>
        </th>
        <th class="cellrowborder" valign="top" width="18%" id="mcps1.2.6.1.4"><p id="p69992486123"><a name="p69992486123"></a><a name="p69992486123"></a><em id="i1599916488127"><a name="i1599916488127"></a><a name="i1599916488127"></a>是否必填</em></p>
        </th>
        <th class="cellrowborder" valign="top" width="25%" id="mcps1.2.6.1.5"><p id="p1999248121214"><a name="p1999248121214"></a><a name="p1999248121214"></a><em id="i299915481121"><a name="i299915481121"></a><a name="i299915481121"></a>描述</em></p>
        </th>
        </tr>
        </thead>
        <tbody><tr id="row9999134818128"><td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.6.1.1 "><p id="p14999124811212"><a name="p14999124811212"></a><a name="p14999124811212"></a><em id="i39991748101218"><a name="i39991748101218"></a><a name="i39991748101218"></a>batch_size</em></p>
        </td>
        <td class="cellrowborder" valign="top" width="15%" headers="mcps1.2.6.1.2 "><p id="p599924815129"><a name="p599924815129"></a><a name="p599924815129"></a><em id="i8999184811212"><a name="i8999184811212"></a><a name="i8999184811212"></a>int</em></p>
        </td>
        <td class="cellrowborder" valign="top" width="17%" headers="mcps1.2.6.1.3 "><p id="p179992484129"><a name="p179992484129"></a><a name="p179992484129"></a><em id="i1799913488128"><a name="i1799913488128"></a><a name="i1799913488128"></a>1</em></p>
        </td>
        <td class="cellrowborder" valign="top" width="18%" headers="mcps1.2.6.1.4 "><p id="p179991348181213"><a name="p179991348181213"></a><a name="p179991348181213"></a><em id="i20999134812126"><a name="i20999134812126"></a><a name="i20999134812126"></a>否</em></p>
        </td>
        <td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.6.1.5 "><p id="p899916487125"><a name="p899916487125"></a><a name="p899916487125"></a><em id="i99999482127"><a name="i99999482127"></a><a name="i99999482127"></a>训练集的batch_size。</em></p>
        </td>
        </tr>
        <tr id="row14999148161210"><td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.6.1.1 "><p id="p199915488129"><a name="p199915488129"></a><a name="p199915488129"></a><em id="i11999448141216"><a name="i11999448141216"></a><a name="i11999448141216"></a>lr</em></p>
        </td>
        <td class="cellrowborder" valign="top" width="15%" headers="mcps1.2.6.1.2 "><p id="p7999124813124"><a name="p7999124813124"></a><a name="p7999124813124"></a><em id="i7999748151214"><a name="i7999748151214"></a><a name="i7999748151214"></a>float</em></p>
        </td>
        <td class="cellrowborder" valign="top" width="17%" headers="mcps1.2.6.1.3 "><p id="p902049121213"><a name="p902049121213"></a><a name="p902049121213"></a><em id="i100124914123"><a name="i100124914123"></a><a name="i100124914123"></a>0.000028</em></p>
        </td>
        <td class="cellrowborder" valign="top" width="18%" headers="mcps1.2.6.1.4 "><p id="p19004917125"><a name="p19004917125"></a><a name="p19004917125"></a><em id="i208494126"><a name="i208494126"></a><a name="i208494126"></a>否</em></p>
        </td>
        <td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.6.1.5 "><p id="p10134915129"><a name="p10134915129"></a><a name="p10134915129"></a><em id="i101949121214"><a name="i101949121214"></a><a name="i101949121214"></a>训练时的学习率.当使用8卡训练时，学习率为0.000028。</em></p>
        </td>
        </tr>
        <tr id="row100124911121"><td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.6.1.1 "><p id="p150849131211"><a name="p150849131211"></a><a name="p150849131211"></a><em id="i1101549151218"><a name="i1101549151218"></a><a name="i1101549151218"></a>momentum</em></p>
        </td>
        <td class="cellrowborder" valign="top" width="15%" headers="mcps1.2.6.1.2 "><p id="p19054914124"><a name="p19054914124"></a><a name="p19054914124"></a><em id="i10144919126"><a name="i10144919126"></a><a name="i10144919126"></a>float</em></p>
        </td>
        <td class="cellrowborder" valign="top" width="17%" headers="mcps1.2.6.1.3 "><p id="p6011490123"><a name="p6011490123"></a><a name="p6011490123"></a><em id="i00144917122"><a name="i00144917122"></a><a name="i00144917122"></a>0</em></p>
        </td>
        <td class="cellrowborder" valign="top" width="18%" headers="mcps1.2.6.1.4 "><p id="p301449191215"><a name="p301449191215"></a><a name="p301449191215"></a><em id="i180104910126"><a name="i180104910126"></a><a name="i180104910126"></a>否</em></p>
        </td>
        <td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.6.1.5 "><p id="p1702495127"><a name="p1702495127"></a><a name="p1702495127"></a><em id="i170249181214"><a name="i170249181214"></a><a name="i170249181214"></a>训练时的动量。</em></p>
        </td>
        </tr>
        <tr id="row001549161215"><td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.6.1.1 "><p id="p1608498123"><a name="p1608498123"></a><a name="p1608498123"></a><em id="i501049191215"><a name="i501049191215"></a><a name="i501049191215"></a>epoch_size</em></p>
        </td>
        <td class="cellrowborder" valign="top" width="15%" headers="mcps1.2.6.1.2 "><p id="p1064915124"><a name="p1064915124"></a><a name="p1064915124"></a><em id="i20104911127"><a name="i20104911127"></a><a name="i20104911127"></a>int</em></p>
        </td>
        <td class="cellrowborder" valign="top" width="17%" headers="mcps1.2.6.1.3 "><p id="p80164951212"><a name="p80164951212"></a><a name="p80164951212"></a><em id="i190184921219"><a name="i190184921219"></a><a name="i190184921219"></a>800</em></p>
        </td>
        <td class="cellrowborder" valign="top" width="18%" headers="mcps1.2.6.1.4 "><p id="p1811749171212"><a name="p1811749171212"></a><a name="p1811749171212"></a><em id="i161114981216"><a name="i161114981216"></a><a name="i161114981216"></a>否</em></p>
        </td>
        <td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.6.1.5 "><p id="p8114941212"><a name="p8114941212"></a><a name="p8114941212"></a><em id="i9174981214"><a name="i9174981214"></a><a name="i9174981214"></a>总训练轮数</em></p>
        </td>
        </tr>
        </tbody>
        </table>

### 创建训练作业

1. 登录ModelArts。
2. 创建训练作业。

1. 使用华为云帐号登录[ModelArts管理控制台](https://console.huaweicloud.com/modelarts)，在左侧导航栏中选择“训练管理 \> 训练作业（New）”，默认进入“训练作业”列表。
2. 单击“创建训练作业”，进入“创建训练作业”页面，在该页面填写训练作业相关参数。

    1. 填写基本信息。

        基本信息包含“名称”和“描述”。

    2. 填写作业参数。

        包含数据来源、算法来源等关键信息。本步骤只提供训练任务部分参数配置说明，其他参数配置详情请参见[《ModelArts AI 工程师用户指南](https://support.huaweicloud.com/modelarts/index.html)》中“训练管理（new）”。

        **MindSpore和TensorFlow创建训练作业步骤**

        **表 1**  参数说明

        <a name="table96111035134613"></a>
        <table><thead align="left"><tr id="zh-cn_topic_0000001178072725_row1727593212228"><th class="cellrowborder" valign="top" width="20%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001178072725_p102751332172212"><a name="zh-cn_topic_0000001178072725_p102751332172212"></a><a name="zh-cn_topic_0000001178072725_p102751332172212"></a>参数名称</p>
        </th>
        <th class="cellrowborder" valign="top" width="20%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001178072725_p186943411156"><a name="zh-cn_topic_0000001178072725_p186943411156"></a><a name="zh-cn_topic_0000001178072725_p186943411156"></a>子参数</p>
        </th>
        <th class="cellrowborder" valign="top" width="60%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001178072725_p1827543282216"><a name="zh-cn_topic_0000001178072725_p1827543282216"></a><a name="zh-cn_topic_0000001178072725_p1827543282216"></a>说明</p>
        </th>
        </tr>
        </thead>
        <tbody><tr id="zh-cn_topic_0000001178072725_row780219161358"><td class="cellrowborder" valign="top" width="20%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001178072725_p0803121617510"><a name="zh-cn_topic_0000001178072725_p0803121617510"></a><a name="zh-cn_topic_0000001178072725_p0803121617510"></a>算法</p>
        </td>
        <td class="cellrowborder" valign="top" width="20%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001178072725_p186947411520"><a name="zh-cn_topic_0000001178072725_p186947411520"></a><a name="zh-cn_topic_0000001178072725_p186947411520"></a>我的算法</p>
        </td>
        <td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001178072725_p20803141614514"><a name="zh-cn_topic_0000001178072725_p20803141614514"></a><a name="zh-cn_topic_0000001178072725_p20803141614514"></a>选择“我的算法”页签，勾选上文中创建的算法。</p>
        <p id="zh-cn_topic_0000001178072725_p24290418284"><a name="zh-cn_topic_0000001178072725_p24290418284"></a><a name="zh-cn_topic_0000001178072725_p24290418284"></a>如果没有创建算法，请单击“创建”进入创建算法页面，详细操作指导参见“创建算法”。</p>
        </td>
        </tr>
        <tr id="zh-cn_topic_0000001178072725_row1927503211228"><td class="cellrowborder" valign="top" width="20%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001178072725_p327583216224"><a name="zh-cn_topic_0000001178072725_p327583216224"></a><a name="zh-cn_topic_0000001178072725_p327583216224"></a>训练输入</p>
        </td>
        <td class="cellrowborder" valign="top" width="20%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001178072725_p1069419416510"><a name="zh-cn_topic_0000001178072725_p1069419416510"></a><a name="zh-cn_topic_0000001178072725_p1069419416510"></a>数据来源</p>
        </td>
        <td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001178072725_p142750323227"><a name="zh-cn_topic_0000001178072725_p142750323227"></a><a name="zh-cn_topic_0000001178072725_p142750323227"></a>选择OBS上数据集存放的目录。</p>
        </td>
        </tr>
        <tr id="zh-cn_topic_0000001178072725_row127593211227"><td class="cellrowborder" valign="top" width="20%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001178072725_p9744151562"><a name="zh-cn_topic_0000001178072725_p9744151562"></a><a name="zh-cn_topic_0000001178072725_p9744151562"></a>训练输出</p>
        </td>
        <td class="cellrowborder" valign="top" width="20%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001178072725_p1027563212210"><a name="zh-cn_topic_0000001178072725_p1027563212210"></a><a name="zh-cn_topic_0000001178072725_p1027563212210"></a>模型输出</p>
        </td>
        <td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001178072725_p13275113252214"><a name="zh-cn_topic_0000001178072725_p13275113252214"></a><a name="zh-cn_topic_0000001178072725_p13275113252214"></a>选择训练结果的存储位置（OBS路径），请尽量选择空目录来作为训练输出路径。</p>
        </td>
        </tr>
        <tr id="zh-cn_topic_0000001178072725_row18750142834916"><td class="cellrowborder" valign="top" width="20%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001178072725_p5751172811492"><a name="zh-cn_topic_0000001178072725_p5751172811492"></a><a name="zh-cn_topic_0000001178072725_p5751172811492"></a>规格</p>
        </td>
        <td class="cellrowborder" valign="top" width="20%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001178072725_p107514288495"><a name="zh-cn_topic_0000001178072725_p107514288495"></a><a name="zh-cn_topic_0000001178072725_p107514288495"></a>-</p>
        </td>
        <td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001178072725_p3751142811495"><a name="zh-cn_topic_0000001178072725_p3751142811495"></a><a name="zh-cn_topic_0000001178072725_p3751142811495"></a>Ascend: 1*Ascend 910(32GB) | ARM: 24 核 96GB</p>
        </td>
        </tr>
        <tr id="zh-cn_topic_0000001178072725_row16275103282219"><td class="cellrowborder" valign="top" width="20%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001178072725_p15275132192213"><a name="zh-cn_topic_0000001178072725_p15275132192213"></a><a name="zh-cn_topic_0000001178072725_p15275132192213"></a>作业日志路径</p>
        </td>
        <td class="cellrowborder" valign="top" width="20%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001178072725_p1369484117516"><a name="zh-cn_topic_0000001178072725_p1369484117516"></a><a name="zh-cn_topic_0000001178072725_p1369484117516"></a>-</p>
        </td>
        <td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001178072725_p227563218228"><a name="zh-cn_topic_0000001178072725_p227563218228"></a><a name="zh-cn_topic_0000001178072725_p227563218228"></a>设置训练日志存放的目录。请注意选择的OBS目录有读写权限。</p>
        </td>
        </tr>
        </tbody>
        </table>

3. 单击“提交”，完成训练作业的创建。

    训练作业一般需要运行一段时间，根据您选择的数据量和资源不同，训练时间将耗时几分钟到几十分钟不等。

### 查看训练任务日志

1. 在ModelArts管理控制台，在左侧导航栏中选择“训练管理 \> 训练作业（New）”，默认进入“训练作业”列表。_
2. 在训练作业列表中，您可以单击作业名称，查看该作业的详情。

    详情中包含作业的基本信息、训练参数、日志详情和资源占用情况。

    在OBS桶的train_output文件夹下可以看到ckpt模型和air模型的生成。
