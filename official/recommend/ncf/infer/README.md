## 推理

* **[准备推理数据](#准备推理数据.md)**  

* **[模型转换](#模型转换.md)**  

* **[mxBase推理](#mxBase推理.md)**  

* **[MindX SDK推理](#MindX-SDK推理.md)**

### 准备推理数据

准备模型转换和模型推理所需目录及数据。

1. 下载源码包。

    单击“下载模型脚本”和“下载模型”，下载所需软件包。

2. 将源码上传至推理服务器任意目录并解压（如：“/home/HwHiAiUser“）。

3. 编译镜像 **（需要安装软件依赖时选择）** 。

    **docker build -t** _infer\_image_ **--build-arg FROM\_IMAGE\_NAME=**_base\_image:tag_ **.**

    **表 2**  _镜像编译参数说明_

    <a name="table82851171646"></a>
    <table><thead align="left"><tr id="zh-cn_topic_0304403934_row9243114772414"><th class="cellrowborder" valign="top" width="40%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0304403934_p524364716241"><a name="zh-cn_topic_0304403934_p524364716241"></a><a name="zh-cn_topic_0304403934_p524364716241"></a>参数</p>
    </th>
    <th class="cellrowborder" valign="top" width="60%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0304403934_p172431247182412"><a name="zh-cn_topic_0304403934_p172431247182412"></a><a name="zh-cn_topic_0304403934_p172431247182412"></a>说明</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="zh-cn_topic_0304403934_row52431473244"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p144312172333"><a name="p144312172333"></a><a name="p144312172333"></a><em id="i290520133315"><a name="i290520133315"></a><a name="i290520133315"></a>infer_image</em></p>
    </td>
    <td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0304403934_p10243144712410"><a name="zh-cn_topic_0304403934_p10243144712410"></a><a name="zh-cn_topic_0304403934_p10243144712410"></a>推理镜像名称，根据实际写入。</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0304403934_row1624394732415"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0304403934_p92434478242"><a name="zh-cn_topic_0304403934_p92434478242"></a><a name="zh-cn_topic_0304403934_p92434478242"></a><em id="i78645182347"><a name="i78645182347"></a><a name="i78645182347"></a>base_image</em></p>
    </td>
    <td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0304403934_p324313472240"><a name="zh-cn_topic_0304403934_p324313472240"></a><a name="zh-cn_topic_0304403934_p324313472240"></a>基础镜像，可从Ascend Hub上下载。</p>
    </td>
    </tr>
    <tr id="row2523459163416"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p55241359203412"><a name="p55241359203412"></a><a name="p55241359203412"></a><em id="i194517711355"><a name="i194517711355"></a><a name="i194517711355"></a>tag</em></p>
    </td>
    <td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1952435919341"><a name="p1952435919341"></a><a name="p1952435919341"></a>镜像tag，请根据实际配置，如：21.0.2。</p>
    </td>
    </tr>
    </tbody>
    </table>

    > ![输入图片说明](https://images.gitee.com/uploads/images/2021/0926/181247_df155dc2_8725359.gif "icon-note.gif")**说明：**
    > 不要遗漏命令结尾的“.“。

4. 准备数据。

    由于后续推理均在容器中进行，因此需要把用于推理的数据集、模型文件、代码等均放在同一数据路径中，后续示例将以“/home/HwHiAiUser/ncf“为例。

    ```shell
    /home/HwHiAiUser/ncf
    ├──infer                         # MindX高性能预训练模型新增
    │   ├── README.md                # 离线推理文档
    │   ├── convert                  # 转换om模型命令
    │   │   └── convert.sh
    │   ├── data                     # 包括模型文件、模型eval数据集、模型相关配置文件
    │   │   ├── config
    │   │   │   └── NCF.pipeline
    │   │   ├── datasets
    │   │   ├── input
    │   │   └── model
    │   ├── mxbase                   # 基于mxbase推理
    │   │   ├── CMakeLists.txt
    │   │   ├── build.sh
    │   │   └── src
    │   │       ├── MxNCFBase.cpp
    │   │       ├── MxNCFBase.h
    │   │       └── main.cpp
    │   ├── sdk                      # 基于sdk.run包推理
    │   │   ├── ncf_tensorinfer.py
    │   │   └── run.sh
    │   ├──util                      # 工具脚本
    │   │   ├── constants.py
    │   │   ├── create_data.py
    │   │   └── process_data.py
    │   └── docker_start_infer.sh    # 启动容器脚本
    ```

    * 在infer/data目录下创建datasets, input, model目录

        ```shell
        cd /home/HwHiAiUser/ncf/infer/data
        mkdir datasets
        mkdir input
        mkdir model
        ```

    * 下载[MovieLens datasets](http://files.grouplens.org/datasets/movielens/)中的ml-1m.zip以及ml-20m.zip
    * 将下载的zip数据集文件放入数据集目录/home/HwHiAiUser/ncf/infer/data/datasets
    * 执行infer/util目录下的create_data.py

        #### ml-1m

        ```shell
        python3.7 create_data.py --data_path="/home/HwHiAiUser/ncf/infer/data/datasets"
        ```

        #### ml-20m

        ```shell
        python3.7 create_data.py --data_path="/home/HwHiAiUser/ncf/infer/data/datasets" --dataset="ml-20m"
        ```

        请耐心等待，特别是ml-20m处理时间较长，执行完成后在infer/data/datasets目录下生成ml-1m或者ml-20m数据文件夹

    * 如果input目录下已存在tensor文件夹则删除

        ```shell
        cd /home/HwHiAiUser/ncf/infer/data/input
        rm -rf tensor_*
        ```

    * 执行infer/util目录下的process_data.py

        #### ml-1m

        ```shell
        python3.7 process_data.py --data_path="/home/HwHiAiUser/ncf/infer/data/datasets"
        ```

        #### ml-20m

        ```shell
        python3.7 process_data.py --data_path="/home/HwHiAiUser/ncf/infer/data/datasets" --dataset="ml-20m"
        ```

        请耐心等待，特别是ml-20m处理时间较长，执行成功后会在infer/data/input目录下生成tensor_0, tensor_1, tensor_2三个数据文件夹供推理使用

5. 启动容器。

    进入“infer“目录，执行以下命令，启动容器。

    **bash docker\_start\_infer.sh** _docker\_image:tag_ _model\_dir_

    **表 3**  _容器启动参数说明_

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

    ```shell
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

### 模型转换

1. 准备模型文件。

    * 将训练生成的ncf_1m.air模型和ncf_20m.air模型放入infer/data/model目录下

2. 模型转换。

    进入“infer/convert“目录进行模型转换，转换详细信息可查看转换脚本，在**convert.sh**脚本文件中，配置相关参数。

    ```shell
    air_path=$1
    om_name=$2
    atc \
      --framework=1 \              # 1代表MindSpore。
      --model="${air_path}" \      # 待转换的air模型，模型可以通过训练生成或通过“下载模型”获得。
      --output="${om_name}" \      # 转换后输出的om模型。
      --soc_version=Ascend310 \    # 模型转换时指定芯片版本。
    ```

    转换命令如下。

    **bash convert.sh** _air\_path_ _om\_name_

    **表 4**  _模型转换参数说明_

    <a name="table15982121511203"></a>
    <table><thead align="left"><tr id="row1598241522017"><th class="cellrowborder" valign="top" width="40%" id="mcps1.2.3.1.1"><p id="p189821115192014"><a name="p189821115192014"></a><a name="p189821115192014"></a>参数</p>
    </th>
    <th class="cellrowborder" valign="top" width="60%" id="mcps1.2.3.1.2"><p id="p1982161512206"><a name="p1982161512206"></a><a name="p1982161512206"></a>说明</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row0982101592015"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1598231542020"><a name="p1598231542020"></a><a name="p1598231542020"></a>air_path</p>
    </td>
    <td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p598231511200"><a name="p598231511200"></a><a name="p598231511200"></a>AIR文件路径。</p>
    </td>
    </tr>
    <tr id="row109831315132011"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p598319158204"><a name="p598319158204"></a><a name="p598319158204"></a>om_name</p>
    </td>
    <td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1898316155207"><a name="p1898316155207"></a><a name="p1898316155207"></a>生成的OM文件名，转换脚本会在此基础上添加.om后缀。</p>
    </td>
    </tr>
    </tbody>
    </table>

    * 执行infer/convert/convert.sh

        #### ml-1m

        ```shell
        cd /home/HwHiAiUser/ncf/infer/convert
        bash convert.sh ../data/model/ncf_1m.air ../data/model/ncf_1m
        ```

        #### ml-20m

        ```shell
        cd /home/HwHiAiUser/ncf/infer/convert
        bash convert.sh ../data/model/ncf_20m.air ../data/model/ncf_20m
        ```

        执行完成后会在infer/data/model目录下生成ncf_1m.om和ncf_20m.om模型文件。

### mxBase推理

1. 编译工程。

    ```shell
    cd /home/HwHiAiUser/ncf/infer/mxbase
    bash build.sh
    ```

    在当前目录生成可执行文件ncf

2. 运行推理服务。

    #### ml-1m

    ```shell
    ./ncf ../data/model/ncf_1m.om ../data/input/ 1
    ```

    #### ml-20m

    ```shell
    ./ncf ../data/model/ncf_20m.om ../data/input/ 1
    ```

    注意infer/data/input下的数据产生于哪个数据集则执行对应指令

3. 查看推理结果。

    推理脚本会在命令行显示如下结果：

    ```shell
    I0916 14:39:12.324308 139530 main.cpp:104] ==============================================================
    I0916 14:39:12.324316 139530 main.cpp:107] average HR = 0.689238, average NDCG = 0.412503
    I0916 14:39:12.324322 139530 main.cpp:108] ==============================================================
    ```

### MindX SDK推理

1. 修改配置文件。

    ```shell
    cd /home/HwHiAiUser/ncf/infer/data/config
    vim NCF.pipeline
    ```

    注意infer/data/input下的数据产生于哪个数据集则修改”modelPath”字段(默认为“../data/model/ncf_1m.om”)与之对应。即ml-1m保持默认，ml-20m修改为“../data/model/ncf_20m.om”

2. 运行推理服务。

    ```shell
    cd /home/HwHiAiUser/ncf/infer/sdk
    bash run.sh
    ```

3. 查看推理结果。

    推理脚本会在命令行终端显示如下结果：

    ```
    ====================Average Eval Begin!====================
    average HR = 0.689238, average NDCG = 0.412503
    ====================Average Eval Finish!====================
    Infer 4 batches, cost total time: 0.438486 sec.
    Average cost 0.109621 sec per batch
    ```

## 在ModelArts上应用

* **[创建OBS桶](#创建OBS桶.md)**  

* **[创建算法（适用于MindSpore和TensorFlow）](#创建算法（适用于MindSpore和TensorFlow）.md)**  

* **[创建训练作业](#创建训练作业.md)**  

* **[查看训练任务日志](#查看训练任务日志.md)**

### 创建OBS桶

1. 创建桶。

    登录[OBS管理控制台](https://storage.huaweicloud.com/obs)，创建OBS桶。具体请参见[创建桶](https://support.huaweicloud.com/usermanual-obs/obs_03_0306.html)章节。例如，创建名称为“ncf”的OBS桶

    * 点击”创建桶“
    * ”区域“选择”华北-北京四“
    * ”存储类别“选取”标准存储“
    * ”桶ACL“选取”私有“
    * 关闭”多AZ“
    * 输入全局唯一桶名称
    * 点击”确定“

2. 创建文件夹存放数据。

    创建用于存放数据的文件夹，具体请参见[新建文件夹](https://support.huaweicloud.com/usermanual-obs/obs_03_0316.html)章节。例如，在已创建的OBS桶“ncf”中创建如下模型目录。

    ![输入图片说明](https://gitee.com/evocrow/models/raw/ncf/official/recommend/ncf/modelarts/res/obs_directory.png "obs-directory.png")

    * code：存放训练脚本目录
    * dataset：存放训练数据集目录
    * logs：存放训练日志目录
    * output：训练生成ckpt和pb模型目录

3. 上传代码

    * 进入ncf代码文件根目录
    * 将ncf目录下的文件全部上传至code文件夹下(infer目录可以不用上传)

4. 上传数据集

    * 创建数据集目录，例如/home/HwHiAiUser/ncf/infer/data/datasets
    * 下载[MovieLens datasets](http://files.grouplens.org/datasets/movielens/)中的ml-1m.zip以及ml-20m.zip
    * 将下载的zip数据集文件放入数据集目录/home/HwHiAiUser/ncf/infer/data/datasets
    * 执行infer/util目录下的create_data.py

        #### ml-1m

        ```shell
        python3.7 create_data.py --data_path="/home/HwHiAiUser/ncf/infer/data/datasets"
        ```

        #### ml-20m

        ```shell
        python3.7 create_data.py --data_path="/home/HwHiAiUser/ncf/infer/data/datasets" --dataset="ml-20m"
        ```

        请耐心等待，特别是ml-20m处理时间较长，执行完成后在infer/data/datasets目录下生成ml-1m或者ml-20m数据文件夹

    * 将数据集目录下生成的ml-1m或者ml-20m目录上传至obs桶里的dataset目录下

### 创建算法（适用于MindSpore和TensorFlow）

1. 使用华为云帐号登录[ModelArts管理控制台](https://console.huaweicloud.com/modelarts)，在左侧导航栏中选择“算法管理”。

2. 在“我的算法管理”界面，单击左上角“创建”，进入“创建算法”页面。

3. 在“创建算法”页面，填写相关参数，然后单击“提交”。

    1. 设置算法基本信息。

    2. 设置“创建方式”为“自定义脚本”。

        用户需根据实际算法代码情况设置“AI引擎”、“代码目录”和“启动文件”。选择的AI引擎和编写算法代码时选择的框架必须一致。例如编写算法代码使用的是MindSpore，则在创建算法时也要选择MindSpore。

        ![输入图片说明](https://gitee.com/evocrow/models/raw/ncf/official/recommend/ncf/modelarts/res/create_algorithm.png "Creating-an-algorithm.png")

        **表 5** _创建算法参数说明_

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
        <td class="cellrowborder" valign="top" width="70.53%" headers="mcps1.2.3.1.2 "><p id="p2099724810127"><a name="p2099724810127"></a><a name="p2099724810127"></a><em id="i17997144871212"><a name="i17997144871212"></a><a name="i17997144871212"></a>算法代码存储的OBS路径。上传训练脚本，如：/obs桶/ncf/code</em></p>
        </td>
        </tr>
        <tr id="row899794811124"><td class="cellrowborder" valign="top" width="29.470000000000002%" headers="mcps1.2.3.1.1 "><p id="p799714482129"><a name="p799714482129"></a><a name="p799714482129"></a><em id="i399704871210"><a name="i399704871210"></a><a name="i399704871210"></a>启动文件</em></p>
        </td>
        <td class="cellrowborder" valign="top" width="70.53%" headers="mcps1.2.3.1.2 "><p id="p13997154831215"><a name="p13997154831215"></a><a name="p13997154831215"></a><em id="i11997648161214"><a name="i11997648161214"></a><a name="i11997648161214"></a>启动文件：启动训练的python脚本，如：/obs桶/ncf/code/start.py</em></p>
        <div class="notice" id="note1799734891214"><a name="note1799734891214"></a><a name="note1799734891214"></a><span class="noticetitle"> 须知： </span><div class="noticebody"><p id="p7998194814127"><a name="p7998194814127"></a><a name="p7998194814127"></a><em id="i199987481127"><a name="i199987481127"></a><a name="i199987481127"></a>需要把modelarts/目录下的start.py启动脚本拷贝到根目录下。</em></p>
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

        **表 6** _超参说明_

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
        <tbody><tr id="row9999134818128"><td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.6.1.1 "><p id="p14999124811212"><a name="p14999124811212"></a><a name="p14999124811212"></a><em id="i39991748101218"><a name="i39991748101218"></a><a name="i39991748101218"></a>dataset</em></p>
        </td>
        <td class="cellrowborder" valign="top" width="15%" headers="mcps1.2.6.1.2 "><p id="p599924815129"><a name="p599924815129"></a><a name="p599924815129"></a><em id="i8999184811212"><a name="i8999184811212"></a><a name="i8999184811212"></a>string</em></p>
        </td>
        <td class="cellrowborder" valign="top" width="17%" headers="mcps1.2.6.1.3 "><p id="p179992484129"><a name="p179992484129"></a><a name="p179992484129"></a><em id="i1799913488128"><a name="i1799913488128"></a><a name="i1799913488128"></a>ml-1m</em></p>
        </td>
        <td class="cellrowborder" valign="top" width="18%" headers="mcps1.2.6.1.4 "><p id="p179991348181213"><a name="p179991348181213"></a><a name="p179991348181213"></a><em id="i20999134812126"><a name="i20999134812126"></a><a name="i20999134812126"></a>是</em></p>
        </td>
        <td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.6.1.5 "><p id="p899916487125"><a name="p899916487125"></a><a name="p899916487125"></a><em id="i99999482127"><a name="i99999482127"></a><a name="i99999482127"></a>训练所用数据集</em></p>
        </td>
        </tr>
        <tr id="row001549161215"><td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.6.1.1 "><p id="p1608498123"><a name="p1608498123"></a><a name="p1608498123"></a><em id="i501049191215"><a name="i501049191215"></a><a name="i501049191215"></a>device_target</em></p>
        </td>
        <td class="cellrowborder" valign="top" width="15%" headers="mcps1.2.6.1.2 "><p id="p1064915124"><a name="p1064915124"></a><a name="p1064915124"></a><em id="i20104911127"><a name="i20104911127"></a><a name="i20104911127"></a>string</em></p>
        </td>
        <td class="cellrowborder" valign="top" width="17%" headers="mcps1.2.6.1.3 "><p id="p80164951212"><a name="p80164951212"></a><a name="p80164951212"></a><em id="i190184921219"><a name="i190184921219"></a><a name="i190184921219"></a>Ascend</em></p>
        </td>
        <td class="cellrowborder" valign="top" width="18%" headers="mcps1.2.6.1.4 "><p id="p1811749171212"><a name="p1811749171212"></a><a name="p1811749171212"></a><em id="i161114981216"><a name="i161114981216"></a><a name="i161114981216"></a>是</em></p>
        </td>
        <td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.6.1.5 "><p id="p8114941212"><a name="p8114941212"></a><a name="p8114941212"></a><em id="i9174981214"><a name="i9174981214"></a><a name="i9174981214"></a>-</em></p>
        </td>
        </tr>
        <tr id="row721249101214"><td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.6.1.1 "><p id="p3274911218"><a name="p3274911218"></a><a name="p3274911218"></a><em id="i721949161219"><a name="i721949161219"></a><a name="i721949161219"></a>train_epochs</em></p>
        </td>
        <td class="cellrowborder" valign="top" width="15%" headers="mcps1.2.6.1.2 "><p id="p32144971218"><a name="p32144971218"></a><a name="p32144971218"></a><em id="i92184917128"><a name="i92184917128"></a><a name="i92184917128"></a>int</em></p>
        </td>
        <td class="cellrowborder" valign="top" width="17%" headers="mcps1.2.6.1.3 "><p id="p112749171218"><a name="p112749171218"></a><a name="p112749171218"></a><em id="i1821849171216"><a name="i1821849171216"></a><a name="i1821849171216"></a>14</em></p>
        </td>
        <td class="cellrowborder" valign="top" width="18%" headers="mcps1.2.6.1.4 "><p id="p1521349101219"><a name="p1521349101219"></a><a name="p1521349101219"></a><em id="i321149151212"><a name="i321149151212"></a><a name="i321149151212"></a>是</em></p>
        </td>
        <td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.6.1.5 "><p id="p142104910122"><a name="p142104910122"></a><a name="p142104910122"></a><em id="i1822493125"><a name="i1822493125"></a><a name="i1822493125"></a>训练epochs数</em></p>
        </td>
        </tr>
        <tr id="row92114911217"><td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.6.1.1 "><p id="p82849131211"><a name="p82849131211"></a><a name="p82849131211"></a><em id="i9354971217"><a name="i9354971217"></a><a name="i9354971217"></a>batch_size</em></p>
        </td>
        <td class="cellrowborder" valign="top" width="15%" headers="mcps1.2.6.1.2 "><p id="p03134912123"><a name="p03134912123"></a><a name="p03134912123"></a><em id="i332494124"><a name="i332494124"></a><a name="i332494124"></a>int</em></p>
        </td>
        <td class="cellrowborder" valign="top" width="17%" headers="mcps1.2.6.1.3 "><p id="p1131549151220"><a name="p1131549151220"></a><a name="p1131549151220"></a><em id="i9384951220"><a name="i9384951220"></a><a name="i9384951220"></a>256</em></p>
        </td>
        <td class="cellrowborder" valign="top" width="18%" headers="mcps1.2.6.1.4 "><p id="p13354911122"><a name="p13354911122"></a><a name="p13354911122"></a><em id="i5304912127"><a name="i5304912127"></a><a name="i5304912127"></a>否</em></p>
        </td>
        <td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.6.1.5 "><p id="p11374921213"><a name="p11374921213"></a><a name="p11374921213"></a><em id="i17304917122"><a name="i17304917122"></a><a name="i17304917122"></a>batch_size大小。</em></p>
        </td>
        </tr>
        </tbody>
        </table>

### 创建训练作业

1. 使用华为云帐号登录[ModelArts管理控制台](https://console.huaweicloud.com/modelarts)，在左侧导航栏中选择“训练管理 \> 训练作业（New）”，默认进入“训练作业”列表。

2. 单击“创建训练作业”，进入“创建训练作业”页面，在该页面填写训练作业相关参数。

    1. 填写基本信息。

        基本信息包含“名称”和“描述”。

    2. 填写作业参数。

        包含数据来源、算法来源等关键信息。本步骤只提供训练任务部分参数配置说明，其他参数配置详情请参见[《ModelArts AI 工程师用户指南](https://support.huaweicloud.com/modelarts/index.html)》中“训练模型（new）”。

        **MindSpore和TensorFlow创建训练作业步骤**

        ![输入图片说明](https://gitee.com/evocrow/models/raw/ncf/official/recommend/ncf/modelarts/res/config_train_work.png "Creating-a-training-job.png")

        **表 7**  _训练作业参数说明_

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
        <td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001178072725_p142750323227"><a name="zh-cn_topic_0000001178072725_p142750323227"></a><a name="zh-cn_topic_0000001178072725_p142750323227"></a>选择OBS上数据集存放的目录,如:/obs桶/ncf/dataset</p>
        </td>
        </tr>
        <tr id="zh-cn_topic_0000001178072725_row127593211227"><td class="cellrowborder" valign="top" width="20%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001178072725_p9744151562"><a name="zh-cn_topic_0000001178072725_p9744151562"></a><a name="zh-cn_topic_0000001178072725_p9744151562"></a>训练输出</p>
        </td>
        <td class="cellrowborder" valign="top" width="20%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001178072725_p1027563212210"><a name="zh-cn_topic_0000001178072725_p1027563212210"></a><a name="zh-cn_topic_0000001178072725_p1027563212210"></a>模型输出</p>
        </td>
        <td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001178072725_p13275113252214"><a name="zh-cn_topic_0000001178072725_p13275113252214"></a><a name="zh-cn_topic_0000001178072725_p13275113252214"></a>选择训练结果的存储位置（OBS路径），如:/obs桶/ncf/output, 请尽量选择空目录来作为训练输出路径。</p>
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
        <td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001178072725_p227563218228"><a name="zh-cn_topic_0000001178072725_p227563218228"></a><a name="zh-cn_topic_0000001178072725_p227563218228"></a>设置训练日志存放的目录, 如：/obs桶/ncf/logs, 请注意选择的OBS目录有读写权限。</p>
        </td>
        </tr>
        </tbody>
        </table>

        > ![输入图片说明](https://images.gitee.com/uploads/images/2021/0926/181247_df155dc2_8725359.gif "icon-note.gif")**说明：**
        > 超参dataset默认为ml-1m, 如需使用ml-20m数据集则更改为ml-20m

### 查看训练任务日志

1. 在ModelArts管理控制台，在左侧导航栏中选择“训练管理 \> 训练作业（New）”，默认进入“训练作业”列表。
2. 在训练作业列表中，您可以单击作业名称，查看该作业的详情。

    详情中包含作业的基本信息、训练参数、日志详情和资源占用情况。

    ![输入图片说明](https://gitee.com/evocrow/models/raw/ncf/official/recommend/ncf/modelarts/res/logs.png "Logs.png")
