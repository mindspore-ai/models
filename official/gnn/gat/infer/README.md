
# 推理

## 准备推理数据

1. 下载源码包。

   单击“下载模型脚本”和“下载模型”，并下载所需MindX SDK开发套件（mxManufacture）。

   脚本目录结构如下：

   ```shell
   infer
   ├──README.md              # 离线推理文档
   ├──convert
   │    └──run.sh            # om模型转换脚本
   │──data
   │    ├──model             # 模型文件, 将ModelArts在文件夹 ~/results/model/下生成的文件拷贝过来
   │    ├──config            # 模型相关配置文件（pipeline）
   │    └──input             # 模型输入数据集, 将ModelArts在文件夹 ~/results/data/下生成的文件拷贝过来
   │        ├──cora
   │        └──citeseer
   │───mxbase                # 基于mxbase推理脚本
   │    ├──src
   │    │   ├──GatNerBase.cpp
   │    │   ├──GatNerBase.h
   │    │   ├──half.h
   │    │   └──main.cpp
   │    ├──CMakeLists.txt
   │    ├──build.sh
   │    └──run.sh
   │──sdk                    # 基于sdk包推理脚本
   │    ├──main_infer.py
   │    └──run.sh
   └── docker_start_infer.sh # 启动容器脚本
   ```

2. 将源码上传至推理服务器任意目录并解压（如：“/home/data/cz“）。

   在 gat 文件下创建 results 文件夹。

3. 编译镜像。

   ```shell
   docker build -t infer_image --build-arg FROM_IMAGE_NAME=base_image:tag --build-arg SDK_PKG= sdk_pkg .
   ```

   **表 1**  参数说明

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
   <td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1952435919341"><a name="p1952435919341"></a><a name="p1952435919341"></a>镜像tag，请根据实际配置，如：21.0.1。</p>
   </td>
   </tr>
   <tr id="zh-cn_topic_0304403934_row132436473240"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0304403934_p1824319472242"><a name="zh-cn_topic_0304403934_p1824319472242"></a><a name="zh-cn_topic_0304403934_p1824319472242"></a>sdk_pkg</p>
   </td>
   <td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0304403934_p7243144712249"><a name="zh-cn_topic_0304403934_p7243144712249"></a><a name="zh-cn_topic_0304403934_p7243144712249"></a>下载的mxManufacture包名称，如Ascend-mindxsdk-mxmanufacture_<em id="i061383054119"><a name="i061383054119"></a><a name="i061383054119"></a>{version}</em>_linux-<em id="i1055956194514"><a name="i1055956194514"></a><a name="i1055956194514"></a>{arch}</em>.run。</p>
   </td>
   </tr>
   </tbody>
   </table>

>![输入图片说明](https://images.gitee.com/uploads/images/2021/0719/172222_3c2963f4_923381.gif "icon-note.gif") **说明：**
   >不要遗漏命令结尾的“.“。

4. 准备数据。

   执行脚本，导出准备用于推理的数据。

      ```shell
   # 导出 Cora 推理数据
   python preprocess_infer.py --dataset cora
   # 导出 Citeseer 推理数据
   python preprocess_infer.py --dataset citeseer
      ```

   将文件夹 results/data/ 下生成的推理数据拷贝到 infer/data/input 目录下。

5. 启动容器。

    进入“infer“目录，执行以下命令，启动容器。

      ```shell
   bash docker_start_infer.sh docker_image:tag model_dir
     ```

   > ![输入图片说明](https://images.gitee.com/uploads/images/2021/0926/181445_0077d606_8725359.gif) **说明：**
      > MindX SDK开发套件（mxManufacture）已安装在基础镜像中，安装路径：“/usr/local/sdk_home“。

   **表 2** 参数说明

   | 参数           | 说明                                  |
   | -------------- | ------------------------------------- |
   | *docker_image* | 推理镜像名称，根据实际写入。          |
   | *tag*          | 镜像tag，请根据实际配置，如：21.0.2。 |
   | model_dir      | 推理代码路径。                        |

   启动容器时会将推理芯片和数据路径挂载到容器中。可根据需要通过修改**docker_start_infer.sh**的device来指定挂载的推理芯片。

## 模型转换

   1. 准备模型文件。

      * 将ModelArts训练之后导出的 results/model/**.air 模型文件放入 infer/data/model 目录下

   2. 模型转换。

      * 执行 infer/convert/run.sh， 转换命令如下 。

      ```
      cd ./infer/convert
      # 对Cora模型进行转换
      sh run.sh cora
      # 对Citeseer模型进行转换
      sh run.sh citeseer
      ```

      执行完成后会在 infer/data/model 目录下生成 **.om 模型文件，注意此处 om 文件名需与 pipeline 中的保持一致。

## MxBase推理

   2. 编译工程。

      ```
      cd ./infer/mxbase
      sh build.sh
      ```

      在当前目录生成可执行文件。

   3. 运行推理服务。

      执行推理脚本，命令如下 。

      ```
      # 对 Cora 模型运行推理
      sh run.sh cora
      # 对Citeseer数据集运行推理
      sh run.sh citeseer
      ```

   4. 观察结果。

      推理生成的结果保存至result文件夹中，进入result文件查看推理结果，推理脚本会在命令行终端显示推理精度信息如下：

      ```
      # cora 数据集
      output shape is: 2708 7
      ==============================================================
      infer result of ../data/input/cora/ is:
      ==============================================================
      TP: 2264, FP: 444
      ==============================================================
      Precision: 0.836041
      ==============================================================
      # citeseer 数据集
      output shape is: 3312 6
      ==============================================================
      infer result of ../data/input/citeseer/ is:
      ==============================================================
      TP: 2363, FP: 949
      ==============================================================
      Precision: 0.713466
      ==============================================================
      ```

## MindX SDK推理

   1. 修改配置文件。

      修改pipeline文件。

      ```
      vim infer/data/config/gat_cora.pipeline
      vim infer/data/config/gat_citeseer.pipeline
      ```

      如需替换模型，修改”modelPath”字段对应的模型路径

   2. 运行推理服务。

      在当前目录新建output文件夹，执行推理脚本，命令如下 。

      ```
      cd infer/sdk
      bash run.sh [Dataset] # Dataset 在["cora", "citeseer"]中选择
      ```

      查看推理结果 。

      推理生成的结果保存至output文件夹中，进入output文件查看推理结果，推理脚本会在命令行终端显示推理精度信息如下：

      ```
      # cora 数据集
      feature.txt shape : [1, 3880564]
      Send successfully!
      adjacency.txt shape : [1, 7333264]
      Send successfully!
      ============================  Infer Result ============================
      Pred_label  label:[3 4 4 ... 1 3 3]
      Infer acc:0.832000
      =======================================================================

      # citeseer 数据集
      feature.txt shape : [1, 12264336]
      Send successfully!
      adjacency.txt shape : [1, 10969344]
      Send successfully!
      ============================  Infer Result ============================
      Pred_label  label:[3 1 5 ... 3 1 5]
      Infer acc:0.720000
      =======================================================================
      ```

3. 性能测试。

      开启性能统计开关，在sdk.conf配置文件中，设置 enable_ps=true，开启性能统计开关。

      调整性能统计时间间隔，设置ps_interval_time=2，每隔2秒，进行一次性能统计。

      进入infer/sdk目录，执行推理命令脚本，启动SDK推理服务 。

4. 查看性能结果。

      在日志目录"~/MX_SDK_HOME/logs"查看性能统计结果。

      ```
      performance-statistics.log.e2e.xx×
      performance-statistics.log.plugin.xx×
      performance-statistics.log.tpr.xxx
      ```

      其中e2e日志统计端到端时间，plugin日志统计单插件时间。

# 在ModelArts上应用

## 创建OBS桶

1. 创建桶。

   * 点击”创建桶“
   * ”区域“选择”华北-北京四“
   * ”存储类别“选取”标准存储“
   * ”桶ACL“选取”私有“
   * 关闭”多AZ“
   * 输入全局唯一桶名称, 例如 “S3"
   * 点击”确定“

2. 创建文件夹存放数据。

   在创建的桶中创建以下文件夹：

   * gat：存放训练脚本、数据集、训练生成ckpt模型
   * logs：存放训练日志目录

3. 上传代码

   * 进入 gat 代码文件根目录
   * 将 gat 目录下的文件全部上传至 obs://S3/gat 文件夹下

## 创建算法

1. 使用华为云帐号登录[ModelArts管理控制台](https://console.huaweicloud.com/modelarts)，在左侧导航栏中选择“算法管理”。
2. 在“我的算法管理”界面，单击左上角“创建”，进入“创建算法”页面。
3. 在“创建算法”页面，填写相关参数，然后单击“提交”。
4. 设置算法基本信息如下。

```text
# ==================================创建算法==========================================
   # (1) 上传你的代码和数据集到 S3 桶上
   # (2) 创建方式: 自定义脚本
   AI引擎：Ascend-Powered-Engine mindspore_1.3.0-cann_5.0.2-py_3.7-euler_2.8.3-aarch64
         代码目录： /S3/gat/
         启动文件： /S3/gat/trainmodelarts.py
   # (3) 超参：
         名称               类型            必需
         dataset           String          是
         data_dir          String          是
         output_dir        String          是
         train_nodes_num   Integer         是
   # (4) 自定义超参：支持
   # (5) 输入数据配置:  "映射名称 = '数据来源'", "代码路径参数 = 'data_dir'"
   # (6) 输出数据配置:  "映射名称 = '模型输出'", "代码路径参数 = 'output_dir'"
   # (7) 添加训练约束： 否
```

## 创建训练作业

1. 登录ModelArts。

2. 创建训练作业。

    训练作业参数配置说明如下。

   ```text
   # ==================================创建训练作业=======================================
   # (1) 算法： 在我的算法中选择前面创建的算法
   # (2) 训练输入： '/S3/gat/data/'
   # 在OBS桶/S3/gat/目录下新建results文件夹
   # (3) 训练输出： '/S3/gat/results/'

   # 训练 cora 数据集
   # (4) 超参：
            "dataset = 'cora'"
            "data_dir = 'obs://S3/gat/data/'"
            "output_dir='obs://S3/gat/results/'"
            "train_nodes_num=140"
   # 训练 citeseer 数据集
   # (4) 超参：
            "dataset = 'citeseer'"
            "data_dir = 'obs://S3/data/'"
            "output_dir='obs://S3/gat/results/'"
            "train_nodes_num=120"

   # (5) 设置作业日志路径
   ```

3. 单击“提交”，完成训练作业的创建。

   训练作业一般需要运行一段时间，根据您选择的数据量和资源不同，训练时间将耗时几分钟左右。训练结果模型将保存在 obs://S3/gat/results/model/ 文件夹下。

## 查看训练任务日志

1. 训练完成后进入logs文件夹，点击对应当次训练作业的日志文件即可。

2. logs文件夹内生成日志文件，您可在  /logs 文件夹下的日志文件中找到如下结果：

      ```text
      Epoch:0, train loss=1.99011, train acc=0.17143 | val loss=1.97765, val acc=0.28400
      Epoch:1, train loss=1.99014, train acc=0.19286 | val loss=1.97180, val acc=0.16400
      Epoch:2, train loss=1.97114, train acc=0.18571 | val loss=1.96929, val acc=0.15600
      Epoch:3, train loss=1.96467, train acc=0.12857 | val loss=1.96686, val acc=0.15600
      Epoch:4, train loss=1.96707, train acc=0.20714 | val loss=1.96546, val acc=0.15600
      Epoch:5, train loss=1.97173, train acc=0.18571 | val loss=1.96412, val acc=0.16400
      ...
      Epoch:195, train loss=1.42962, train acc=0.55714 | val loss=1.44235, val acc=0.80200
      Epoch:196, train loss=1.49868, train acc=0.49286 | val loss=1.44454, val acc=0.80000
      Epoch:197, train loss=1.46144, train acc=0.50000 | val loss=1.44664, val acc=0.78800
      Epoch:198, train loss=1.43329, train acc=0.49286 | val loss=1.44816, val acc=0.78200
      Epoch:199, train loss=1.32220, train acc=0.50714 | val loss=1.44778, val acc=0.78200
      Test loss=1.4548203, test acc=0.8319999
      ...
      ```

      可以得到最后训练的 cora 模型的测试精度为 0.832 ， citeseet 模型的测试精度为 0.732。