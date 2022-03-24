# Skipgram

## 准备推理数据

准备模型转换和模型推理所需目录及数据。

1. 单击“下载模型脚本”和“下载模型”，下载所需软件包。

2. 将源码上传至推理服务器任意目录并解压。

3. 进入容器。（根据实际情况修改命令）

   连接服务器后进入/infer目录执行以下命令进入容器：

   ```python
   bash docker_start_infer.sh [docker_image:tag] [model_dir]
   ```

4. 准备数据。

   由于后续推理均在容器中进行，因此需要把用于推理的图片、数据集、模型文件、代码等均放在同一数据路径中。

   > 注意：请先创建源码没有的文件夹。下载的模型文件以及创建的推理文件对应放在如下文件夹中，并且根据实际情况在程序中修改具体的文件目录。

   ```linux
   ..
   ├── infer                 # MindX高性能预训练模型新增
       ├── convert
          └──air-to-om.sh    # 转换om模型命令，AIPP
       ├── model             # 用于存放air、om模型文件
       ├── data              # 模型相关配置文件（如label、SDK的pipeline）
           └── config        # 推理所需的配置文件
               └── skipgram.pipeline  # pipeline文件
       ├── sdk               # 基于sdk.run包推理；如果是C++实现，存放路径一样
           ├── main.py       # 推理验证程序
           ├── run.sh
           ├── infer_eval.py       # 精度测试程序
       ├── docker_start_infer.sh                # 容器启动文件
   ```

   AIR模型可通过“模型训练”后转换生成或通过“下载模型”获取。

   **说明：**
   MindX SDK开发套件（mxManufacture)已安装在基础镜像中，安装路径：“/usr/local/sdk\_home“。

## 模型转换

在容器内进行模型转换（**提供模型转换参数、配置，以及转换后的om模型。**根据模型实际情况，提供多种模型转换方式，提供转换中涉及的参数配置信息，保证最终输出OM模型。

1. 下载.air格式模型文件。

2. 将下载好的模型文件存入/model文件夹

3. 模型转换。

   进入“infer/convert“目录进行模型转换，转换详细信息可查看转换脚本和对应的aipp配置文件，运行**air-to-om.sh**脚本文件中

   转换命令如下。

   **bash air-to-om.sh**

   > 注意：脚本文件中输入模型路径要根据实际情况在.sh文件中进行修改。

   **表 1**  参数说明

   | 参数              | 说明                                              |
   | ----------------- | ------------------------------------------------- |
   | model_path        | AIR文件路径。                                     |
   | output_model_name | 生成的OM文件名，转换脚本会在此基础上添加.om后缀。 |

## MindX SDK推理

> 注意：
>
> 模型完整的功能是推理单词之间的相似度，其需要完整的Embedding矩阵来进行精度的评估。
>
> 使用MindX SDK推理旨在验证模型输入输出正常，并且能够根据输入单词的索引位置来计算centerword的单词的向量表示。
>
> | 模型输入                         | 模型输出                     |
> | -------------------------------- | ---------------------------- |
> | 1.center word 单词索引位置编号   | 1.center word 单词的向量表示 |
> | 2.positive word 单词索引位置编号 |                              |
> | 3.negative word 单词索引位置编号 |                              |

1. 准备推理所需要的数据文件。（以center word为例：）

   - 进入../infer/sdk目录下。
   - 在main.py同级目录下创建test1.txt空文件。
   - 文件内容为单词索引，例如：48，55，63。（每个数字代表map中的一个单词）
   - 同样创建positive word、negative word数据文件test2.txt/test3.txt。

   > 注意：center word、positive word维度是1、negative word维度是5。所以，test3.txt中五个数字索引对应其他文件中一个索引。
>
   > 测试数据样例:
   >
   > test1.txt：48,55,63
   >
   > test2.txt：15,5,26
   >
   > test3.txt：2,106,169,5,9,48,2,46,655,5,852,2,4,3,7

2. 修改配置文件。

   1. 进入..\infer\data\config 目录下修改skipgram.pipeline文件。

      ```python
      "mxpi_tensorinfer0": {
                  "props": {
                      "dataSource": "appsrc0,appsrc1,appsrc2",
                      "modelPath": "../model/skipgram.om"         # 根据实际情况修改om模型位置
                  },
      ```

3. 进入../infer/sdk目录下修改main.py文件中的输入数据地址以及pipeline文件地址，例如：

   ```python
   send_source_data(num,0, "../test1.txt", streamName, streamManagerApi)
   with open("../data/config/skipgram.pipeline", 'rb') as f:
           pipelineStr = f.read()
   ```

4. 运行推理服务。

   1. 执行推理。

      ```python
      bash run.sh
      ```

   2. 查看推理结果。

      推理的结果可以在运行界面直接看到

## 精度测试

1. 下载下游任务数据集

   - 使用的任务集：[questions-words.txt]((https://code.google.com/archive/p/word2vec/source/default/source))
   - 数据格式：文本文件

2. 修改eval.py中修改测试文档路径为文件真实路径，例如：

   ```python
   args.eval_data_dir = "./eval_data/"
   ```

3. 修改eval.py中修改w2v_emb.npy文件路径为文件真实路径，例如：

   ```python
   w2v_emb_save_dir = "../../temp/w2v_emb"
   ```

4. 运行评估文件并查看结果

   ```python
   python3.7 infer_eval.py
   ```