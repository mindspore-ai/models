
# Duconv MindX推理及mx Base推理

- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [准备推理数据](#准备推理数据)
    - [模型转换](#模型转换)
    - [mxBase推理](#mxBase推理)
    - [MindX SDK推理](#MindX-SDK推理)

## 脚本说明

### 脚本及样例代码

准备模型转换和模型推理所需目录及数据。

```text
    ├── infer
        ├── README_duconv               // duconv MindX推理及mx Base推理说明文件
        ├── convert
           ├── air_to_om.sh             // air转换om模型命令
        ├── data                        // 包括模型文件、模型输入数据集、模型相关配置文件
           ├── config
               ├── duconv.pipeline      // 配置文件
           ├── data                     // 推理所需的数据集
               ├── labels_list.txt
               ├── kn_seq_length.txt
               ├── kn_id.txt
               ├── context_segment_id.txt
               ├── context_pos_id.txt
               ├── context_id.txt
          ├── model                     // air、om模型文件
               ├── duconv.air
               ├── duconv.om
       ├── mxbase                      // mxbase推理
          ├── src
               ├──duconv.cpp           // 详细实现
               ├──duconv.h             // 头文件
               ├──main.cpp             // mxBase 主函数
          ├── build.sh                 // 代码编译脚本
          ├── CMakeLists.txt           // 代码编译设置
       ├── sdk                         // sdk推理
          ├──main.py                   // MindX SDK 运行脚本
          ├──run.sh                    // MindX SDK 运行脚本
       ├──docker_start_infer.sh        // 启动容器脚本
       ├──requirements.txt             // 安装库文件tqdm
       ├──Dockerfile                   // 安装库文件tqdm
  ├── data                             // 推理训练所需数据集
       ├── build.test.txt
       ├── candidate.test.txt
       ├── gene.dict
  ├── src                   //获取精度所需文件
       ├──utils
          ├──extract.py
       ├──eval.py
```

### 准备推理数据

1. 在 infer 目录下或其他文件夹下创建数据集存放文件夹（记为“duconv/data/”），并将所需数据集上传到该文件夹下。“duconv/data/” 下的数据集组织结构如下：

   ```text
   ├── data                     // 推理训练所需数据集
          ├── build.test.txt
          ├── candidate.test.txt
          ├── gene.dict
   ```

2. 数据预处理

   - 进入duconv/infer/dataprocess 目录下执行

         python3.7 dataprocess.py --task_name=match_kn_gene  --vocab_path=[the path of gene.dict] --input_file=[the path of build.test.txt] --output_path=[output_path]
    例如:

          python3.7 dataprocess.py --task_name=match_kn_gene  --vocab_path=duconv/data/gene.dict --input_file=duconv/data/build.test.txt --output_path=duconv/infer/data/data
   - 处理完后在duconv/infer/data/data 目录中得到context_id.txt,context_pos_id.txt,context_segment_id.txt,kn_id.txt,kn_seq_length.txt,labels_list.txt 六个文件

   ```text
   ├── data
          ├── context_id.txt
          ├── context_pos_id.txt
          ├── context_segment_id.txt
          ├── kn_id.txt
          ├── kn_seq_length.txt
          ├── labels_list.txt
   ```

### 模型转换

以下操作均需进入容器中执行。

1. 准备模型文件。  

   - duconv.air

   - 将文件放入duconv/infer/data/model中

2. 模型转换。

    进入“duconv/infer/convert“目录进行模型转换，转换详细信息可查看转换脚本和对应的aipp配置文件，在air_to_om.sh脚本文件中，配置相关参数。

```Shell
       model_path=$1
       output_model_name=$2

       atc --model=$model_path \                             # 带转换模型的路径
           --framework=1 \                                   # 1表示MindSpore
           --output=$output_model_name \                     # 输出om模型的路径
           --input_format=NCHW \                             # 输入格式
           --soc_version=Ascend310 \                         # 模型转换时指定芯片版本
           --precision_mode=allow_fp32_to_fp16               # 模型转换精度
```

转换命令如下。```bash air_to_om.sh  ../data/model/duconv.air ../data/model/duconv```

**表 1**  参数说明

| 参数      | 说明 |
| ----------- | ----------- |
| model_path     | AIR文件路径。      |
| output_model_name   |生成的OM文件名，转换脚本会在此基础上添加.om后缀。       |

### mxBase推理

推理需要安装库文件tqdm，详情见requirements.txt和Dockerfile文档。
在容器内用mxBase进行推理。

进入到duconv/infer/mxbase 目录下:

1. 修改文件中的配置 main.cpp文件中' initParam.modelPath = "../data/model/duconv.om" ' 为所需要的om模型的路径,' std::string dataPath = "../data/data" ',该路径为dataprocess所处理得到数据集的路径。在duconv.cpp中' std::string argsort_path = "./results/score.txt" ',为最后得到推理结果所保存的路径。
2. 编译工程。
   bash build.sh 编译成功后在duconv/infer/mxbase得到build文件夹
3. 在duconv/infer/mxbase 目录下新建文件夹results
4. 运行推理服务。

   - build.test.txt为之前下载的数据集,存在duconv/data 目录下

   - 进入到duconv/infer/mxbase 目录下执行

      ```./build/duconv [the path of build.test.txt]```

   - 例如:

         ```./build/duconv duconv/data/build.test.txt```

5. 观察结果。

   推理结果保存至duconv/infer/mxbase/results中的score.txt

6. 推理精度

   - 进入duconv/src/utils 目录下执行

     ```python3.7 extract.py [the path of candidate.test.txt] [the path of score.txt] [outputpath]```

   - 例如:

     ```python3.7 extract.py duconv/data/candidate.test.txt duconv/infer/mxbase/results/score.txt duconv/infer/mxbase/results/result.txt,其中duconv/infer/mxbase/results/result.txt 为所得到的结果。```

   - 进入duconv/src 目录下执行

     ```python3.7 eval.py [the path of result.txt]```

   - 例如:

      ```python3.7 eval.py duconv/infer/mxbase/results/result.txt```

8. 精度

   - F1: 30.81%

   - BLEU1: 0.280%

   - BLEU2: 0.149%

   - DISTINCT1: 0.117%

   - DISTINCT2: 0.398%

### MindX SDK推理

1. 修改配置文件。

- 修改pipeline文件，"modelPath": "../data/model/duconv.om"为om模型所在路径,appsrc0-appsrc4分别为模型的5个输入。

2. 运行推理服务。
    1. 在duconv/infer/sdk 目录下新建文件夹results
    2. 执行推理。
   进入duconv/infer/sdk 目录下执行

       ```bash run.sh```

    3. 查看推理结果:推理结果保存至duconv/infer/sdk/results 中的score.txt

4. 执行精度测试。

- 进入duconv/src/utils 目录下执行

 ```python3.7 extract.py [the path of candidate.test.txt] [the path of score.txt] [outputpath]```

- 例如:

 ```python3.7 extract.py duconv/data/candidate.test.txt duconv/infer/sdk/results/score.txt duconv/infer/sdk/results/result.txt,其中duconv/infer/sdk/results/result.txt 为所得到的结果。```

- 进入duconv/src 目录下执行

 ```python3.7 eval.py [the path of result.txt]```

- 例如:

 ```python3.7 eval.py duconv/infer/sdk/results/result.txt```

5. 查看精度结果。

- F1: 30.81%

- BLEU1: 0.280%

- BLEU2: 0.149%

- DISTINCT1: 0.117%

- DISTINCT2: 0.398%
