# Transformer MindX推理及mxBase推理

- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [准备推理数据](#准备推理数据)
    - [模型转换](#模型转换)
    - [mxBase推理](#mxBase推理)
    - [MindX SDK推理](#MindX-SDK推理)

## 脚本说明

### 脚本及样例代码

```text
├── infer                                  // 推理 MindX高性能预训练模型新增
    ├── convert                            // 转换om模型命令，AIPP
       ├── air_to_om.sh  
    ├── data                               // 包括模型文件、模型输入数据集、模型相关配置文件
       ├── config                          // 配置文件
           ├── transformer.pipeline
       ├── data                            // 推理所需的数据集
           ├── 00_source_eos_ids
           ├── 01_source_eos_mask          // 经过处理后的数据集
           ├── vocab.bpe.32000             // 计算精度所用数据集
           ├── newstest2014.tok.de         // 计算精度所用数据集
           ├── test.all                    // 原始数据集
           ├──newstest2014-l128-mindrecord
           ├──newstest2014-l128-mindrecord.db
      ├── model                            // air、om模型文件
           ├── transformer.air
           ├── transformer.om
   ├── mxbase                              // mxbase推理
      ├── src
           ├──transformer.cpp
           ├──Transformer.h
           ├──main.cpp
      ├── build.sh
      ├── CMakeLists.txt
      ├── post_process.py
   ├── sdk                                // sdk推理
      ├──main.py
   ├──docker_start_infer.sh               // 启动容器脚本
   ├──multi-bleu.perl                     // 计算精度脚本
```

### 准备推理数据

准备模型转换和模型推理所需目录及数据。

1. 处理数据集

- 进入容器执行以下命令:

启动容器,进入Transformer/infer目录,执行以下命令,启动容器。

```Shell
bash docker_start_infer.sh  docker_image  tag  model_dir
```

**表 2**  参数说明

  | 参数      | 说明 |
  | ----------- | ----------- |
  | docker_image      | 推理镜像名称，根据实际写入。      |
  | tag   | 镜像tag，请根据实际配置，如：21.0.2。       |
  | model_dir  | 推理代码路径。      |

- 启动容器时会将推理芯片和数据路径挂载到容器中。可根据需要通过修改**docker\_start\_infer.sh**的device来指定挂载的推理芯片。

```Shell
docker run -it
--device=/dev/davinci0        # 可根据需要修改挂载的npu设备
--device=/dev/davinci_manager
```

>**说明：**  
>MindX SDK开发套件（mxManufacture）已安装在基础镜像中，安装路径：“/usr/local/sdk_home“。

2. 下载软件包。

   单击“下载模型脚本”和“下载模型”，下载所需软件包。

3. 将模型脚本和模型上传至推理服务器任意目录并解压（如“/home/Transformer”）

```shell
# 在环境上执行
unzip Transformer_for_MindSpore_{version}_code.zip
cd Transformer_for_MindSpore_{version}_code/infer && dos2unix `find .`
unzip ../../Transformer_for_MindSpore_{version}_model.zip
```

4. 启动容器后，进入“Transformer“代码目录

执行命令如下:

```Shell
bash wmt16_en_de.sh
```

假设您已获得下列文件,将以下文件移入到代码目录“Transformer/infer/data/data/“目录下

```text
├── wmt16_en_de
    vocab.bpe.32000
    newstest2014.tok.bpe.32000.en
    newstest2014.tok.bpe.32000.de
    newstest2014.tok.de
```

进入“Transformer/infer/data/data/“目录

执行命令如下:

```Shell
paste newstest2014.tok.bpe.32000.en newstest2014.tok.bpe.32000.de > test.all
```

将default_config.yaml中bucket改为bucket: [128]

```text
# create_data.py
input_file: ''
num_splits: 16
clip_to_max_len: False
max_seq_length: 128
bucket: [128]
```

进入“Transformer/“目录

执行命令如下:

```Shell
python3 create_data.py --input_file ./infer/data/data/test.all --vocab_file ./infer/data/data/vocab.bpe.32000 --output_file ./infer/data/data/newstest2014-l128-mindrecord --num_splits 1 --max_seq_length 128 --clip_to_max_len True
```

更改default_config.yaml中参数：

```text
#eval_config/cfg edict
data_file: './infer/data/data/newstest2014-l128-mindrecord'
...
#'preprocess / from eval_config'
result_path: "./infer/data/data"
```

接着执行命令：

```Shell
python3 preprocess.py
```

执行后在“Transformer/infer/data/data“目录中得到文件夹如下:

```txt
├──data
   00_source_eos_ids
   01_source_eos_mask
```

### 模型转换

以下操作均需进入容器中执行。

1. 准备模型文件。  

- transformer.air

- 将文件放入Transformer/infer/data/model中

2. 模型转换。

    进入“Transformer/infer/convert“目录进行模型转换，转换详细信息可查看转换脚本和对应的aipp配置文件，**在air_to_om.sh**脚本文件中，配置相关参数。

```Shell
model_path=$1
output_model_name=$2

atc --model=$model_path \                             # 带转换模型的路径
    --framework=1 \                                   # 1表示MindSpore
    --output=$output_model_name \                     # 输出om模型的路径
    --input_format=NCHW \                             # 输入格式
    --soc_version=Ascend310 \                         # 模型转换时指定芯片版本
    --op_select_implmode=high_precision \             # 模型转换精度
    --precision_mode=allow_fp32_to_fp16               # 模型转换精度
```

转换命令如下:

```Shell
bash air_to_om.sh  [input_path] [output_path]
e.g.
bash air_to_om.sh ../data/model/transformer.air ../data/model/transformer
```

**表 3**  参数说明

| 参数      | 说明 |
| ----------- | ----------- |
| input_path     | AIR文件路径。      |
| output_path   |生成的OM文件名，转换脚本会在此基础上添加.om后缀。       |

### mxBase推理

已进入推理容器环境,具体操作请参见“准备容器环境”。

1. 根据实际情况修改Transformer.h文件中的推理结果保存路径。

```c
 private:
    std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
    MxBase::ModelDesc modelDesc_ = {};
    uint32_t deviceId_ = 0;
    std::string outputDataPath_ = "./result/result.txt";
};

#endif
```

2. 在“infer/mxbase”目录下，编译工程

```shell
bash build.sh
```

编译完成后在mxbase目录下得到以下新文件:

```text
├── mxbase
    ├── build                               // 编译后的文件
    ├── result                              //用于存放推理结果的空文件夹
    ├── Transformer                         // 用于执行的模型文件
```

 运行推理服务。

   在“infer/mxbase”目录下，运行推理程序脚本

```shell
./Transformer [model_path] [input_data_path/] [output_data_path]
e.g.
./Transformer ../data/model/transformer.om ../data/data ./result
```

**表 4** 参数说明：

| 参数             | 说明         |
| ---------------- | ------------ |
| model_path | 模型路径 |
| input_data_path | 处理后的数据路径 |
| output_data_path | 输出推理结果路径 |

3. 处理结果。

修改参数

```python
path = "./result"                  #推理结果所在文件夹

filenames = os.listdir(path)
result = "./results.txt"           #处理推理结果后文件所在路径
```

在“infer/mxbase”目录下执行:

```shell
python3 post_process.py
```

在“infer/mxbase”目录下得到results.txt

4. 推理精度

进入"Transformer/"目录下执行：

```shell
bash scripts/process_output.sh REF_DATA EVAL_OUTPUT VOCAB_FILE
e.g.
bash scripts/process_output.sh ./infer/data/data/newstest2014.tok.de ./infer/mxbase/results.txt ./infer/data/data/vocab.bpe.32000
```

进入"Transformer/infer/"目录下执行：

```shell
perl multi-bleu.perl REF_DATA.forbleu < EVAL_OUTPUT.forbleu
e.g.
perl multi-bleu.perl ./data/data/newstest2014.tok.de.forbleu < ./mxbase/results.txt.forbleu
```

得到精度BLEU为27.24

### MindX SDK推理

已进入推理容器环境。具体操作请参见“准备容器环境”。

1. 准备模型推理文件

    (1)进入Transformer/infer/data/config目录，transformer.pipeline文件中的"modelPath": "../model/transformer.om"为om模型所在路径。

```txt
    {
    "transformer": {
        "stream_config": {
            "deviceId": "0"
        },
        "appsrc0": {
            "props": {
                "blocksize": "409600"
            },
            "factory": "appsrc",
            "next": "mxpi_tensorinfer0:0"
        },
        "appsrc1": {
            "props": {
                "blocksize": "409600"
            },
            "factory": "appsrc",
            "next": "mxpi_tensorinfer0:1"
        },
        "mxpi_tensorinfer0": {
            "props": {
                "dataSource":"appsrc0,appsrc1",
                "modelPath": "../data/model/transformer.om"
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
            "props": {
                "blocksize": "4096000"
            },
            "factory": "appsink"
        }
    }
}
```

(2) 根据实际情况修改main.py文件中的 **pipeline路径** 、**数据集路径**、**推理结果路径**文件路径。

```python
def run():
    """
    read pipeline and do infer
    """
    # init stream manager
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        return

    # create streams by pipeline config file
    with open("../data/config/transformer.pipeline", 'rb') as f:                       #pipeline路径
        pipelineStr = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipelineStr)

    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        return
    stream_name = b'transformer'
    predictions = []
    path = '../data/data/00_source_eos_ids'                                           #数据集路径
    path1 = '../data/data/01_source_eos_mask'                                         #数据集路径
    files = os.listdir(path)
    for i in range(len(files)):
        full_file_path = os.path.join(path, "transformer_bs_1_" + str(i) + ".bin")
        full_file_path1 = os.path.join(path1, "transformer_bs_1_" + str(i) + ".bin")
        source_ids = np.fromfile(full_file_path, dtype=np.int32)
        source_mask = np.fromfile(full_file_path1, dtype=np.int32)
        source_ids = np.expand_dims(source_ids, 0)
        source_mask = np.expand_dims(source_mask, 0)
        print(source_ids)
        print(source_mask)
        if not send_source_data(0, source_ids, stream_name, stream_manager_api):
            return
        if not send_source_data(1, source_mask, stream_name, stream_manager_api):
            return
        # Obtain the inference result by specifying streamName and uniqueId.
        key_vec = StringVector()
        key_vec.push_back(b'mxpi_tensorinfer0')
        infer_result = stream_manager_api.GetProtobuf(stream_name, 0, key_vec)
        if infer_result.size() == 0:
            print("inferResult is null")
            return
        if infer_result[0].errorCode != 0:
            print("GetProtobuf error. errorCode=%d" % (infer_result[0].errorCode))
            return
        result = MxpiDataType.MxpiTensorPackageList()
        result.ParseFromString(infer_result[0].messageBuf)
        res = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.int32)
        print(res)
        predictions.append(res.reshape(1, 1, 81))
    # decode and write to file
    f = open('./results', 'w')                                                        #推理结果路径
    for batch_out in predictions:
        token_ids = [str(x) for x in batch_out[0][0].tolist()]
        f.write(" ".join(token_ids) + "\n")
    f.close()
    # destroy streams
    stream_manager_api.DestroyAllStreams()
```

2. 运行推理服务,进入“Transformer/infer/sdk” 目录下执行。

```Shell
python3 main.py
```

3. 计算推理精度。

进入"Transformer/"目录下执行：

```shell
bash scripts/process_output.sh REF_DATA EVAL_OUTPUT VOCAB_FILE
e.g.
bash scripts/process_output.sh ./infer/data/data/newstest2014.tok.de ./infer/sdk/results ./infer/data/data/vocab.bpe.32000
```

进入"Transformer/infer/"目录下执行：

```shell
perl multi-bleu.perl REF_DATA.forbleu < EVAL_OUTPUT.forbleu
e.g.
perl multi-bleu.perl ./data/data/newstest2014.tok.de.forbleu < ./sdk/results.forbleu
```

得到精度BLEU为27.24