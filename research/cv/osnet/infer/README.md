# OSNet MindX推理及mx Base推理

- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [准备推理数据](#准备推理数据)
    - [模型转换](#模型转换)
    - [mxBase推理](#mxBase推理)
    - [MindX SDK推理](#MindX-SDK推理)

## 脚本说明

### 脚本及样例代码

```text
├── infer                                 // 推理 MindX高性能预训练模型新增
    ├── convert                           // 转换om模型命令，AIPP
       ├── air_to_om.sh  
    ├── data                              // 包括模型文件、模型输入数据集、模型相关配置文件
       ├── config                         // 配置文件
           ├── OSNet.pipeline
       ├── data                           // 推理所需的数据集
           ├── market1501                 // 原始数据集
       ├── model                          // air、om模型文件
           ├── OSNet.air
           ├── OSNet.om
   ├── mxbase                             // mxbase推理
      ├── src
           ├──OSNet.cpp
           ├──OSNet.h
           ├──main.cpp
      ├── build.sh
      ├── CMakeLists.txt
   ├── sdk                                // sdk推理
      ├──main.py
   ├──utils
      ├── cal.py                          // 跑完推理计算精度
   ├──docker_start_infer.sh               // 启动容器脚本
```

### 准备推理数据

准备模型转换和模型推理所需目录及数据。

1. 下载数据集

- 进入容器执行以下命令:

启动容器,进入OSNet/infer目录,执行以下命令,启动容器。

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

3. 将模型脚本和模型上传至推理服务器任意目录并解压（如“/home/osnet”）

```shell
# 在环境上执行
unzip OSNet_for_MindSpore_{version}_code.zip
cd OSNet_for_MindSpore_{version}_code/infer && dos2unix `find .`
unzip ../../OSNet_for_MindSpore_{version}_model.zip
```

### 模型转换

以下操作均需进入容器中执行。

1. 准备模型文件。  

- OSNet.air

- 将文件放入OSNet/infer/data/model中

2. 模型转换。

    进入“OSNet/infer/convert“目录进行模型转换，转换详细信息可查看转换脚本和对应的aipp配置文件，**在air_to_om.sh**脚本文件中，配置相关参数。

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

转换命令如下:

```Shell
bash air_to_om.sh  [input_path] [output_path]
e.g.
bash air_to_om.sh  ../data/model/OSNet.air ../data/model/OSNet
```

**表 3**  参数说明

| 参数      | 说明 |
| ----------- | ----------- |
| input_path     | AIR文件路径。      |
| output_path   |生成的OM文件名，转换脚本会在此基础上添加.om后缀。       |

### mxBase推理

已进入推理容器环境,具体操作请参见“准备容器环境”。

- 修改osnet_config.yml中

```text
workers = 1
batch_size_test = 1
```

1. (1)修改数据预处理代码OSNet/preprocess.py的推理数据集路径以及处理完后数据保存的路径。

```Python
def preprocess(result_path):
    '''preprocess data to .bin files for ascend310'''
    _, query_dataset = dataset_creator(root='./infer/data/data', height=config.height, width=config.width,     # 原始数据集路径
                                       dataset=config.target, norm_mean=config.norm_mean,
                                       norm_std=config.norm_std, batch_size_test=config.batch_size_test,
                                       workers=config.workers, cuhk03_labeled=config.cuhk03_labeled,
                                       cuhk03_classic_split=config.cuhk03_classic_split, mode='query')
    _, gallery_dataset = dataset_creator(root='./infer/data/data', height=config.height,                       # 原始数据集路径
                                         width=config.width, dataset=config.target,
                                         norm_mean=config.norm_mean, norm_std=config.norm_std,
                                         batch_size_test=config.batch_size_test,
                                         workers=config.workers, cuhk03_labeled=config.cuhk03_labeled,
                                         cuhk03_classic_split=config.cuhk03_classic_split, mode='gallery')
    img_path = os.path.join(result_path, "img_data")
    label_path = os.path.join(result_path, "label")
    camid_path = os.path.join(result_path, "camlabel")
    os.makedirs(img_path)
    os.makedirs(label_path)
    os.makedirs(camid_path)

    for idx, data in enumerate(query_dataset.create_dict_iterator(output_numpy=True, num_epochs=1)):
        img_data = data["img"]
        img_label = data["pid"]
        img_cam_label = data["camid"]

        file_name = "query_" + str(config.target) + "_" + str(config.batch_size_test) + "_" + str(idx) + ".bin"
        img_file_path = os.path.join(img_path, file_name)
        img_data.tofile(img_file_path)

        label_file_path = os.path.join(label_path, file_name)
        img_label.tofile(label_file_path)

        camlabel_file_path = os.path.join(camid_path, file_name)
        img_cam_label.tofile(camlabel_file_path)

    for idx, data in enumerate(gallery_dataset.create_dict_iterator(output_numpy=True, num_epochs=1)):
        img_data = data["img"]
        img_label = data["pid"]
        img_cam_label = data["camid"]

        file_name = "gallery_" + str(config.target) + "_" + str(config.batch_size_test) + "_" + str(idx) + ".bin"
        img_file_path = os.path.join(img_path, file_name)
        img_data.tofile(img_file_path)

        label_file_path = os.path.join(label_path, file_name)
        img_label.tofile(label_file_path)

        camlabel_file_path = os.path.join(camid_path, file_name)
        img_cam_label.tofile(camlabel_file_path)

    print("=" * 20, "export bin files finished", "=" * 20)


if __name__ == '__main__':
    preprocess('./infer/data/pre_data')                              # 处理后数据集保存路径

```

(2)进入"OSNet/"目录下执行以下命令:

```Shell
python3 preprocess.py
```

执行后在preprocess.py设置的路径中得到处理后的数据,如

```Text
|——infer
    |——data
        |——pre_data
            |——label
            |——img_data
            |——camlabel
```

2. 根据实际情况修改OSNet.h文件中的推理结果保存路径。

```c
 private:
    std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
    MxBase::ModelDesc modelDesc_ = {};
    uint32_t deviceId_ = 0;
    std::string outputDataPath_ = "./result";                         //推理结果存储路径
    std::vector<uint32_t> inputDataShape_ = {1, 3, 256, 128};         //输入图片大小

    uint32_t inputDataSize_ = 98304;
```

3. 在“infer/mxbase”目录下，编译工程

```shell
bash build.sh
```

编译完成后在mxbase目录下得到以下新文件:

```text
├── mxbase
    ├── build                               // 编译后的文件
    ├── result                              //用于存放推理结果的空文件夹
    ├── OSNet                               // 用于执行的模型文件
```

 运行推理服务。

   在“infer/mxbase”目录下，运行推理程序脚本

```shell
./OSNet [model_path] [input_data_path/] [output_data_path]
e.g.
./OSNet ../data/model/OSNet.om ../data/pre_data/img_data/ ./result
```

**表 4** 参数说明：

| 参数             | 说明         |
| ---------------- | ------------ |
| model_path | 模型路径 |
| input_data_path | 处理后的数据路径 |
| output_data_path | 输出推理结果路径 |

4. 观察结果。

屏幕会打印推理结果，推理结果示例如下所示。

```Shell
root@c7ff72accbe0:/home/data/sdx_mindx/osnet/infer/mxbase# ./OSNet ../data/model/OSNet.om ../data/pre_data/img_data/ ./result
WARNING: Logging before InitGoogleLogging() is written to STDERR
I20220429 12:48:03.005131 103710 main.cpp:54] =======================================  !!!Parameters setting!!!========================================
I20220429 12:48:03.005193 103710 main.cpp:57] ==========  loading model weights from: ../data/model/OSNet.om
I20220429 12:48:03.005199 103710 main.cpp:60] ==========  input data path = ../data/pre_data/img_data/
I20220429 12:48:03.005208 103710 main.cpp:63] ==========  output data path = ./result WARNING: please make sure that this folder is created in advance!!!
I20220429 12:48:03.005213 103710 main.cpp:66] ========================================  !!!Parameters setting!!! ========================================
I20220429 12:48:03.204056 103710 ModelInferenceProcessor.cpp:22] Begin to ModelInferenceProcessor init
I20220429 12:48:03.383780 103710 ModelInferenceProcessor.cpp:69] End to ModelInferenceProcessor init
I20220429 12:48:03.400103 103710 main.cpp:86] Processing: 1/19281 ---> gallery_market1501_1_15099.bin
I20220429 12:48:03.400619 103710 OSNet.cpp:94] ==========  datasize ---> 393216
I20220429 12:48:03.409279 103710 OSNet.cpp:152] file path for saving result: ./result/gallery_market1501_1_15099.txt
I20220429 12:48:03.410348 103710 main.cpp:86] Processing: 2/19281 ---> gallery_market1501_1_10994.bin
I20220429 12:48:03.410562 103710 OSNet.cpp:94] ==========  datasize ---> 393216
I20220429 12:48:03.418956 103710 OSNet.cpp:152] file path for saving result: ./result/gallery_market1501_1_10994.txt
I20220429 12:48:03.419979 103710 main.cpp:86] Processing: 3/19281 ---> gallery_market1501_1_2615.bin
I20220429 12:48:03.420171 103710 OSNet.cpp:94] ==========  datasize ---> 393216
I20220429 12:48:03.428331 103710 OSNet.cpp:152] file path for saving result: ./result/gallery_market1501_1_2615.txt
I20220429 12:48:03.429333 103710 main.cpp:86] Processing: 4/19281 ---> gallery_market1501_1_3524.bin
I20220429 12:48:03.429540 103710 OSNet.cpp:94] ==========  datasize ---> 393216
I20220429 12:48:03.437510 103710 OSNet.cpp:152] file path for saving result: ./result/gallery_market1501_1_3524.txt
I20220429 12:48:03.438498 103710 main.cpp:86] Processing: 5/19281 ---> query_market1501_1_2514.bin
I20220429 12:48:03.438685 103710 OSNet.cpp:94] ==========  datasize ---> 393216
I20220429 12:48:03.446488 103710 OSNet.cpp:152] file path for saving result: ./result/query_market1501_1_2514.txt
I20220429 12:48:03.447482 103710 main.cpp:86] Processing: 6/19281 ---> query_market1501_1_1251.bin
I20220429 12:48:03.447669 103710 OSNet.cpp:94] ==========  datasize ---> 393216
I20220429 12:48:03.456101 103710 OSNet.cpp:152] file path for saving result: ./result/query_market1501_1_1251.txt
I20220429 12:48:03.457108 103710 main.cpp:86] Processing: 7/19281 ---> gallery_market1501_1_8220.bin
I20220429 12:48:03.457304 103710 OSNet.cpp:94] ==========  datasize ---> 393216
I20220429 12:48:03.465936 103710 OSNet.cpp:152] file path for saving result: ./result/gallery_market1501_1_8220.txt
I20220429 12:48:03.466929 103710 main.cpp:86] Processing: 8/19281 ---> gallery_market1501_1_13949.bin
I20220429 12:48:03.467118 103710 OSNet.cpp:94] ==========  datasize ---> 393216
I20220429 12:48:03.475270 103710 OSNet.cpp:152] file path for saving result: ./result/gallery_market1501_1_13949.txt
I20220429 12:48:03.476289 103710 main.cpp:86] Processing: 9/19281 ---> gallery_market1501_1_12555.bin
I20220429 12:48:03.476495 103710 OSNet.cpp:94] ==========  datasize ---> 393216
I20220429 12:48:03.484515 103710 OSNet.cpp:152] file path for saving result: ./result/gallery_market1501_1_12555.txt
I20220429 12:48:03.485651 103710 main.cpp:86] Processing: 10/19281 ---> gallery_market1501_1_13106.bin
I20220429 12:48:03.485848 103710 OSNet.cpp:94] ==========  datasize ---> 393216
I20220429 12:48:03.493811 103710 OSNet.cpp:152] file path for saving result: ./result/gallery_market1501_1_13106.txt
I20220429 12:48:03.494897 103710 main.cpp:86] Processing: 11/19281 ---> gallery_market1501_1_5269.bin
I20220429 12:48:03.495096 103710 OSNet.cpp:94] ==========  datasize ---> 393216
I20220429 12:48:03.503592 103710 OSNet.cpp:152] file path for saving result: ./result/gallery_market1501_1_5269.txt
I20220429 12:48:03.504662 103710 main.cpp:86] Processing: 12/19281 ---> gallery_market1501_1_3742.bin
I20220429 12:48:03.504873 103710 OSNet.cpp:94] ==========  datasize ---> 393216
I20220429 12:48:03.513337 103710 OSNet.cpp:152] file path for saving result: ./result/gallery_market1501_1_3742.txt
I20220429 12:48:03.515179 103710 main.cpp:86] Processing: 13/19281 ---> gallery_market1501_1_8480.bin
I20220429 12:48:03.515384 103710 OSNet.cpp:94] ==========  datasize ---> 393216
I20220429 12:48:03.523627 103710 OSNet.cpp:152] file path for saving result: ./result/gallery_market1501_1_8480.txt
I20220429 12:48:03.524935 103710 main.cpp:86] Processing: 14/19281 ---> gallery_market1501_1_15730.bin
I20220429 12:48:03.525136 103710 OSNet.cpp:94] ==========  datasize ---> 393216
I20220429 12:48:03.533098 103710 OSNet.cpp:152] file path for saving result: ./result/gallery_market1501_1_15730.txt
I20220429 12:48:03.534142 103710 main.cpp:86] Processing: 15/19281 ---> query_market1501_1_3255.bin
I20220429 12:48:03.534337 103710 OSNet.cpp:94] ==========  datasize ---> 393216
I20220429 12:48:03.542346 103710 OSNet.cpp:152] file path for saving result: ./result/query_market1501_1_3255.txt
I20220429 12:48:03.543452 103710 main.cpp:86] Processing: 16/19281 ---> gallery_market1501_1_9761.bin
I20220429 12:48:03.543649 103710 OSNet.cpp:94] ==========  datasize ---> 393216
I20220429 12:48:03.552081 103710 OSNet.cpp:152] file path for saving result: ./result/gallery_market1501_1_9761.txt
I20220429 12:48:03.553216 103710 main.cpp:86] Processing: 17/19281 ---> gallery_market1501_1_15189.bin
I20220429 12:48:03.553413 103710 OSNet.cpp:94] ==========  datasize ---> 393216
I20220429 12:48:03.561837 103710 OSNet.cpp:152] file path for saving result: ./result/gallery_market1501_1_15189.txt
...

```

 推理结果以txt格式保存。

 ```Text
|——result
    |——gallery_market1501_1_0.txt
    |——gallery_market1501_1_1.txt
    ...
    |——query_market1501_1_0.txt
    |——query_market1501_1_1.txt
    ...
```  

5. 推理精度
- 进入OSNet/infer/utils 目录下执行：

```Shell
python3 cal.py  [result_path]  [label_file]  [camlabel_file]
e.g
python3 cal.py  --result_path=../mxbase/result/ --label_file=../data/pre_data/label/ --camlabel_file=../data/pre_data/camlabel/
```

**表 5** 参数说明：

| 参数             | 说明           |
| ---------------- |--------------|
| result_path | 推理结果路径       |
| label_file | 数据预处理label路径 |
| camlabel_file | 数据预处理camlabel路径     |

6. 精度

```Text
** Results **
Dataset:market1501
mAP: 83.8%
CMC curve
Rank-1  : 93.9%
Rank-5  : 95.7%
Rank-10 : 96.9%
Rank-20 : 97.5%
```

### MindX SDK推理

已进入推理容器环境。具体操作请参见“准备容器环境”。

- 修改osnet_config.yml中

```text
workers = 1
batch_size_test = 1
```

1. 准备模型推理文件

    (1)进入OSNet/infer/data/config目录，OSNet.pipeline文件中的"modelPath": "../data/model/OSNet.om"为om模型所在路径。

```txt
    {
    "OSNet": {
        "stream_config": {
            "deviceId": "0"
        },
        "appsrc0": {
            "factory": "appsrc",
            "next": "mxpi_tensorinfer0"
        },
        "mxpi_tensorinfer0": {
            "props": {
                "dataSource": "appsrc0",
                "modelPath": "../data/model/OSNet.om"
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

(2) 根据实际情况修改main.py文件中的 **pipeline路径** 、**数据集路径**文件路径,这里的数据集为原始的推理数据集,不需要经过预处理。

```python
def run():
    """
    read pipeline and do infer
    """
    # init stream manager
    os.makedirs("./result")                         # 创建数据保存文件夹
    os.makedirs("./result/result")                  # 创建推理结果保存文件夹
    os.makedirs("./result/label")                   # 创建label数据预处理保存文件夹
    os.makedirs("./result/camlabel")                # 创建camlabel数据预处理保存文件夹
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        return

    # create streams by pipeline config file
    with open("../data/config/OSNet.pipeline", 'rb') as f:               # pipeline文件路径
        pipelineStr = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipelineStr)

    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        return
    _, query_dataset = dataset_creator(root='../data/data/', height=config.height, width=config.width,          # 原始数据集路径
                                       dataset=config.target, norm_mean=config.norm_mean,
                                       norm_std=config.norm_std, batch_size_test=config.batch_size_test,
                                       workers=config.workers, cuhk03_labeled=config.cuhk03_labeled,
                                       cuhk03_classic_split=config.cuhk03_classic_split, mode='query')
    _, gallery_dataset = dataset_creator(root='../data/data/', height=config.height,                             # 原始数据集路径
                                         width=config.width, dataset=config.target,
                                         norm_mean=config.norm_mean, norm_std=config.norm_std,
                                         batch_size_test=config.batch_size_test, workers=config.workers,
                                         cuhk03_labeled=config.cuhk03_labeled,
                                         cuhk03_classic_split=config.cuhk03_classic_split,
                                         mode='gallery')
    def feature_extraction(eval_dataset, name):
        infer_total_time = 0
        for idx, data in enumerate(eval_dataset.create_dict_iterator(output_numpy=True, num_epochs=1)):
            imgs, pids, camids = data['img'], data['pid'], data['camid']
            stream_name = b'OSNet'
            if not send_source_data(0, imgs, stream_name, stream_manager_api):
                return
            # Obtain the inference result by specifying streamName and uniqueId.
            key_vec = StringVector()
            key_vec.push_back(b'mxpi_tensorinfer0')
            start_time = time.time()
            infer_result = stream_manager_api.GetProtobuf(stream_name, 0, key_vec)
            infer_total_time += time.time() - start_time
            if infer_result.size() == 0:
                print("inferResult is null")
                return
            if infer_result[0].errorCode != 0:
                print("GetProtobuf error. errorCode=%d" % (infer_result[0].errorCode))
                return
            result = MxpiDataType.MxpiTensorPackageList()
            result.ParseFromString(infer_result[0].messageBuf)
            res = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype='<f4')
            if name == 'query':
                file_name = './result/result/' + "query_" + str(config.target) + "_" + str(config.batch_size_test) + "_" + str(idx) + ".txt"             # query result文件名称
                label_file_path = './result/label/' + "query_" + str(config.target) + "_" + str(config.batch_size_test) + "_" + str(idx) + ".bin"        # query label文件名称
                camlabel_file_path = './result/camlabel/' + "query_" + str(config.target) + "_" + str(config.batch_size_test) + "_" + str(idx) + ".bin"  # query camlabel文件名称
            else:
                file_name = './result/result/' + "gallery_" + str(config.target) + "_" + str(config.batch_size_test) + "_" + str(idx) + ".txt"           # gallery result文件名称
                label_file_path = './result/label/' + "gallery_" + str(config.target) + "_" + str(config.batch_size_test) + "_" + str(idx) + ".bin"      # gallery label文件名称
                camlabel_file_path = './result/camlabel/' + "gallery_" + str(config.target) + "_" + str(config.batch_size_test) + "_" + str(idx) + ".bin" # gallery camlabel文件名称
```

2. 进入OSNet/infer/sdk 目录下执行。

```Shell
python3 main.py
```

在脚本设置的数据保存文件夹中得到以下文件:

 ```Text
|——result
    |——result
    |——label
    |——camlabel
```

3. 推理精度
- 进入OSNet/infer/utils 目录下执行：

```Shell
python3 cal.py  [result_path]  [label_file]  [camlabel_file]
e.g
python3 cal.py  --result_path=../sdk/result/result/ --label_file=../sdk/result/label/ --camlabel_file=../sdk/result/camlabel/
```

**表 5** 参数说明：

| 参数             | 说明           |
| ---------------- |--------------|
| result_path | 推理结果路径       |
| label_file | 数据预处理label路径 |
| camlabel_file | 数据预处理camlabel路径     |

4. 精度

```Text
** Results **
Dataset:market1501
mAP: 83.8%
CMC curve
Rank-1  : 93.9%
Rank-5  : 95.8%
Rank-10 : 96.9%
Rank-20 : 97.5%
```