# SimCLR MindX推理及mx Base推理

- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [准备推理数据](#准备推理数据)
    - [模型转换](#模型转换)
    - [mxBase推理](#mxBase推理)
    - [MindX SDK推理](#MindX-SDK推理)

## 脚本说明

### 脚本及样例代码

准备模型转换和模型推理所需目录及数据，在infer/data目录下新建空文件夹data、model用于存放数据集和模型文件。

```text
├── infer                                  // 推理 MindX高性能预训练模型新增
    ├── convert                            // 转换om模型命令，AIPP
       ├── convert.sh
       ├── aipp.config  
    ├── data                               // 包括模型文件、模型输入数据集、模型相关配置文件
       ├── config                          // 配置文件
           ├── simclr.cfg
      ├── cifar-10-batches-py              // 原始数据集
      ├── cifar10.py                       // 处理原始数据集代码
      ├── model                            // air、om模型文件
           ├── simclr.air
           ├── simclr.om
   ├── mxbase                              // mxbase推理
      ├── main.cpp
      ├── run.sh
      ├── SimCLRClassifyOpencv.cpp
      ├── SimCLRClassifyOpencv.h
      ├── build.sh
      ├── CMakeLists.txt
      ├── result                          // 用来存放mxbase推理结果
   ├── sdk                                // sdk推理
      ├──run.sh
      ├──main.py
      ├──simclr.pipeline
      ├──classification_task_metric.py
   ├──docker_start_infer.sh               // 启动容器脚本
```

### 准备推理数据

1. 下载数据集

- 进入容器执行以下命令:

- 启动容器。

进入simclr/infer目录,执行以下命令,启动容器。

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

3. 将模型脚本和模型上传至推理服务器任意目录并解压（如“/home/simclr”）

```shell
# 在环境上执行
unzip simclr_for_MindSpore_{version}_code.zip
cd simclr_for_MindSpore_{version}_code/infer && dos2unix `find .`
unzip ../../simclr_for_MindSpore_{version}_model.zip
```

4. 下载数据集cifar10,下载链接cifar10 download url: http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

### 模型转换

以下操作均需进入容器中执行。

1. 准备模型文件。  

- simclr.air

- 将文件放入simclr/infer/data/model中

2. 模型转换。

    进入“simclr/infer/convert“目录进行模型转换，转换详细信息可查看转换脚本和对应的aipp配置文件，**在convert.sh**脚本文件中，配置相关参数。

```Shell
input_air_path=$1
aipp_cfg_file=$2
output_om_path=$3

atc --input_format=NCHW \
    --framework=1 \                        # 1代表MindSpore。
    --model="${input_air_path}" \          # 待转换的air模型，模型可以通过训练生成或通过“下载模型”获得。
    --input_shape="x:1,3,32,32"  \         # 输入数据的shape。input取值根据实际使用场景确定。
    --output="${output_om_path}" \         # 转换后输出的om模型。
    --insert_op_conf="${aipp_cfg_file}" \  # aipp配置文件。
    --soc_version=Ascend310 \              # 模型转换时指定芯片版本。
    --precision_mode=allow_fp32_to_fp16 \  # 混合精度
    --op_select_implmode=high_precision \  # 高精度
    --output_type=FP32                     # 精度级别
```

转换命令如下:

```Shell
bash convert.sh air_path aipp_cfg_path om_path
e.g.
bash convert.sh ../data/model/simclr.air ./aipp.config ../data/model/simclr
```

**表 3**  参数说明

| 参数      | 说明 |
| ----------- | ----------- |
| air_path     | AIR文件路径。      |
| aipp_cfg_path | aipp配置文件路径。 |
| output_model_name   | 生成的OM文件名，转换脚本会在此基础上添加.om后缀。       |

### mxBase推理

已进入推理容器环境,具体操作请参见“准备容器环境”。

1. (1)修改预处理代码simclr/infer/data/cifar10.py的推理数据集路径,处理完后数据保存的路径。

```Python
loc_1 = './train_cifar10/' # 处理后保存train集的路径（不要手动建立）
loc_2 = './test_cifar10/'  # 处理后保存test集的路径（不要手动建立）

def unpickle(file_name):
    import pickle
    with open(file_name, 'rb') as fo:
        dict_res = pickle.load(fo, encoding='bytes')
    return dict_res

def cifar10_img():
    """change img"""
    file_dir = './cifar-10-batches-py' #原始数据集路径
    os.mkdir(loc_1)
    os.mkdir(loc_2)
```

(2)进入"simclr/infer/data"目录下执行以下命令:

```Shell
python3 cifar10.py
```

执行后在cifar10.py设置的路径中得到处理后的数据,如

```Text
|——test_cifar10
    |——010003.bmp
    |——010010.bmp
    |——010021.bmp
    ...
|——test_label.txt
```

2. 在infer/mxbase下创建result目录作为结果保存目录。

```shell
  mkdir result
```

3. 修改main.py的main函数中的初始化参数。

```c++
 InitParam initParam = {};
 initParam.deviceId = 0;
 initParam.classNum = CLASS_NUM;
 initParam.labelPath = "../data/config/cifar10_clsidx_to_labels.names"; ##根据实际存放位置进行修改
 initParam.topk = 5;
 initParam.softmax = false;
 initParam.checkTensor = true;
 initParam.modelPath = "../data/model/simclr.om";##根据实际存放位置进行修改
 auto simclr = std::make_shared<SimCLRClassifyOpencv>();
 APP_ERROR ret = simclr->Init(initParam);
 if (ret != APP_ERR_OK) {
     LogError << "SimCLRClassify init failed, ret=" << ret << ".";
     return ret;
  }
```

3. 在“infer/mxbase”目录下，编译工程

```shell
bash build.sh
```

 运行推理服务。

   在“infer/mxbase”目录下，运行推理程序脚本

```shell
 bash run.sh
```

编译完成后在mxbase目录下得到以下新文件:

```text
├── mxbase
    ├── build                               // 编译后的文件
    ├── result                              //用于存放推理结果的空文件夹
    ├── simclr                               // 用于执行的模型文件
```

6. 观察结果。

屏幕会打印推理结果，推理结果示例如下所示。

```Shell
    I0225 08:32:36.061944  1390 ModelInferenceProcessor.cpp:22] Begin to ModelInferenceProcessor init
I0225 08:32:36.233608  1390 ModelInferenceProcessor.cpp:69] End to ModelInferenceProcessor init
I0225 08:32:36.233760  1390 Resnet50PostProcess.cpp:57] Start to Init Resnet50PostProcess.
I0225 08:32:36.233768  1390 PostProcessBase.cpp:69] Start to LoadConfigDataAndLabelMap in  PostProcessBase.
I0225 08:32:36.235853  1390 Resnet50PostProcess.cpp:66] End to Init Resnet50PostProcess.
I0225 08:32:36.249092  1390 SimCLRClassifyOpencv.cpp:96] image size after resize32 32
I0225 08:32:36.253885  1390 Resnet50PostProcess.cpp:79] Start to Process Resnet50PostProcess.
I0225 08:32:36.253993  1390 Resnet50PostProcess.cpp:126] End to Process Resnet50PostProcess.
I0225 08:32:36.254004  1390 SimCLRClassifyOpencv.cpp:154] image path/home/data/zd_mindx/simclr/infer/data/test_cifar10/010003.bmp
I0225 08:32:36.254012  1390 SimCLRClassifyOpencv.cpp:158] file path for saving resultresult/010003_1.txt
I0225 08:32:36.254070  1390 SimCLRClassifyOpencv.cpp:170]  className:{0: 'airplane', confidence:0.984752 classIndex:0
I0225 08:32:36.254092  1390 SimCLRClassifyOpencv.cpp:170]  className:2: 'bird', confidence:0.0139367 classIndex:2
I0225 08:32:36.254101  1390 SimCLRClassifyOpencv.cpp:170]  className:3: 'cat', confidence:0.000629689 classIndex:3
I0225 08:32:36.254104  1390 SimCLRClassifyOpencv.cpp:170]  className:7: 'horse', confidence:0.000453366 classIndex:7
I0225 08:32:36.254108  1390 SimCLRClassifyOpencv.cpp:170]  className:8: 'ship', confidence:0.000139931 classIndex:8
I0225 08:32:36.254278  1390 SimCLRClassifyOpencv.cpp:96] image size after resize32 32
I0225 08:32:36.258800  1390 Resnet50PostProcess.cpp:79] Start to Process Resnet50PostProcess.
I0225 08:32:36.258880  1390 Resnet50PostProcess.cpp:126] End to Process Resnet50PostProcess.

```

 推理结果以txt格式保存。

 ```Text
|——result
    |——010003_1.txt
    |——010010_1.txt
    |——010021_1.txt
    ...
```  

### MindX SDK推理

已进入推理容器环境。具体操作请参见“准备容器环境”。

1. 准备模型推理文件

    (1)进入simclr/infer/sdk目录，simclr.pipeline文件中的"modelPath": "../model/simclr.om"为om模型所在路径。

```txt
{
    "im_simclr": {
        "stream_config": {
            "deviceId": "0"
        },
        "appsrc1": {
            "props": {
                "blocksize": "409600"
            },
            "factory": "appsrc",
            "next": "mxpi_imagedecoder0"
        },
        "mxpi_imagedecoder0": {
            "props": {
                "handleMethod": "opencv"
            },
            "factory": "mxpi_imagedecoder",
            "next": "mxpi_imageresize0"
        },
        "mxpi_imageresize0": {
            "props": {
                "handleMethod": "opencv",
                "resizeType": "Resizer_Stretch",
                "resizeHeight": "32",
                "resizeWidth": "32"
            },
            "factory": "mxpi_imageresize",
            "next": "mxpi_tensorinfer0"
        },
        "mxpi_tensorinfer0": {
            "props": {
                "dataSource": "mxpi_imageresize0",
                "modelPath": "../data/model/simclr.om", ##路径根据实际修改
                "waitingTime": "2000",
                "outputDeviceId": "-1"
            },
            "factory": "mxpi_tensorinfer",
            "next": "mxpi_classpostprocessor0"
        },
        "mxpi_classpostprocessor0": {
            "props": {
                "dataSource": "mxpi_tensorinfer0",
                "postProcessConfigPath": "../data/config/simclr.cfg", ##路径根据实际修改
                "labelPath": "../data/config/cifar10_clsidx_to_labels.names", ##路径根据实际修改
                "postProcessLibPath": "../../../lib/modelpostprocessors/libresnet50postprocess.so" ##路径根据实际修改
            },
            "factory": "mxpi_classpostprocessor",
            "next": "mxpi_dataserialize0"
        },
        "mxpi_dataserialize0": {
            "props": {
                "outputDataKeys": "mxpi_classpostprocessor0"
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

(2) 根据实际情况修改main.py文件中的 **pipeline路径**。

```python
def run():
    # init stream manager
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        return

    # create streams by pipeline config file
    with open("./simclr.pipeline", 'rb') as f:  #修改为实际pipeline路径
        pipelineStr = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipelineStr)

    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        return
```

2. 运行推理服务,进入simclr/infer/sdk 目录下执行。

```Shell
bash run.sh ../data/test_cifar10/ simclr_result
```

3. 在simclr/infer/sdk下执行以下命令运算精度。

```shell
python3.7 classification_task_metric.py simclr_result/ ../data/test_lable.txt ./ ./simclr_result.json
```

4. 查看simclr_result.json观察结果。

```json
{
  "title": "Overall statistical evaluation",
  "value": [
    {
      "key": "Number of images",
      "value": "10000"
    },
    {
      "key": "Number of classes",
      "value": "5"
    },
    {
      "key": "Top1 accuracy",
      "value": "84.5%"
    },
    {
      "key": "Top2 accuracy",
      "value": "94.41%"
    },
    {
      "key": "Top3 accuracy",
      "value": "97.44%"
    },
    {
      "key": "Top4 accuracy",
      "value": "98.79%"
    },
    {
      "key": "Top5 accuracy",
      "value": "99.44%"
    }
  ]
}
```