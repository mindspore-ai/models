# Srcnn MindX推理及mx Base推理

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
       ├── air_to_om.sh  
    ├── data                               // 包括模型文件、模型输入数据集、模型相关配置文件
       ├── config                          // 配置文件
           ├── srcnn.pipeline
       ├── data                            // 推理所需的数据集
           ├── Set5                        // 原始数据集
           ├── Set                         // 经过处理后的数据集
           ├── result                      // 用来存放sdk推理结果
      ├── model                            // air、om模型文件
           ├── srcnn.air
           ├── srcnn.om
   ├── mxbase                              // mxbase推理
      ├── src
           ├──srcnn.cpp
           ├──Srcnn.h
           ├──main.cpp
      ├── build.sh
      ├── CMakeLists.txt
      ├── result                          // 用来存放mxbase推理结果
   ├── sdk                                // sdk推理
      ├──main.py
   ├──docker_start_infer.sh               // 启动容器脚本
   ├──utils
      ├── preprocessdata.py               // 跑mxbase前处理
      ├── cal.py                          // 跑完mxbase计算精度
```

### 准备推理数据

1. 下载数据集

- 进入容器执行以下命令:

- 启动容器。

进入srcnn/infer目录,执行以下命令,启动容器。

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

3. 将模型脚本和模型上传至推理服务器任意目录并解压（如“/home/srcnn”）

```shell
# 在环境上执行
unzip Srcnn_for_MindSpore_{version}_code.zip
cd Srcnn_for_MindSpore_{version}_code/infer && dos2unix `find .`
unzip ../../Srcnnt_for_MindSpore_{version}_model.zip
```

4. 下载数据集Set5,下载链接Set5download url: https://gitee.com/a1085728420/srcnn-dataset

### 模型转换

以下操作均需进入容器中执行。

1. 准备模型文件。  

- srcnn.air

- 将文件放入srcnn/infer/data/model中

2. 模型转换。

    进入“srcnn/infer/convert“目录进行模型转换，转换详细信息可查看转换脚本和对应的aipp配置文件，**在air_to_om.sh**脚本文件中，配置相关参数。

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
bash air_to_om.sh  ../data/model/srcnn.air ../data/model/srcnn
```

**表 3**  参数说明

| 参数      | 说明 |
| ----------- | ----------- |
| input_path     | AIR文件路径。      |
| output_path   |生成的OM文件名，转换脚本会在此基础上添加.om后缀。       |

### mxBase推理

已进入推理容器环境,具体操作请参见“准备容器环境”。

1. (1)修改预处理代码srcnn/infer/utils/preprocessdata.py的推理数据集路径,处理完后数据保存的路径。

```Python
def preprocess():
    imgs = os.listdir('../data/data/Set5')   #推理数据集路径
    imgNum = len(imgs)
    for i in range(imgNum):
        scale = 2
        s = imgs[i]
        target = s.split('.')[0]
        hr = pil_image.open('../data/data/Set5' + "/" + imgs[i]).convert('RGB')   #推理数据集路径
        hr_width = 512
        hr_height = 512
        hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
        lr = hr.resize((hr_width // scale, hr_height // scale), resample=pil_image.BICUBIC)
        lr = lr.resize((lr.width * scale, lr.height * scale), resample=pil_image.BICUBIC)
        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)

        y = convert_rgb_to_y(lr)
        y /= 255.
        y = np.expand_dims(np.expand_dims(y,0),0)
        y.tofile('../data/data/Set' + '/' + target + ".bin")   #处理完后数据保存的路径
```

(2)在srcnn/infer/data/data/下新建空文件夹Set，用于存放预处理后的数据，准备推理数据,进入"srcnn/infer/mxbase"目录下执行以下命令:

```Shell
python3.7 preprocessdata.py
```

执行后在preprocessdata.py设置的路径中得到处理后的数据,如

```Text
|——Set
    |——baby.bin
    |——head.bin
    |——woman.bin
    |——butterfly.bin
    |——bird.bin
```

2. 根据实际情况修改Srcnn.h文件中的推理结果保存路径。

```c
 private:
    std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
    MxBase::ModelDesc modelDesc_ = {};
    uint32_t deviceId_ = 0;
    std::string outputDataPath_ = "./result";   //推理结果保存路径
    std::vector<uint32_t> inputDataShape_ = {1, 1, 512, 512};   //输入图片大小
    uint32_t inputDataSize_ = 262144;
```

3. 在“infer/mxbase”目录下，编译工程

```shell
bash build.sh
```

 运行推理服务。

   在“infer/mxbase”目录下，运行推理程序脚本

```shell
./srcnn [model_path] [input_data_path/] [output_data_path]
e.g.
./srcnn ../data/model/srcnn.om ../data/data/Set/ ./result
```

**表 4** 参数说明：

| 参数             | 说明         |
| ---------------- | ------------ |
| model_path | 模型路径 |
| input_data_path | 处理后的数据路径 |
| output_data_path | 输出推理结果路径 |

编译完成后在mxbase目录下得到以下新文件:

```text
├── mxbase
    ├── build                               // 编译后的文件
    ├── result                              //用于存放推理结果的空文件夹
    ├── srcnn                               // 用于执行的模型文件
```

6. 观察结果。

屏幕会打印推理结果，推理结果示例如下所示。

```Shell
    root@ebee36aeb8e9:/home/data/sdx_mindx/srcnn/infer/mxbase# ./srcnn ../data/model/srcnn.om ../data/data/Set/ ./result
    WARNING: Logging before InitGoogleLogging() is written to STDERR
    I0309 12:53:25.605813 184842 main.cpp:54] =======================================  !!!Parameters setting!!!========================================
    I0309 12:53:25.605866 184842 main.cpp:57] ==========  loading model weights from: ../data/model/srcnn.om
    I0309 12:53:25.605872 184842 main.cpp:60] ==========  input data path = ../data/data/Set/
    I0309 12:53:25.605875 184842 main.cpp:63] ==========  output data path = ./result WARNING: please make sure that this folder is created in advance!!!
    I0309 12:53:25.605878 184842 main.cpp:66] ========================================  !!!Parameters setting!!! ========================================
    I0309 12:53:25.771435 184842 ModelInferenceProcessor.cpp:22] Begin to ModelInferenceProcessor init
    I0309 12:53:25.814025 184842 ModelInferenceProcessor.cpp:69] End to ModelInferenceProcessor init
    I0309 12:53:25.814093 184842 main.cpp:86] Processing: 1/5 ---> bird.bin
    I0309 12:53:25.815188 184842 srcnn.cpp:91] ==========  datasize ---> 1048576
    I0309 12:53:25.841895 184842 main.cpp:86] Processing: 2/5 ---> head.bin
    I0309 12:53:25.842526 184842 srcnn.cpp:91] ==========  datasize ---> 1048576
    I0309 12:53:25.868932 184842 main.cpp:86] Processing: 3/5 ---> woman.bin
    I0309 12:53:25.869599 184842 srcnn.cpp:91] ==========  datasize ---> 1048576
    I0309 12:53:25.895300 184842 main.cpp:86] Processing: 4/5 ---> baby.bin
    I0309 12:53:25.895898 184842 srcnn.cpp:91] ==========  datasize ---> 1048576
    I0309 12:53:25.922716 184842 main.cpp:86] Processing: 5/5 ---> butterfly.bin
    I0309 12:53:25.923260 184842 srcnn.cpp:91] ==========  datasize ---> 1048576
    I0309 12:53:25.948767 184842 main.cpp:95] infer succeed and write the result data with binary file !
    I0309 12:53:26.105157 184842 DeviceManager.cpp:83] DestroyDevices begin
    I0309 12:53:26.105196 184842 DeviceManager.cpp:85] destroy device:0
    I0309 12:53:26.322234 184842 DeviceManager.cpp:91] aclrtDestroyContext successfully!
    I0309 12:53:28.022562 184842 DeviceManager.cpp:99] DestroyDevices successfully
    I0309 12:53:28.022591 184842 main.cpp:102] Infer images sum 5, cost total time: 114.624 ms.
    I0309 12:53:28.022621 184842 main.cpp:103] The throughput: 43.621 bin/sec.
    I0309 12:53:28.022625 184842 main.cpp:104] ==========  The infer result has been saved in ---> ./result

```

 推理结果以bin格式保存。

 ```Text
|——result
    |——baby.bin
    |——head.bin
    |——woman.bin
    |——butterfly.bin
    |——bird.bin
```  

5. 推理精度

- 配置srcnn/infer/utils/cal.py推理数据集路径和推理结果路径,这里的推理数据集为原始的数据集,不需要经过预处理:

```python
def run():
    image_path = '../data/data/Set5'          #推理数据集路径
    result_path = '../mxbase/result'                  #推理结果路径
    if os.path.isdir(image_path):
        img_infos = os.listdir(image_path)
    for i in range(len(img_infos)):
        img_infos[i],ty = os.path.splitext(img_infos[i])
```

- 进入srcnn/infer/mxbase 目录下执行：

```Shell
python3.7 cal.py
```

6. 精度

   PSNR: 42.2986

### MindX SDK推理

已进入推理容器环境。具体操作请参见“准备容器环境”。

1. 准备模型推理文件

    (1)进入srcnn/infer/data/config目录，srcnn.pipeline文件中的"modelPath": "../model/srcnn.om"为om模型所在路径。

```txt
    {
    "srcnn": {
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
                "modelPath": "../data/model/srcnn.om"
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
```

(2) 根据实际情况修改main.py文件中的 **pipeline路径** 、**数据集路径**、**推理结果路径**文件路径,这里的数据集为原始的推理数据集,不需要经过预处理。

```python
def run():
    """
    read pipeline and do infer
    """
    # init stream manager
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    infer_total_time = 0
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        return

    # create streams by pipeline config file
    with open("../data/config/srcnn.pipeline", 'rb') as f:       # pipeline文件路径
        pipelineStr = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipelineStr)

    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        return

    stream_name = b'srcnn'
    image_path = '../data/data/Set5'                             # 推理数据集所在路径
    if os.path.isdir(image_path):
        img_infos = os.listdir(image_path)
    for i in range(len(img_infos)):
        img_infos[i],ty = os.path.splitext(img_infos[i])
    psnr1 = []
    scale = 2
    for i in range(len(img_infos)):
        hr = pil_image.open(image_path + "/" + img_infos[i]+'.png').convert('RGB')
        hr_width = 512
        hr_height = 512
        hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
        lr = hr.resize((hr_width // scale, hr_height // scale), resample=pil_image.BICUBIC)
        lr = lr.resize((lr.width * scale, lr.height * scale), resample=pil_image.BICUBIC)
        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)

        y = convert_rgb_to_y(lr)  
        y /= 255.
        y = np.expand_dims(np.expand_dims(y,0),0)

        y0 = convert_rgb_to_y(hr)
        y0 /= 255.
        y0 = np.expand_dims(np.expand_dims(y0,0),0)
        if not send_source_data(0, y, stream_name, stream_manager_api):
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
        preds = res.reshape(512,512);  
        preds = np.expand_dims(np.expand_dims(preds,0),0)
        psnr = calc_psnr(y0, preds)
        psnr = psnr.item(0)
        psnr1.append(psnr)
        preds.tofile('./result/'+img_infos[i]+'.bin')    # 推理结果保存路径
```

2. 在sdk目录下新建空文件夹result用于存放推理后的结果，运行推理服务,进入srcnn/infer/sdk 目录下执行。

```Shell
python3.7 main.py
```

3. 观察结果。

   PSNR: 42.2986