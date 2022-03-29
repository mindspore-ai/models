# faceboxes MindX推理及mx Base推理

## 源码介绍

```tex
faceboxes_for_mindspore_{version}_code    # 代码示例中不体现具体版本号，用{version}代替
├── infer                # MindX高性能预训练模型新增  
│   └── README.md        # 离线推理文档
│   ├── convert          # 转换om模型命令，AIPP
│   │   ├── faceboxes.aippconfig
│   │   └── atc.sh
│   ├── data             # 包括模型文件、模型输入数据集、模型相关配置文件（如label、SDK的pipeline）
│   │   ├── input
│   │   │   └── ground_truth
│   │   │   ├── images    # WIDER_val
│   │   │   └── val_img_list.txt
│   │   └── config
│   │   │   └── faceboxes.pipeline
│   │   └── model
│   │   │   └── FaceBoxes.air
│   ├── mxbase           # 基于mxbase推理
│   │   ├── src
│   │   │   ├── FaceboxesDetection.cpp
│   │   │   ├── FaceboxesDetection.h
│   │   │   ├── test.jpg  # mxbase用的图片
│   │   │   └── main.cpp
│   │   ├── CMakeLists.txt
│   │   └── build.sh
│   └── sdk                # 基于sdk推理
│       ├── main.py
│       └── eval_result.py # 精度验证
├── README.md                      // descriptions about googlenet
├── scripts
│   ├──run_distribute_train.sh     // shell script for distributed on Ascend
│   ├──run_standalone.sh           // shell script for training standalone on Ascend
│   └──run_eval.sh                 // shell script for evaluation on Ascend
├── src
│   ├──dataset.py                  // creating dataset
│   ├──network.py                  // faceboxes architecture
│   ├──config.py                   // parameter configuration
│   ├──augmentation.py             // data augment method
│   ├──loss.py                     // loss function
│   ├──utils.py                    // data preprocessing
│   └──lr_schedule.py              // learning rate schedule
├── data
│   └── widerface                  // dataset data
│       ├── train
│       │   ├── annotations        // place the dowmloaded training anotations here
│       │   ├── images             // place the training data here
│       │   └── train_img_list.txt # preprocess生成的用于训练的数据集信息
│       └── val
│           ├── ground_truth       // place the dowmloaded eval ground truth label here
│           ├── images             // place the eval data here
│           └── val_img_list.txt   # preprocess生成的用于评估的数据集信息
├── train.py                       // training script
├── eval.py                        // evaluation script
├── eval.py                        // export mindir script
├── preprocess.py                  // generate image list txt file
└── requirements.txt               // other requirements for Faceboxes
```

## 准备推理数据

准备模型转换和模型推理所需目录及数据。

1. 下载所需软件包、模型脚本、模型。

2. 将源码上传至推理服务器任意目录并解压（如：“/home/data/sdx_mindx/faceboxes“)。

   ```tex
   cd /home/data/sdx_mindx/faceboxes/infer && dos2unix `find .`
   ```

3. 进入容器。（根据实际情况修改命令）

   ```tex
   bash docker_start_infer.sh docker_image:tag model_dir
   ```

   **表 2**  参数说明

   | 参数         | 说明                                  |
   | ------------ | ------------------------------------- |
   | docker_image | 推理镜像名称，根据实际写入。          |
   | tag          | 镜像tag，请根据实际配置，如：21.0.2。 |
   | model_dir    | 推理代码路径。                        |

   启动容器时会将推理芯片和数据路径挂载到容器中。可根据需要通过修改**docker\_start\_infer.sh**的device来指定挂载的推理芯片。

   ```tex
   docker run -it \
     --device=/dev/davinci0 \        # 可根据需要修改挂载的npu设备
     --device=/dev/davinci_manager \
   ```

4. 准备数据。

   由于后续推理均在容器中进行，因此需要把用于推理的图片、数据集、模型文件、代码等均放在数据路径中。

   将获取的数据集依次放于下面相应代码目录中，但要删除几张图片，train中的images需要删除4张：**0_Parade_Parade_0_452.jpg、2_Demonstration_Political_Rally_2_444.jpg、39_Ice_Skating_iceskiing_39_380.jpg、46_Jockey_Jockey_46_576.jpg**，val中的images需要删除1张:**6_Funeral_Funeral_6_618.jpg**。

   从前面数据准备中[here](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fzisianw%2FWIDER-to-VOC-annotations)下载的annotations以及从[here](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2FpeteryuX%2Fretinaface-tf2%2Ftree%2Fmaster%2Fwiderface_evaluate%2Fground_truth)下载的ground_truth放入对应代码目录中。

   通过preprocess.py生成需要用于训练和评估的2个txt文件，（**注意：需要完成删除图片的步骤才能进行这一步**），进入到/faceboxes目录执行命令:

   ```bash
   python3.7 preprocess.py
   ```

   ```tex
   ..
   ├── data
   │   └── widerface                  // dataset data
   │       ├── train
   │       │   ├── annotations        // place the dowmloaded training anotations here
   │       │   ├── images             // place the training data here
   │       │   └── train_img_list.txt # preprocess生成的用于训练的数据集信息
   │       └── val
   │           ├── ground_truth       // place the dowmloaded eval ground truth label here
   │           ├── images             // place the eval data here
   │           └── val_img_list.txt   # preprocess生成的用于评估的数据集信息
   ```

   将上面删除好后的train中的images放入下面对应的代码目录中，mxbase/src中的test.jpg是从images中随意挑出的一张，依旧把从前面下载的ground_truth放入对应代码目录中。将上面生成的val_img_list.txt放入相应代码目录中。

   ```tex
   ..
   ├── infer                # MindX高性能预训练模型新增
   │   └── README.md        # 离线推理文档
   │   ├── convert          # 转换om模型命令，AIPP
   │   │   ├── faceboxes.aippconfig
   │   │   └── atc.sh
   │   ├── data             # 包括模型文件、模型输入数据集、模型相关配置文件（如label、SDK的pipeline）
   │   │   ├── input
   │   │   │   └── ground_truth
   │   │   │   ├── images    # WIDER_val
   │   │   │   └── val_img_list.txt
   │   │   └── config
   │   │   │   └── faceboxes.pipeline
   │   │   └── model
   │   │   │   └── FaceBoxes.air  #模型的AIR文件
   │   ├── mxbase           # 基于mxbase推理
   │   │   ├── src
   │   │   │   ├── FaceboxesDetection.cpp
   │   │   │   ├── FaceboxesDetection.h
   │   │   │   ├── test.jpg  # mxbase用的图片
   │   │   │   └── main.cpp
   │   │   ├── CMakeLists.txt
   │   │   └── build.sh
   │   └── sdk        # 基于sdk推理
   │       ├── run.sh
   │       ├── main.py
   │       └── eval_result.py # 精度验证
   ```

   AIR模型可通过“模型训练”后转换生成或通过“下载模型”获取。

## 模型转换

1. 准备模型文件。

   进入infer/data/model目录，将训练所得AIR模型文件上传到该目录下。

2. 模型转换。

   进入“infer/convert“目录进行模型转换，转换详细信息可查看转换脚本和对应的aipp配置文件，**在atc.sh**脚本文件中，配置相关参数

   ```tex
   atc --model=../data/model/FaceBoxes.air \         # 原始模型文件路径与文件名
       --framework=1 \                               # 模型框架类型 1：MindSpore
       --output=../data/model/FaceBoxes \            # 存放转换后的离线模型的路径以及文件名
       --soc_version=Ascend310 \                     # 指定模型转换时的芯片版本
       --input_format=NCHW \                         # 输入数据格式
       --input_shape="images:1,3,2496,1056" \        # 指定模型输入数据的shape
       --output_type=  FP32 \                        # 指定网络输出数据类型或指定某个输出节点的输出类型
       --insert_op_conf=./faceboxes.aippconfig       # 插入算子的配置文件路径与文件名，例如aipp预处理算子
   ```

   转换命令如下。

   ```tex
   bash atc.sh [air_path] [om_path]
   例如：
   bash  atc.sh ../data/model/FaceBoxes.air ../data/model/FaceBoxes
   # 转换后的模型结果为FaceBoxes.om，将放在/home/faceboxes/infer/data/model/目录下
   ```

## mxBase推理

### 前提条件

已进入推理容器环境。具体操作请参见“准备推理数据”。

### 操作步骤

进入infer/mxbase目录。  

1. 编译工程。

   运行命令

   ```tex
   bash build.sh
   ```

   其中，main.cpp文件中，模型路径参数需要根据实际情况修改，此处模型的目录请写具体的目录。

   ```c++
   void InitFaceboxesParam(InitParam &initParam) {
       initParam.deviceId = 0;
       initParam.modelPath = "/home/data/sdx_mindx/FaceBoxes/infer/data/model/FaceBoxes.om";
   }
   ```

   编译生成**FaceboxesDetection**文件至build.sh同目录文件**build**中

2. 运行推理服务。
   运行命令

   ```tex
   cd build/
   ./FaceboxesDetection ../test.jpg
   ```

3. 观察结果。

   推理结果会直接输出在终端。

## MindX SDK推理

### 前提条件

已进入推理容器环境。具体操作请参见“准备推理数据”。

### 操作步骤  

**进入infer/sdk目录。**  

1. 修改配置文件。

   实际情况修改当前目录下的pipeline文件，模型推理路径需要根据实际情况修改。此处模型路径应该填写具体路径，如："../data/model/FaceBoxes.om"。

   ```tex
   {
           "detection": {
               "stream_config": {
                   "deviceId": "0"
               },
               "appsrc0": {
                   "props": {
                       "blocksize": "409600"
                   },
                   "factory": "appsrc",
                   "next": "mxpi_imagedecoder0"
               },
               "mxpi_imagedecoder0": {
                   "props": {
                       "deviceId": "0"
                   },
                   "factory": "mxpi_imagedecoder",
                   "next": "mxpi_imageresize0"
               },
               "mxpi_imageresize0": {
                   "props": {
                       "dataSource": "mxpi_imagedecoder0",
                       "resizeType": "Resizer_KeepAspectRatio_Fit",
                       "resizeHeight": "2496",
                       "resizeWidth": "1056"
                   },
                   "factory": "mxpi_imageresize",
                   "next": "mxpi_tensorinfer0"
               },
               "mxpi_tensorinfer0": {
                   "props": {
                       "dataSource": "mxpi_imageresize0",
                       "modelPath": "../data/model/FaceBoxes.om"
                   },
                   "factory": "mxpi_tensorinfer",
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

2. 性能测试。

   执行以下命令:

   ```tex
   vi /usr/local/sdk_home/mxManufacture/config/sdk.conf
   # 打开性能统计开关。将“enable_ps”参数设置为true，“ps_interval_time”参数设置为6。
   ```

3. 当前infer目录下，运行推理服务。 运行命令:

   ```tex
   bash run.sh    或者   bash run.sh --image-path ../data/input
   ```

   可以看到每张图片的推理结果。

   在完成每张图片的推理后会进行精度评估，可以看到评估结果:

   ```text
   ============== Eval starting ==============
   Easy   Val AP : 0.8560
   Medium Val AP : 0.7795
   Hard   Val AP : 0.4098
   ============== Eval done ==============
   ```

4. 将在日志目录"/usr/local/sdk_home/mxManufacture/logs"查看性能统计结果，输入以下命令:

   ```tex
   cd /usr/local/sdk_home/mxManufacture/logs
   ```

   performance—statistics.log.e2e.xxx

   performance—statistics.log.plugin.xxx

   performance—statistics.log.tpr.xxx

   其中e2e日志统计端到端时间，plugin日志统计单插件时间。可以输入**ll**查看

5. 输入以下命令查看性能:

   ```tex
   cat performance-statistics.log.plugin.20220311-013512.48159 | grep mxpi_tensorinfer0
   ```

   取几组average时间的平均值，sdk推理性能=1000000/平均值（imgs/sec）。

   得到faceboxes推理性能为42 images/sec。

