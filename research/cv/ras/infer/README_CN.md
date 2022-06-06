# RAS MindX推理及mx Base推理

- 脚本说明
    - [脚本及样例代码](https://gitee.com/mindspore/models/tree/master/research/cv/ras/infer#脚本及样例代码)
    - [数据集准备](https://gitee.com/mindspore/models/tree/master/research/cv/ras/infer#数据集准备)
    - [模型转换](https://gitee.com/mindspore/models/tree/master/research/cv/ras/infer#模型转换)
    - [MindX SDK 启动流程](https://gitee.com/mindspore/models/tree/master/research/cv/ras/infer#mindx-sdk-启动流程)
    - [mx Base 推理流程](https://gitee.com/mindspore/models/tree/master/research/cv/ras/infer#mx-base-推理流程)

## 脚本说明

### 脚本及样例代码

```text
├── RAS
  ├── Readme.md
  ├──infer
  │   ├──README_CN.md                       // 推理介绍
  │   ├──requirements.txt                   // 必要依赖
  │   ├──convert
  │      ├──convert.sh                      // air模型转om模型脚本
  │   ├──data
  │      ├──config
  │          ├──ras.pipeline                // pipeline配置文件
  │   ├──mxbase
  │      ├──src
  │          ├──RAS.cpp                     // Mxbase，ras源文件
  │          ├──RAS.h                       // Mxbase，ras头文件
  │          ├──main.cpp                    // Mxbase，主函数源文件
  │      ├──build.sh                        // c++编译启动脚本
  │      ├──CMakeLists.txt                  // CMakeLists配置文件
  │   ├──sdk
  │      ├──main.py                         // sdk推理
  │      ├──run_sdk.sh                      // sdk推理启动脚本
  │   ├──docker_start_infer.sh              // 容器启动脚本
  │   ├──Dockerfile
```

### 数据集准备

此处以 DUTS-TE 原始数据集为例。

1、[DUTS-TE数据集下载](http://saliencydetection.net/duts/download/DUTS-TE.zip)  解压后使用 /DUTS_TE/images 作为测试集。

下载好所需数据集，在“ras/infer/data/”目录下创建“data/”文件夹，将推理数据集存放在“infer/data/data/”路径下。

**示例代码如下：**

```shell
cd /ras/infer/data/
mkdir data/
```

```text
..
├── ras
  ├──infer
  │   ├──README_CN.txt                      // 推理介绍
  │   ├──requirements.txt                   // 必要依赖
  │   ├──convert
  │      ├──convert.sh                      // air模型转om模型脚本
  │   ├──data
  │      ├──data                            // 存放数据集
  │          ├──images                      // 图像
  │          ├──gts                         // 标签
  │      ├──config
  │          ├──ras.pipeline                // pipeline配置文件
  │   ├──mxbase
  │      ├──src
  │          ├──RAS.cpp                     // Mxbase，ras源文件
  │          ├──RAS.h                       // Mxbase，ras头文件
  │          ├──main.cpp                    // Mxbase，主函数源文件
  │      ├──build.sh                        // c++编译启动脚本
  │      ├──CMakeLists.txt                  // CMakeLists配置文件
  │   ├──sdk
  │      ├──main.py                         // sdk推理
  │      ├──run_sdk.sh                      // sdk推理启动脚本
  │   ├──docker_start_infer.sh              // 容器启动脚本
  │   ├──Dockerfile
```

2、MindX SDK 上述数据集准备好后即可执行 “MindX SDK 启动流程” 步骤。

3、mx Base 上述数据集准备好后即可执行 “mx Base 启动流程” 步骤。

### 模型转换

1、将air模型存放在“ras/infer/data/model/”下。

进入“infer/convert/“目录进行模型转换，转换详细信息可查看转换脚本，**在convert.sh**脚本文件中，配置相关参数，尽量使用绝对路径。

```shell
cd ras/infer/convert/
bash convert.sh ../infer/data/model/RAS.air ../infer/data/model/ras
#/{path}/infer/data/model/ras，路径最后的ras是om文件的名称，并不是路径。
```

### MindX SDK 启动流程

1. 修改配置文件。

   a. 修改infer/data/config/pipeline文件。

   ```text
   {
   "ras": {
       "appsrc0": {
           "factory": "appsrc",
           "next": "modelInfer"
           },
       "modelInfer": {
           "props": {
               "modelPath": "../data/model/ras.om",  # 可根据实际情况修改
               "dataSource": "appsrc0"
           },
           "factory": "mxpi_tensorinfer",
           "next": "dataserialize"
           },
       "dataserialize": {
           "props": {
                "outputDataKeys": "modelInfer"
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

2. 修改启动脚本

   在“ras/infer/sdk/”目录下创建“results/”目录，用来存放输出结果日志文件。

   示例代码如下：

   ```shell
   mkdir results
   ```

   根据实际情况修改“run.sh”文件。

   ```shell
   python3 main.py
   exit 0
   ```

3. 运行推理服务。

   a. 若要观测推理性能，需要打开性能统计开关。如下将“enable_ps”参数设置为“true”，“ps_interval_time”参数设置为“6”。

   **vim** /usr/local/sdk_home/mxManufacture/config/sdk.conf

   ```shell
   # MindX SDK configuration file
   # whether to enable performance statistics, default is false [dynamic config]
   enable_ps=true
   ...
   ps_interval_time=6
   ...
   ```

   b. 执行推理。

   在“ras/infer/sdk/”目录下，执行推理命令。

   ```shell
   bash run.sh
   ```

   c. 查看推理性能和精度。

   1. 请确保性能开关已打开，在日志目录“/usr/local/sdk_home/mxManufacture/logs”查看性能统计结果。

      ```shell
      performance—statistics.log.e2e.xxx
      performance—statistics.log.plugin.xxx
      performance—statistics.log.tpr.xxx
      ```

      其中e2e日志统计端到端时间，plugin日志统计单插件时间。
   2. 在“ras/infer/sdk/”目录下。

      执行推理任务后，推理精度会直接显示在界面上。

      ```shell
       ...
      Fmeasure is 0.053
      ---------------  5018 OK ----------------
      -------------- EVALUATION END --------------------
      Average Fmeasure is 0.819
      ```

      可以在"ras/infer/sdk/results/"目录下查看results.txt文件，查看每轮推理精度。

      ```shell
      cat rasults.txt
      ```

### mx Base 推理流程

在容器内用mxBase进行推理。

1. 修改配置文件。

   可根据实际情况修改配置文件“mxbase/src/main.cpp”中的数据集路径，推理路径和推理模型路径。

   ```python
   int main(int argc, char* argv[]) {
       LogInfo << "=======================================  !!!Parameters setting!!!" << \
                  "========================================";
       std::string model_path = argv[1];
       LogInfo << "==========  loading model weights from: " << model_path;
       std::string input_data_path = argv[2];
       LogInfo << "==========  input data path = " << input_data_path;
       std::string output_data_path = argv[3];
       LogInfo << "==========  output data path = " << output_data_path << \
                  " WARNING: please make sure that this folder is created in advance!!!";
   ```

2. 编译工程。

   在“ras/infer/mxbase/”目录下，创建“results/”目录用于存放推理结果，运行编译脚本，编译成功后在此目录下生成build文件夹。

   示例代码如下：

   ```shell
   mkdir results
   bash build.sh
   ```

3. 运行推理服务。

   在“ras/infer/mxbase/build/”路径下运行编译生成的ras文件,还要输入模型，数据集和结果存放的路径。

   示例代码如下：

   ```shell
   ./ras ../../data/model/ras.om ../../data/data/ ../results/
   ```

4. 观察结果。

   执行推理任务后，推理精度会直接显示在界面上。

   ```shell
   ...
   I20220523 01:43:04.816983   490 RAS.cpp:170] ================    Fmeasure Calculate    ===============
   I20220523 01:43:04.816991   490 RAS.cpp:171]  | Average Fmeasure : 0.820369.
   I20220523 01:43:04.816996   490 RAS.cpp:172] =========================================================
   I20220523 01:43:04.817243   490 main.cpp:96] infer succeed and write the result data with binary file !
   I20220523 01:43:04.923189   490 DeviceManager.cpp:98] DestroyDevices begin
   I20220523 01:43:04.923221   490 DeviceManager.cpp:100] destroy device:0
   I20220523 01:43:05.148255   490 DeviceManager.cpp:106] aclrtDestroyContext successfully!
   I20220523 01:43:05.148598   490 DeviceManager.cpp:116] DestroyDevices successfully
   I20220523 01:43:05.148643   490 main.cpp:103] Infer images sum 5019, cost total time: 176307 ms.
   I20220523 01:43:05.148664   490 main.cpp:104] The throughput: 28.4674 bin/sec.
   I20220523 01:43:05.148669   490 main.cpp:105] ==========  The infer result has been saved in ---> ./results.txt
   ```

   可以在"ras/infer/mxbase/results/"目录下查看results.txt文件，查看每轮推理精度。

   ```shell
   cat results.txt
   ```
