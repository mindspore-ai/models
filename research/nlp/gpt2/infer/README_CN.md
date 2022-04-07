# GPT-2 MindX推理及mx Base推理

- 脚本说明
    - [脚本及样例代码](https://gitee.com/mindspore/models/tree/master/research/cv/r2plus1d/infer#脚本及样例代码)
    - [数据集准备](https://gitee.com/mindspore/models/tree/master/research/cv/r2plus1d/infer#数据集准备)
    - [模型转换](https://gitee.com/mindspore/models/tree/master/research/cv/r2plus1d/infer#模型转换)
    - [MindX SDK 启动流程](https://gitee.com/mindspore/models/tree/master/research/cv/r2plus1d/infer#mindx-sdk-启动流程)
    - [mx Base 推理流程](https://gitee.com/mindspore/models/tree/master/research/cv/r2plus1d/infer#mx-base-推理流程)

## 脚本说明

### 脚本及样例代码

```text
├── GPT-2
    ├── README_CN.md                             // GPT-2 的说明文件
    ├──infer
      |   ├── README_CN.md                       // GPT-2 的 MindX SDK 推理及 mx Base 推理的说明文件
      │   ├──convert
      │      ├──convert.sh                       // 将 air 模型转换为 om 模型的脚本
      │   ├──data
      │      ├──config
      │          ├──gpt2.pipeline                // MindX SDK运行所需的 pipline 配置文件
      │   ├──mxbase
      │      ├──src
      │          ├──gpt2.cpp                     // Mxbase，gpt2源文件
      │          ├──gpt2.h                       // Mxbase，gpt2头文件
      │          ├──main.cpp                     // Mxbase，主函数源文件
      │      ├──build.sh                         // c++编译启动脚本
      │      ├──CMakeLists.txt                   // CMakeLists配置文件
      │   ├──sdk
      │      ├──main.py                          // 语言建模任务sdk推理
      │      ├──run_sdk.sh                       // 语言建模任务sdk推理启动脚本
      │   ├──utils
      │      ├──data_processor_seq.py            // 推理数据集预处理
      │   ├──docker_start_infer.sh               // 容器启动脚本
      ├── ......                                 // 其他代码文件
```

### 数据集准备

此处以 PTB 原始数据集为例。

1、[PTB数据集下载](https://gitee.com/link?target=http%3A%2F%2Fwww.fit.vutbr.cz%2F~imikolov%2Frnnlm%2Fsimple-examples.tgz) 解压后使用 `/simple-examples/data/ptb.test.txt` 测试集，使用 `/simple-examples/data/ptb.train.txt` 作为训练集。

由于后续推理均在容器中进行，将下载好的数据集放在./GPT-2/datasets目录内，后续示例将以“/home/HwHiAiUser“为例。

```text
..
├── GPT-2
│  ├──datasets                     # 存放下载的数据集
│     └──simple-examples
│         └──data
│             └──ptb.test.txt
│             ├──ptb.train.txt
│  ├── infer                       # MindX高性能预训练模型新增
│     └── README.md                # 离线推理文档
│     ├── convert                  # 转换om模型命令
│     │   ├──convert.sh
│     ├── data                     # 包括模型文件、模型输入数据集、模型相关配置文件（如SDK的pipeline）
│     │   ├── data                 # 推理所需输入文件
│     │   ├── model                # om模型文件
│     │   └── config               # 推理所需的配置文件
│     ├── mxbase                   # 基于mxbase推理
│     └── sdk                      # 基于sdk.run包推理
│     └── utils
│     │   └──data_processor_seq.py # 数据集预处理脚本
│     └──docker_start_infer.sh     # 启动容器脚本
```

在“GPT-2/infer/data/”目录下创建“data/”文件夹，在路径“GPT-2/infer/utils/”路径下，运行data_processor_seq.py文件，生成推理所需数据集，将推理数据集存放在“infer/data/data/”路径下，尽量使用绝对路径。

示例代码如下：

```shell
 python3 data_processor_seq.py --input_file="../../datasets/simple-examples/ptb.test.txt" --output_file="../data/data/" --num_splits=1 --max_length=1024 --dataset="ptb" --vocab_file="../../src/utils/pretrain-data/gpt2-vocab.json" --merge_file="../../src/utils/pretrain-data/gpt2-merges.txt"
```

2、MindX SDK 使用预处理后的数据集，上述数据集准备好后即可执行 “MindX SDK 启动流程” 步骤。

3、mx Base 使用预处理后的数据集，上述数据集准备好后即可执行 “mx Base 启动流程” 步骤。

### 模型转换

1、在"GPT-2/infer/data/"目录下创建“model/”目录，将准备好的ckpt文件存放在“model/”路径下，在"GPT-2/"目录下运行export.py文件，将训练好的ckpt模型冻结为air模型。

```shell
mkdir infer/data/model/
python export.py
--load_ckpt_path="/{path}/gpt2_language_model_small_Lamb_1_bs1-1_42068.ckpt"
--save_air_path="/{path}/"
```

2、将air模型存放在“GPT-2/infer/data/model/”下。

进入“infer/convert/“目录进行模型转换，转换详细信息可查看转换脚本，**在convert.sh**脚本文件中，配置相关参数，尽量使用绝对路径。

```shell
bash convert.sh /{path}/gpt2.air /{path}/infer/data/model/gpt2
#/{path}/infer/data/model/gpt2，路径最后的gpt2是om文件的名称，并不是路径。
```

### MindX SDK 启动流程

1. 修改配置文件。

   a. 修改infer/data/config/pipeline文件。

   ```text
   {
       "im_gpt2": {
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
           "appsrc2": {
               "props": {
                   "blocksize": "409600"
                },
                "factory": "appsrc",
                "next": "mxpi_tensorinfer0:2"
           },
           "mxpi_tensorinfer0": {
               "props": {
                   "dataSource": "appsrc0,appsrc1,appsrc2",
                   "modelPath": "../data/model/gpt2.om"  # 根据实际情况修改配置推理模型路径
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
               "factory": "appsink"
           }
       }
   }
   ```

2. 修改启动脚本

   在“GPT-2/infer/sdk/”目录下创建“results/”目录，用来存放输出结果日志文件。

   示例代码如下：

   ```shell
   mkdir results
   ```

   根据实际情况修改“run_sdk.sh”文件中的pipeline文件和数据集，日志文件路径。

   ```shell
   set -e

   # Simple log helper functions
   info() { echo -e "\033[1;34m[INFO ][MxStream] $1\033[1;37m" ; }
   warn() { echo >&2 -e "\033[1;31m[WARN ][MxStream] $1\033[1;37m" ; }

   #to set PYTHONPATH, import the StreamManagerApi.py
   export PYTHONPATH=$PYTHONPATH:${MX_SDK_HOME}/python

   python3 main.py --data_dir="../data/data/" --pipeline_path="../data/config/gpt2.pipeline" --logs_dir="./results/"
    # 根据实际情况修改配置pipeline文件和数据集路径。
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

   在“GPT-2/infer/sdk/”目录下，执行推理命令。

   ```shell
   bash run_sdk.sh
   ```

   c. 查看推理性能和精度。

   1. 请确保性能开关已打开，在日志目录“/usr/local/sdk_home/mxManufacture/logs”查看性能统计结果。

      ```shell
      performance—statistics.log.e2e.xxx
      performance—statistics.log.plugin.xxx
      performance—statistics.log.tpr.xxx
      ```

      其中e2e日志统计端到端时间，plugin日志统计单插件时间。

   2. 在“GPT-2/infer/sdk/”目录下。

      执行推理任务后，推理精度会直接显示在界面上。

      ```shell
       ...
       | Current Loss: 3.746609
       | Current PLL: 42.377156251052035
       | Average Loss: 3.308676
       | Average PLL: 27.348904393471983
      ```

      可以在"GPT-2/infer/sdk/results/"目录下查看result.txt文件，查看每轮推理精度。

      ```shell
      cat score.txt
      ```

      ##

### mx Base 推理流程

在容器内用mxBase进行推理。

1. 修改配置文件。

   可根据实际情况修改配置文件“mxbase/src/main.cpp”中的数据集路径，推理路径和推理模型路径。

   ```python
   int main(int argc, char* argv[]) {
       std::string dataPath = "../data/data/";          # 根据实际情况修改推理模型路径
       std::string inferPath = "../mxbase/";            # 根据实际情况修改推理模型路径

       InitParam initParam = {};
       initParam.deviceId = 0;
       initParam.modelPath = "../data/model/gpt2.om";   # 根据实际情况修改推理模型路径

       auto model_gpt2 = std::make_shared<gpt2>();
       APP_ERROR ret = model_gpt2->Init(initParam);
       if (ret != APP_ERR_OK) {
           LogError << "gpt2 init failed, ret=" << ret << ".";
           return ret;
       }
   ```

2. 编译工程。

   在“GPT-2/infer/mxbase/”目录下，创建“results/”目录用于存放推理结果，运行编译脚本，编译成功后在此目录下生成build文件夹。

   示例代码如下：

   ```shell
   mkdir results
   bash build.sh
   ```

3. 运行推理服务。

   在“GPT-2/infer/mxbase/”路径下运行编译生成的gpt2文件。

   示例代码如下：

   ```shell
   ./build/gpt2
   ```

4. 观察结果。

   执行推理任务后，推理精度会直接显示在界面上。

   ```shell
   ...
   I0318 05:08:32.299561 23973 gpt2.cpp:188] ==============================================================
   I0318 05:08:32.299571 23973 gpt2.cpp:189] infer result of ../Mxbase/results/ is:
   I0318 05:08:32.299608 23973 gpt2.cpp:196]  | Current Loss : 3.74661.
   I0318 05:08:32.299614 23973 gpt2.cpp:197]  | Current PPL : 42.3772.
   I0318 05:08:32.299623 23973 gpt2.cpp:198] ==============================================================
   I0318 05:08:33.296967 23973 DeviceManager.cpp:83] DestroyDevices begin
   I0318 05:08:33.297009 23973 DeviceManager.cpp:85] destroy device:0
   I0318 05:08:33.572667 23973 DeviceManager.cpp:91] aclrtDestroyContext successfully!
   I0318 05:08:34.977069 23973 DeviceManager.cpp:99] DestroyDevices successfully
   I0318 05:08:34.977207 23973 main.cpp:88] ================    PPL Calculate    ===============
   I0318 05:08:34.977217 23973 main.cpp:89]  | Average Loss : 3.30868.
   I0318 05:08:34.977243 23973 main.cpp:90]  | Average PPL : 27.3489.
   I0318 05:08:34.977254 23973 main.cpp:91] ====================================================
   ```

   可以在"GPT-2/infer/mxbase/results/"目录下查看result.txt文件，查看每轮推理精度。

   ```shell
   cat score.txt
   ```
