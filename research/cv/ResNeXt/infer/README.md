# 离线推理过程

## MindX SDK推理

1. 准备容器环境。

- 准备SDK安装包和Dockerfile文件(resnext50 /Dockerfile），将其一同上传至服务器任意目录（如/home/data）
  Dockerfile文件内容：

  ~~~ Dockerfile
  ARG FROM_IMAGE_NAME
  #基础镜像
  FROM $FROM_IMAGE_NAME
  ARG SDK_PKG
  #将SDK安装包拷贝到镜像中
  COPY ./$SDK_PKG .
  ~~~

- 下载MindX SDK开发套件（mxManufacture）。

  将下载的软件包上传到代码目录“/home/data/resnext50/infer”下并执行安装命令：./Ascend-mindxsdk-mxManufacture_xxx_linux-x86_64.run --install --install-path=安装路径（安装路径为源代码存放路径，如第一点所说的/home/data/infer）。

- 编译推理镜像：

  ```shell
  #非root权限，需在指令前面加"sudo"
  docker build -t infer_image --build-arg FROM_IMAGE_NAME=base_image:tag --build-arg SDK_PKG=sdk_pkg .
  ```

  | 参数          | 说明                                                         |
  | ------------- | ------------------------------------------------------------ |
  | *infer_image* | 推理镜像名称，根据实际写入。                                 |
  | *base_image*  | 基础镜像，可从Ascend Hub上下载,如ascendhub.huawei.com/public-ascendhub/ascend-infer-x86。 |
  | *tag*         | 镜像tag，请根据实际配置，如：21.0.2。                        |
  | sdk_pkg       | 下载的mxManufacture包名称，如Ascend-mindxsdk-mxmanufacture_*{version}*_linux-*{arch}*.run |
  注：指令末尾的”.“一定不能省略，这代表当前目录。

- 执行以下命令，启动容器实例：

  ```shell
  bash docker_start_infer.sh docker_image:tag model_dir
  ```

  | 参数           | 说明                                         |
  | -------------- | -------------------------------------------- |
  | *docker_image* | 推理镜像名称，推理镜像请从Ascend Hub上下载。 |
  | *tag*          | 镜像tag，请根据实际配置，如：21.0.2。        |
  | *model_dir*    | 推理容器挂载路径，本例中为/home/data         |
  启动容器时会将推理芯片和数据路径挂载到容器中。
  其中docker_start_infer.sh的内容如下：

  ```shell
  #!/bin/bash
  docker_image=$1
  model_dir=$2
  if [ -z "${docker_image}" ]; then
      echo "please input docker_image"
      exit 1
  fi
  if [ ! -d "${model_dir}" ]; then
      echo "please input model_dir"
      exit 1
  fi
  docker run -it \
             --device=/dev/davinci0 \        #请根据芯片的情况更改
             --device=/dev/davinci_manager \
             --device=/dev/devmm_svm \
             --device=/dev/hisi_hdc \
             -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
             -v ${model_dir}:${model_dir} \
             ${docker_image} \
             /bin/bash
  ```

2. 将源代码(resnext50文件夹)上传至服务器任意目录（如：/home/data/，后续示例将以/home/data/resnext50为例）,并进入该目录。
    源码目录结构如下图所示：

```text
/home/data/resnext50
├── infer                # MindX高性能预训练模型新增  
│   └── README.md        # 离线推理文档
│   ├── convert          # 转换om模型命令，AIPP
│   │   ├──aipp.config
│   │   └──atc.sh
│   ├── data             # 包括模型文件、模型输入数据集、模型相关配置文件（如label、SDK的pipeline）
│   │   ├── model
│   │   ├── input
│   │   └── config
│   │   │   ├──imagenet1000_clsidx_to_labels.names
│   │   │   ├──resnext50.cfg
│   │   │   └──resnext50.pipeline
│   ├── mxbase           # 基于mxbase推理
│   │   ├── build
│   │   ├── src
│   │   │   ├── resnext50Classify.cpp
│   │   │   ├── resnext50Classify.h
│   │   │   ├── main.cpp
│   │   │   └── include   #包含运行所需库
│   │   ├── CMakeLists.txt
│   │   └── build.sh
│   └── sdk              # 基于sdk run包推理；如果是C++实现，存放路径一样
│   │   ├── main.py
│   │   └── run.sh
│   └── util             # 精度验证脚本
│   │   ├──imagenet2012_val.txt
│   │   └──task_metric.py
```

3. 将air模型放入resnext50/infer/data/model目录下。

4. 准备AIPP配置文件。

AIPP需要配置aipp.config文件，在ATC转换的过程中插入AIPP算子，即可与DVPP处理后的数据无缝对接，AIPP参数配置请参见《[CANN 开发辅助工具指南 (推理)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》中“ATC工具使用指南”。
aipp.config文件内容如下，该文件放在resnext50/infer/convert目录下：

~~~ config
aipp_op{
    aipp_mode: static
    input_format : RGB888_U8
    rbuv_swap_switch : true
    mean_chn_0 : 0
    mean_chn_1 : 0
    mean_chn_2 : 0
    min_chn_0 : 123.675
    min_chn_1 : 116.28
    min_chn_2 : 103.53
    var_reci_chn_0: 0.0171247538316637
    var_reci_chn_1: 0.0175070028011204
    var_reci_chn_2: 0.0174291938997821
}
~~~

5. 进入resnext50/infer/convert目录，并执行：

```shell
#本例中模型名称为resnext50.air  
bash atc.sh ../data/model/resnext50.air
```

利用ATC工具将air模型转换为om模型，om模型会自动放在resnext50/infer/data/model/文件夹下。
其中atc.sh的内容如下：

~~~ sh
model=$1
atc \
  --model=$model \
  --framework=1 \
  --output=../data/model/resnext50 \
  --output_type=FP32 \
  --soc_version=Ascend310 \
  --input_shape="input:1,224,224,3" \
  --log=info \
  --insert_op_conf=aipp.config
~~~

参数说明：

- --model：待转换的air模型所在路径。
- --framework：1代表MindSpore框架。
- --output：转换后输出的om模型存放路径以及名称。
- --soc_version：生成的模型支持的推理环境。
- --input_shape：输入数据的shape。
- --log=info：打印转换过程中info级别的日志。
- --insert_op_conf：模型转换使用的AIPP配置文件。

6. 数据准备。
    将推离图片数据集放在resnext50/infer/data/input目录下。

- 推理使用的数据集是[ImageNet2012](http://www.image-net.org/)中的验证集，input目录下已有其中十张测试图片。
- 测试集：6.4 GB，50, 000张图像。

7. 准备模型推理所需文件。

    1. 在“/home/data/resnext50/infer/data/config/”目录下编写pipeline文件。
    根据实际情况修改resnext50.pipeline文件中图片规格、模型路径、配置文件路径和标签路径。
    更多介绍请参见《[mxManufacture 用户指南](https://ascend.huawei.com/#/software/mindx-sdk/sdk-detail)》中“基础开发”章节。
    resnext50.pipeline

```pipeline
{
  "im_resnext50": {
    "stream_config": {
      "deviceId": "0"
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
        "resizeHeight": "256",
        "resizeWidth": "256",
        "resizeType": "Resizer_Stretch"
      },
      "factory": "mxpi_imageresize",
      "next": "mxpi_imagecrop0:1"
    },
    "mxpi_imagecrop0": {
      "props": {
        "dataSource": "appsrc1",
        "dataSourceImage": "mxpi_imageresize0",
        "handleMethod": "opencv"
      },
      "factory": "mxpi_imagecrop",
      "next": "mxpi_tensorinfer0"
    },
    "mxpi_tensorinfer0": {
      "props": {
        "dataSource": "mxpi_imagecrop0",
        "modelPath": "../data/model/resnext50.om",
        "waitingTime": "1",
        "outputDeviceId": "-1"
      },
      "factory": "mxpi_tensorinfer",
      "next": "mxpi_classpostprocessor0"
    },
    "mxpi_classpostprocessor0": {
      "props": {
        "dataSource": "mxpi_tensorinfer0",
        "postProcessConfigPath": "../data/config/resnext50.cfg",
        "labelPath": "../data/config/imagenet1000_clsidx_to_labels.names",
        "postProcessLibPath": "/home/data/zjut_mindx/SDK_2.0.2/mxManufacture/lib/modelpostprocessors/libresnet50postprocess.so"
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
    "appsrc1": {
      "props": {
        "blocksize": "409600"
      },
      "factory": "appsrc",
      "next": "mxpi_imagecrop0:0"
    },
    "appsrc0": {
      "props": {
        "blocksize": "409600"
      },
      "factory": "appsrc",
      "next": "mxpi_imagedecoder0"
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

参数说明：

- resizeHeight：图片缩放后高度，请根据实际需求尺寸输入。
- resizeWidth：图片缩放后宽度，请根据实际需求尺寸输入。
- modelPath：模型路径，请根据模型实际路径修改。
- postProcessConfigPath：模型配置文件路径，请根据模型配置文件的实际路径修改。
- labelPath：标签文件路径，请根据标签文件的实际路径修改。

    2. 在“/home/data/resnext50/infer/data/config/”目录下编写resnext50.cfg配置文件。
  配置文件resnext50.cfg内容如下：

```python
CLASS_NUM=1000
SOFTMAX=false
TOP_K=5
```

8. 根据实际情况修改main.py文件中裁剪图片的位置和**pipeline**文件路径。

其中main.py的内容如下：

```python
...
 def _predict_gen_protobuf(self):
        object_list = MxpiDataType.MxpiObjectList()
        object_vec = object_list.objectVec.add()
        object_vec.x0 = 16
        object_vec.y0 = 16
        object_vec.x1 = 240
        object_vec.y1 = 240
...
 def main():
    pipeline_conf = "../data/config/resnext50.pipeline"
    stream_name = b'im_resnext50'

    args = parse_args()
    result_fname = get_file_name(args.result_file)
    pred_result_file = f"{result_fname}.txt"
    dataset = GlobDataLoader(args.glob, limit=None)
    with ExitStack() as stack:
        predictor = stack.enter_context(Predictor(pipeline_conf, stream_name))
        result_fd = stack.enter_context(open(pred_result_file, 'w'))

        for fname, pred_result in predictor.predict(dataset):
            result_fd.write(result_encode(fname, pred_result))

    print(f"success, result in {pred_result_file}")
...
```

9. 进入resnext50/infer/sdk文件夹，执行bash run.sh，推理结果保存在当前目录下的resnext50_sdk_pred_result.txt文件中。

其中run.sh的内容如下：

~~~ bash
set -e
CUR_PATH=$(cd "$(dirname "$0")" || { warn "Failed to check path/to/run.sh" ; exit ; } ; pwd)
# Simple log helper functions
info() { echo -e "\033[1;34m[INFO ][MxStream] $1\033[1;37m" ; }
warn() { echo >&2 -e "\033[1;31m[WARN ][MxStream] $1\033[1;37m" ; }
export LD_LIBRARY_PATH=${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:${MX_SDK_HOME}/opensource/lib64:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:${LD_LIBRARY_PATH}
export GST_PLUGIN_SCANNER=${MX_SDK_HOME}/opensource/libexec/gstreamer-1.0/gst-plugin-scanner
export GST_PLUGIN_PATH=${MX_SDK_HOME}/opensource/lib/gstreamer-1.0:${MX_SDK_HOME}/lib/plugins
#to set PYTHONPATH, import the StreamManagerApi.py
export PYTHONPATH=$PYTHONPATH:${MX_SDK_HOME}/python
python3.7 main.py "../data/input/*.JPEG" resnext50_sdk_pred_result.txt
exit 0
~~~

推理结果：

```shell
ILSVRC2012_val_00000008 415,928,850,968,911
ILSVRC2012_val_00000001 65,62,58,54,56
ILSVRC2012_val_00000009 674,338,333,106,337
ILSVRC2012_val_00000007 334,361,375,7,8
ILSVRC2012_val_00000010 332,338,153,204,190
ILSVRC2012_val_00000005 520,516,431,797,564
ILSVRC2012_val_00000006 62,60,57,67,65
ILSVRC2012_val_00000003 230,231,226,169,193
ILSVRC2012_val_00000002 795,970,537,796,672
ILSVRC2012_val_00000004 809,968,969,504,967
......
```

10. 验证精度，进入resnext50/infer/util目录下并执行：

```shell
python3.7 task_metric.py ../sdk/resnext50_sdk_pred_result.txt imagenet2012_val.txt resnext50_sdk_pred_result_acc.json 5
```

参数说明：

- 第一个参数（../sdk/resnext50_sdk_pred_result.txt）：推理结果保存路径。
- 第二个参数（imagenet2012_val.txt）：验证集标签文件。
- 第三个参数（resnext50_sdk_pred_result_acc.json）：结果文件
- 第四个参数（5）："1"表示TOP-1准确率，“5”表示TOP-5准确率。

11. 查看推理性能和精度

- 打开性能统计开关。将“enable_ps”参数设置为true，“ps_interval_time”参数设置为6。

  **vi /usr/local/sdk_home/mxManufacture/config/sdk.conf**

  ```bash
  # MindX SDK configuration file

  # whether to enable performance statistics, default is false [dynamic config]
  enable_ps=true
  ...
  ps_interval_time=6
  ...
  ```

- 执行run.sh脚本  **bash run.sh**

- 在日志目录"/usr/local/sdk_home/mxManufacture/logs"查看性能统计结果。

  ```bash
  performance--statistics.log.e2e.xxx
  performance--statistics.log.plugin.xxx
  performance--statistics.log.tpr.xxx
  ```

  其中e2e日志统计端到端时间，plugin日志统计单插件时间。

查看推理精度

~~~ bash
cat resnext50_sdk_pred_result_acc.json
~~~

top-5推理精度:

```json
  "total": 50000,
  "accuracy": [
    0.78386,
    0.8793,
    0.91304,
    0.92962,
    0.94016
  ]
```

## mxBase推理

1. 准备容器环境。

- 准备SDK安装包和Dockerfile文件(resnext50/Dockerfile），将其一同上传至服务器任意目录（如/home/data）
  Dockerfile文件内容：

  ```dockerfile
  ARG FROM_IMAGE_NAME
  #基础镜像
  FROM $FROM_IMAGE_NAME
  ARG SDK_PKG
  #将SDK安装包拷贝到镜像中
  COPY ./$SDK_PKG .
  ```

- 下载MindX SDK开发套件（mxManufacture）。

  将下载的软件包上传到代码目录“/home/data/resnext50/infer”下并执行安装命令：./Ascend-mindxsdk-mxManufacture_xxx_linux-x86_64.run --install --install-path=安装路径（安装路径为源代码存放路径，如第一点所说的/home/data/infer）。

- 编译推理镜像。

  ```shell
  #非root权限，需在指令前面加"sudo"
  docker build -t  infer_image  --build-arg FROM_IMAGE_NAME= base_image:tag  --build-arg SDK_PKG= sdk_pkg  .
  ```

  | 参数          | 说明                                                         |
  | ------------- | ------------------------------------------------------------ |
  | *infer_image* | 推理镜像名称，根据实际写入。                                 |
  | *base_image*  | 基础镜像，可从Ascend Hub上下载。                             |
  | *tag*         | 镜像tag，请根据实际配置，如：21.0.2。                        |
  | sdk_pkg       | 下载的mxManufacture包名称，如Ascend-mindxsdk-mxmanufacture_*{version}*_linux-*{arch}*.run |

- 执行以下命令，启动容器实例。

  ```shell
  bash docker_start_infer.sh docker_image:tag model_dir
  ```

  | 参数           | 说明                                         |
  | -------------- | -------------------------------------------- |
  | *docker_image* | 推理镜像名称，推理镜像请从Ascend Hub上下载。 |
  | *tag*          | 镜像tag，请根据实际配置，如：21.0.2。        |
  | *model_dir*    | 推理容器挂载路径，本例中为/home/data         |
  启动容器时会将推理芯片和数据路径挂载到容器中。
  其中docker_start_infer.sh的内容如下：

  ```shell
  #!/bin/bash
  docker_image=$1
  model_dir=$2
  if [ -z "${docker_image}" ]; then
      echo "please input docker_image"
      exit 1
  fi
  if [ ! -d "${model_dir}" ]; then
      echo "please input model_dir"
      exit 1
  fi
  docker run -it \
             --device=/dev/davinci0 \
             --device=/dev/davinci_manager \
             --device=/dev/devmm_svm \
             --device=/dev/hisi_hdc \
             -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
             -v ${model_dir}:${model_dir} \
             ${docker_image} \
             /bin/bash
  ```

2. 将源代码(resnext50文件夹)上传至服务器任意目录（如：/home/data/，后续示例将以/home/data/resnext50为例）,并进入该目录。
    源码目录结构如下图所示：

```text
/home/data/resnext50
├── infer                # MindX高性能预训练模型新增  
│   └── README.md        # 离线推理文档
│   ├── convert          # 转换om模型命令，AIPP
│   │   ├──aipp.config
│   │   └──atc.sh
│   ├── data             # 包括模型文件、模型输入数据集、模型相关配置文件（如label、SDK的pipeline）
│   │   ├── model
│   │   ├── input
│   │   └── config
│   │   │   ├──imagenet1000_clsidx_to_labels.names
│   │   │   ├──resnext50.cfg
│   │   │   └──resnext50.pipeline
│   ├── mxbase           # 基于mxbase推理
│   │   ├── build
│   │   ├── src
│   │   │   ├── resnext50Classify.cpp
│   │   │   ├── resnext50Classify.h
│   │   │   ├── main.cpp
│   │   │   └── include   #包含运行所需库
│   │   ├── CMakeLists.txt
│   │   └── build.sh
│   └── sdk              # 基于sdk run包推理；如果是C++实现，存放路径一样
│   │   ├── main.py
│   │   └── run.sh
│   └── util             # 精度验证脚本
│   │   ├──imagenet2012_val.txt
│   │   └──task_metric.py
```

3. 添加环境变量。

通过vi ~/.bashrc命令打开~/.bashrc文件，将下面的环境变量添加进当前环境，添加好环境变量以后退出文件编辑，执行source ~/.bashrc使环境变量生效。

```bash
export ASCEND_HOME="/usr/local/Ascend"
export ASCEND_VERSION="nnrt/latest"
export ARCH_PATTERN="."
export LD_LIBRARY_PATH="${MX_SDK_HOME}/lib/modelpostprocessors:${LD_LIBRARY_PATH}"
export MXSDK_OPENSOURCE_DIR="${MX_SDK_HOME}/opensource"
```

4. 进行模型转换，详细步骤见上一章MindX SDK推理。
5. 进入resnext50/infer/mxbase目录，并执行:

```shell
bash build.sh
```

其中build.sh的内容如下：

```sh
path_cur=$(dirname $0)
function check_env()
{
    # set ASCEND_VERSION to ascend-toolkit/latest when it was not specified by user
    if [ ! "${ASCEND_VERSION}" ]; then
        export ASCEND_VERSION=ascend-toolkit/latest
        echo "Set ASCEND_VERSION to the default value: ${ASCEND_VERSION}"
    else
        echo "ASCEND_VERSION is set to ${ASCEND_VERSION} by user"
    fi

    if [ ! "${ARCH_PATTERN}" ]; then
        # set ARCH_PATTERN to ./ when it was not specified by user
        export ARCH_PATTERN=./
        echo "ARCH_PATTERN is set to the default value: ${ARCH_PATTERN}"
    else
        echo "ARCH_PATTERN is set to ${ARCH_PATTERN} by user"
    fi
}
function build_resnext50()
{
    cd $path_cur
    rm -rf build
    mkdir -p build
    cd build
    cmake ..
    make
    ret=$?
    if [ ${ret} -ne 0 ]; then
        echo "Failed to build resnext50."
        exit ${ret}
    fi
    make install
}
check_env
build_resnext50
```

6. 在当前目录下执行。

```shell
#假设图片数量为10，推理结果保存在当前目录下的mx_pred_result.txt文件下
./resnext ../data/input 5000
```

- 第一个参数（../data/input）：图片输入路径。
- 第二个参数（50000）：图片输入数量。

mx_pred_result.txt（推理结果）:

```shell
ILSVRC2012_val_00047110 221,267,220,206,266,
ILSVRC2012_val_00014552 550,505,503,899,804,
ILSVRC2012_val_00006604 276,275,287,286,212,
ILSVRC2012_val_00016859 2,3,148,5,19,
ILSVRC2012_val_00020009 336,649,972,299,356,
ILSVRC2012_val_00025515 917,921,454,446,620,
ILSVRC2012_val_00046794 541,632,412,822,686,
ILSVRC2012_val_00035447 856,866,595,586,864,
ILSVRC2012_val_00016392 54,68,56,60,66,
ILSVRC2012_val_00023902 50,44,60,61,54,
ILSVRC2012_val_00000719 268,151,171,237,285,
......
```

7. 验证精度，进入resnext50/infer/util目录下，并执行：

```python
python3.7 task_metric.py ../mxbase/mx_pred_result.txt imagenet2012_val.txt resnext50_mxbase_pred_result_acc.json 5
```

参数说明：

- 第一个参数（../mxbase/mx_pred_result.txt）：推理结果保存路径。
- 第二个参数（imagenet2012_val.txt）：验证集标签文件。
- 第三个参数（resnext50_mxbase_pred_result_acc.json）：结果文件
- 第四个参数（5）："1"表示TOP-1准确率，“5”表示TOP-5准确率。

8. 查看推理精度结果

~~~ bash
cat resnext50_mxbase_pred_result_acc.json
~~~

top-5推理精度:

```json
"accuracy": [
    0.78386,
    0.8793,
    0.91304,
    0.92962,
    0.94016
  ]
```