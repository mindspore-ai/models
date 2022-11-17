# 离线推理过程

## 准备容器环境

1、将源代码(vgg16_mindspore_1.3.0_code)上传至服务器任意目录（如：/home/data/）,并进入该目录。

源码目录结构如下图所示：

```bash
/home/data/vgg16_mindspore_1.3.0_code
├── infer                # MindX高性能预训练模型新增  
│   └── README.md        # 离线推理文档
│   ├── convert          # 转换om模型命令，AIPP
│   │   ├──aipp_vgg16_rgb.config
│   │   └──atc.sh
│   ├── data             # 包括模型文件、模型输入数据集、模型相关配置文件（如label、SDK的pipeline）
│   │   ├── input
│   │   │   └──ILSVRC2012_val_00000001.JPEG
│   │   └── config
│   │   │   ├──vgg16.cfg
│   │   │   └──vgg16.pipeline
│   ├── mxbase           # 基于mxbase推理
│   │   ├── src
│   │   │   ├── Vgg16Classify.cpp
│   │   │   ├── Vgg16Classify.h
│   │   │   └── main.cpp
│   │   ├── CMakeLists.txt
│   │   └── build.sh
│   └── sdk               # 基于sdk.run包推理；如果是C++实现，存放路径一样
│   │   ├── main.py
│   │   └── run.sh
│   └── util              # 精度验证脚本
│   │   └──task_metric.py
│   ├──Dockerfile        #容器文件
│   └──docker_start_infer.sh     # 启动容器脚本
```

2、启动容器
执行以下命令，启动容器实例。

```bash
bash docker_start_infer.sh docker_image:tag model_dir
```

| 参数           | 说明                                         |
| -------------- | -------------------------------------------- |
| *docker_image* | 推理镜像名称，推理镜像请从Ascend Hub上下载。 |
| *tag*          | 镜像tag，请根据实际配置，如：21.0.2。        |
| *model_dir*    | 推理容器挂载路径，本例中为/home/data         |

启动容器时会将推理芯片和数据路径挂载到容器中。
其中docker_start_infer.sh(vgg16_mindspore_1.3.0_code/infer/docker_start_infer.sh)内容如下。
docker_start_infer.sh文件内容

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
           --device=/dev/davinci0 \  #请根据芯片的情况更改
           --device=/dev/davinci_manager \
           --device=/dev/devmm_svm \
           --device=/dev/hisi_hdc \
           -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
           -v ${model_dir}:${model_dir} \
           ${docker_image} \
           /bin/bash
```

## 转换模型

1、将vgg16.air模型放入/vgg16_mindspore_1.3.0_code/infer/data/model/目录下(model文件夹需要自己创建，转换前的air模型和转换后的om模型都放在此文件夹下)。

2、准备AIPP配置文件

AIPP需要配置aipp.config文件，在ATC转换的过程中插入AIPP算子，即可与DVPP处理后的数据无缝对接，AIPP参数配置请参见《[CANN 开发辅助工具指南 (推理)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》中“ATC工具使用指南”。

aipp.config文件内容如下，该文件放在/vgg16_mindspore_1.3.0_code/infer/convert目录下

~~~ config
aipp_op {
  aipp_mode: static
  input_format: RGB888_U8

  rbuv_swap_switch: true

  min_chn_0: 123.675
  min_chn_1: 116.28
  min_chn_2: 103.33
  var_reci_chn_0: 0.0171247538316637
  var_reci_chn_1: 0.0175070028011204
  var_reci_chn_2: 0.0174291938997821
}
~~~

3、进入vgg16_mindspore_1.3.0_code/infer/convert目录，执行命令**bash atc.sh ../data/model/vgg16.air**（本例中模型名称为vgg16.air）。利用ATC工具将air模型转换为om模型，om模型会自动放在vgg16_mindspore_1.3.0_code/infer/data/model/文件夹下。

atc.sh

~~~ shell
model=$1
/usr/local/Ascend/atc/bin/atc \
  --model=$model \
  --framework=1 \
  --output=../data/model/vgg16 \
  --input_shape="input:1,224,224,3" \
  --enable_small_channel=1 \
  --log=error \
  --soc_version=Ascend310 \
  --insert_op_conf=aipp_vgg16_rgb.config

~~~

参数说明：

- --model：待转换的air模型所在路径。

- --framework：1代表MindSpore框架。

- --output：转换后输出的om模型存放路径以及名称。

- --input_shape：输入数据的shape。

- --insert_op_conf：aipp配置文件所在路径。

## mxBase推理

1、添加环境变量

通过**vi ~/.bashrc**命令打开~/.bashrc文件，将下面的环境变量添加进当前环境，添加好环境变量以后退出文件编辑，执行**source ~/.bashrc**使环境变量生效。

```bash
export ASCEND_HOME="/usr/local/Ascend"
export ASCEND_VERSION="nnrt/latest"
export ARCH_PATTERN="."
export LD_LIBRARY_PATH="${MX_SDK_HOME}/lib/modelpostprocessors:${LD_LIBRARY_PATH}"
export MXSDK_OPENSOURCE_DIR="${MX_SDK_HOME}/opensource"
```

2、进入vgg16_mindspore_1.3.0_code/infer/mxbase目录，执行指令**bash build.sh**

build.sh

~~~ shell
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
function build_vgg16()
{
    cd $path_cur
    rm -rf build
    mkdir -p build
    cd build
    cmake ..
    make
    ret=$?
    if [ ${ret} -ne 0 ]; then
        echo "Failed to build vgg16."
        exit ${ret}
    fi
    make install
}
check_env
build_vgg16
~~~

3、执行**./vgg16 ../data/input 50000**，推理结果保存在当前目录下的mx_pred_result.txt文件下。

- 第一个参数（../data/input）：图片输入路径。
- 第二个参数（50000）：图片输入数量。

推理结果

~~~bash
ILSVRC2012_val_00047110 221,267,266,206,220,
ILSVRC2012_val_00014552 505,550,804,899,859,
ILSVRC2012_val_00006604 276,287,289,275,285,
ILSVRC2012_val_00016859 2,3,4,148,5,
ILSVRC2012_val_00020009 336,649,350,371,972,
ILSVRC2012_val_00025515 917,921,446,620,692,
ILSVRC2012_val_00046794 427,504,463,412,686,
ILSVRC2012_val_00035447 856,866,595,730,603,
ILSVRC2012_val_00016392 54,67,68,60,66,
ILSVRC2012_val_00023902 50,49,44,39,62,
ILSVRC2012_val_00000719 268,151,171,158,104,
......
~~~

4、验证精度，进入vgg16_mindspore_1.3.0_code/infer/util目录下，执行**python3.7 task_metric.py ../mxbase/mx_pred_result.txt imagenet2012_val.txt vgg16_mx_pred_result_acc.json 5**

参数说明：

- 第一个参数（../mxbase/mx_pred_result.txt）：推理结果保存路径。
- 第二个参数（image2012_val.txt）：验证集标签文件。
- 第三个参数（vgg16_mx_pred_result_acc.json）：结果文件
- 第四个参数（5）："1"表示TOP-1准确率，“5”表示TOP-5准确率。

5、查看推理精度结果

~~~ bash
cat vgg16_mx_pred_result_acc.json
~~~

top-5推理精度

~~~ bash
  "accuracy": [
    0.73328,
    0.83924,
    0.8786,
    0.90034,
    0.91496
]
...
~~~

## MindX SDK推理

1、数据准备

将推理图片数据集放在vgg16_mindspore_1.3.0_code/infer/data/input目录下

- 本例推理使用的数据集是[ImageNet2012](http://www.image-net.org/)中的验证集，input目录下已有其中十张测试图片

- 测试集：6.4 GB，50, 000张图像

2、准备模型推理所需文件

（1）在“/home/data/vgg16_mindspore_1.3.0_code/infer/data/config/”目录下编写pipeline文件。

根据实际情况修改vgg16.pipeline文件中图片规格、模型路径、配置文件路径和标签路径。

更多介绍请参见《[mxManufacture 用户指南](https://ascend.huawei.com/#/software/mindx-sdk/sdk-detail)》中“基础开发”章节。

vgg16.pipeline

```pipeline
{
  "im_vgg16": {
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
        "modelPath": "../data/model/vgg16.om",
        "waitingTime": "1",
        "outputDeviceId": "-1"
      },
      "factory": "mxpi_tensorinfer",
      "next": "mxpi_classpostprocessor0"
    },
    "mxpi_classpostprocessor0": {
      "props": {
        "dataSource": "mxpi_tensorinfer0",
        "postProcessConfigPath": "../data/config/vgg16.cfg",
        "labelPath": "../data/config/imagenet1000_clsidx_to_labels.names",
        "postProcessLibPath": "/usr/local/sdk_home/mxManufacture/lib/modelpostprocessors/libresnet50postprocess.so"
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

（2）在“/home/data/vgg16_mindspore_1.3.0_code/infer/data/config/”目录下编写vgg16.cfg配置文件。

配置文件vgg16.cfg内容如下。

```cfg
CLASS_NUM=1000
SOFTMAX=false
TOP_K=5
```

（3）进入“/home/data/vgg16_mindspore_1.3.0_code/infer/sdk/”目录。

根据实际情况修改main.py文件中裁剪图片的位置和**pipeline**文件路径。

```pipeline
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
    pipeline_conf = "../data/config/vgg16.pipeline"
    stream_name = b'im_vgg16'

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

3、进入vgg16_mindspore_1.3.0_code/infer/sdk文件夹，执行**bash run.sh**，推理结果保存在当前目录下的vgg16_sdk_pred_result.txt文件中。

run.sh

~~~ shell
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

python3.7 main.py "../data/input/*" vgg16_sdk_pred_result.txt
exit 0
~~~

推理结果

~~~bash
ILSVRC2012_val_00047110 221,267,266,206,220
ILSVRC2012_val_00014552 505,550,804,899,859
ILSVRC2012_val_00006604 276,287,289,275,285
ILSVRC2012_val_00016859 2,3,4,148,5
ILSVRC2012_val_00020009 336,649,350,371,972
ILSVRC2012_val_00025515 917,921,446,620,692
ILSVRC2012_val_00046794 427,504,463,412,686
ILSVRC2012_val_00035447 856,866,595,730,603
ILSVRC2012_val_00016392 54,67,68,60,66
......
~~~

4、验证精度，进入vgg16_mindspore_1.3.0_code/infer/util目录下，执行**python3.7 task_metric.py ../sdk/vgg16_sdk_pred_result.txt imagenet2012_val.txt vgg16_sdk_pred_result_acc.json 5**

参数说明：

- 第一个参数（../sdk/vgg16_sdk_pred_result.txt）：推理结果保存路径。

- 第二个参数（imagenet2012_val.txt）：验证集标签文件。

- 第三个参数（vgg16_sdk_pred_result_acc.json）：结果文件

- 第四个参数（5）："1"表示TOP-1准确率，“5”表示TOP-5准确率。

5、查看推理性能和精度

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

- 查看推理精度

  ```bash
  cat vgg16_sdk_pred_result_acc.json
  ```

  top-5推理精度

  ```bash
  "total": 50000,
    "accuracy": [
      0.73328,
      0.83924,
      0.8786,
      0.90034,
      0.91496
  ]
  ...
  ```

