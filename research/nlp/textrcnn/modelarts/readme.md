# ModelArts脚本

modelarts脚本运行于华为云modelarts一站式开发平台，提供TextCnn模型训练、模型评估和模型冻结等功能，代码目录结构如下：

```bash
modelarts
|-- start.py              # 模型训练脚本
`-- readme.md
```

本项目脚本参数从根目录下 modelarts_config.yaml 文件读取。如需设定或更改脚本参数，请在脚本运行前修订modelarts_config.yaml相关字段。如需在modelarts上运行训练，请将textrcnn/src/model_utils/config.py中config_yaml修改为"modelarts_config.yaml".

default_config.yaml 文件内容如下：

```yaml
# Builtin Configurations(DO NOT CHANGE THESE CONFIGURATIONS unless you know exactly what you are doing)
enable_modelarts: False
# Url for modelarts
data_url: ""
train_url: ""
checkpoint_url: ""
bucket_name: "textrcnn-ms"
# Path for local
data_path: "/cache/data"
output_path: "/cache/train"
load_path: "/cache/checkpoint_path"
device_target: "Ascend"
enable_profiling: False

# ==============================================================================
pos_dir: 'data/rt-polaritydata/rt-polarity.pos'
neg_dir: 'data/rt-polaritydata/rt-polarity.neg'
num_epochs: 10
lstm_num_epochs: 10
batch_size: 1
cell: 'lstm'
ckpt_folder_path: './ckpt'
preprocess_path: './preprocess'
preprocess: 'true'
data_root: './data/'
lr: 0.001  # 1e-3
lstm_lr_init: 0.002  # 2e-3
lstm_lr_end: 0.0005  # 5e-4
lstm_lr_max: 0.003  # 3e-3
lstm_lr_warm_up_epochs: 2
lstm_lr_adjust_epochs: 9
emb_path: './word2vec'
embed_size: 300
save_checkpoint_steps: 9594
keep_checkpoint_max: 10
ckpt_path: ''

# Export related
ckpt_file: ''
file_name: 'textrcnn'
file_format: "MINDIR"

# post_process and result_path related
pre_result_path: "./preprocess_Result"
label_path: "./preprocess_Result/label_ids.npy"
result_path: "./result_Files"

---

# Help description for each configuration
# ModelArts related
enable_modelarts: "Whether training on modelarts, default: False"
data_url: "Url for modelarts"
train_url: "Url for modelarts"
data_path: "The location of the input data."
output_path: "The location of the output file."
device_target: "Running platform, choose from Ascend, GPU or CPU, and default is Ascend."
enable_profiling: 'Whether enable profiling while training, default: False'
# Export related
ckpt_file: "textrcnn ckpt file."
file_name: "textrcnn output file name."
file_format: "file format, choose from MINDIR or AIR"

---
file_format: ["AIR", "MINDIR"]
device_target: ["Ascend"]
```

在modelarts环境下执行模型训练及评估操作时，首先应在华为云对象存储服务obs下创建桶（比如textrcnn），然后在桶内创建code文件夹。

由于脚本执行依赖于r1.3 版本modelzoo目录[textcnn](https://gitee.com/mindspore/mindspore/tree/r1.3/model_zoo/research/nlp/textrcnn/)下的模块库，要求将网站textrcnn目录下的代码（含子目录）整体拷贝至桶内code目录下；同时，将本项目modelarts文件夹也拷贝至code根目录下。

在textcnn桶内分别创建 /data, /output, /logs目录，分别用于存放模型输入参数、生成checkpoint的模型参数、训练过程log内容。

## 执行训练脚本

结合推理服务场景需求，在modelarts平台进行 batchsize=1 的模型训练。修订modelarts_config.yaml文件的如下字段：

```yaml
enable_modelarts: True
batch_size: 1
save_checkpoint_steps: 9594
```

batchsize=64 的模型训练。修订modelarts_config.yaml文件的如下字段：

```yaml
enable_modelarts: True
batch_size: 64
save_checkpoint_steps: 149
```

执行模型训练之前，需下载数据集。数据集使用[Movie Review Data](http://www.cs.cornell.edu/people/pabo/movie-review-data/)，可以从链接网站下载MR数据压缩包，解压后将rt-polarity.neg 和 rt-polarity.pos文件保存在obs桶的 /data 目录下。

同时需下载预训练二进制文件 GoogleNews-vectors-negative300.bin，放入 obs桶的 /code/word2vec 目录下
**you can download from https://code.google.com/archive/p/word2vec/,**
**or from https://pan.baidu.com/s/1NC2ekA_bJ0uSL7BF3SjhIg, code: yk9a**

华为云modelarts WEB UI完成算法管理、训练管理的相关配置，内容如下：

| 参数           | 说明                                         |
| -------------- | -------------------------------------------- |
| *AI引擎*       | Ascend-Powered-Engine： Mindspore-1.3.0-python3.7-aarch64 |
| *执行脚本*      | 指向 obs 桶内 /code/modelarts/train_start.py |
| *代码目录*      | 指向 obs 桶内 /code/                         |
| *训练输入1*      | data_url， 指向 obs 桶内 /data/                         |
| *训练输出1*      | train_url， 指向 obs 桶内 /output/                         |

启动训练，成功后出现logs输出内容：

```bash
epoch: 1 step: 9594, loss is 0.016411183
epoch time: 70598.192 ms, per step time: 7.359 ms
epoch: 2 step: 9594, loss is 0.025501017
epoch time: 10969.065 ms, per step time: 1.143 ms
......

train success
```

输出checkpoint模型参数保存为 obs://textrcnn/output/ckpt/gru-10_9594.ckpt

## 执行精度评估脚本

完成上述模型训练后，可以执行精度评估脚本对得出的模型参数实施精度评估。

华为云modelarts WEB UI完成算法管理、训练管理的相关配置，内容如下：

| 参数           | 说明                                         |
| -------------- | -------------------------------------------- |
| *AI引擎*       | Ascend-Powered-Engine： Mindspore-1.3.0-python3.7-aarch64 |
| *执行脚本*      | 指向 obs 桶内 /code/modelarts/eval_start.py |
| *代码目录*      | 指向 obs 桶内 /code/                         |
| *训练输入1*      | data_url， 指向 obs 桶内 /data/              |
| *训练输入2*      | checkpoint_url， 指向 obs 桶内 /output/ckpt/{ckpt_file_name} |

启动评估，成功后出现logs输出内容：

```bash
==================== {'acc': 0.7743445692883895} ====================
```

## 执行模型冻结脚本

完成上述模型训练、评估模型精度满足要求的基础上，可以执行模型冻结操作，生成推理服务所需模型文件。修订modelarts_config.yaml文件的如下字段：

```bash
enable_modelarts: True
batch_size: 1
data_url: "data/"
checkpoint_file: "output/ckpt/gru_10_149.ckpt"
```

华为云modelarts WEB UI完成算法管理、训练管理的相关配置，内容如下：

| 参数           | 说明                                         |
| -------------- | -------------------------------------------- |
| *AI引擎*       | Ascend-Powered-Engine： Mindspore-1.2.0-python3.7-aarch64 |
| *执行脚本*      | 指向 obs 桶内 /code/modelarts/export_start.py |
| *代码目录*      | 指向 obs 桶内 /code/                         |
| *训练输入1*     | data_url， 指向 obs 桶内 /data/              |
| *训练输入2*     | ckpt_file， 指向 obs 桶内 /output/ckpt/{ckpt_file_name} |
| *训练输出*      | train_url， 指向 obs 桶内 /output/           |

启动冻结模型脚本，执行结束后冻结模型保存为obs://{bucket_name}/output/textrcnn.air，将该模型下载并执行 /infer/convert 目录下的 convert.sh 脚本，可得出适用于推理服务的OM模型。

```bash
convert.sh {air_path} {om_path}
```
