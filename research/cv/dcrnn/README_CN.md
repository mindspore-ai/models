# 目录

<!-- TOC -->

- [目录](#目录)
- [脚本说明](#脚本说明)
    - [推理脚本及样例代码](#脚本及样例代码)
    - [推理过程](#推理过程)
 <!-- /TOC -->

# 脚本说明

## 脚本及样例代码

- 推理脚本文件和代码文件的目录结构如下：

    ```text
  └── dcrnn
        └── mindspore_infer
            ├── ascend_310_infer                  # 310推理
            ├── README_CN.md                      # 模型推理相关说明
            ├── scripts
            │   ├── run_310_infer.sh              # 用于310推理的shell脚本
            ├── data                              # 存放需要进行预处理的数据
            │   ├── test.npz                      # 数据集test
            │   ├── train.npz                     # 数据集train
            │   ├── val.npz                       # 数据集val
            ├── DCRNN_mindir.mindir               # 模型导出文件
            ├── fusion_switch.cfg                 # 推理性能相关配置文件
            ├── postprocess.py                    # 后处理
            ├── preprocess.py                     # 前处理
            ├── preprocess_Result                 # 前处理结果（前处理结束后自动生成）
            │        ├── data                     # 前处理后数据
            │        └── label                    # 前处理后标签
    ```

## 推理过程

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

### 推理

在310机器中执行推理前，需要通过export.py导出mindir文件，mindir文件的地址将作为后续run_infer_310.sh推理脚本的参数。

在执行推理前，数据集需要首先进行前处理（该步骤已经集成在后续推理脚本中，无须单独运行命令），而后利用生成在preprocess_Result目录下的data数据进行推理。

前处理同样在310上进行，请先将train.npz、test.npz、val.npz数据集放置于data目录下，该目录位置将作为后续推理脚本run_infer_310.sh的参数。为了方便理解，脚本中有关前处理的命令如下，参数int-dataset-path即为data的目录地址。

```bash
python preprocess.py --init-dataset-path=$data_path &> pre_info.log
```

其中pre_info.log为日志文件，若未创建则在前处理过程中自动创建，您也可自行更改日志名称。
如果出现文件无法找到的情况，请检查是否输入正确地址。

若成功完成数据预处理步骤，即可获得preprocess_Result文件夹。 preprocess_Result存放于目录mindspore_infer，其下分别有两个子文件夹，data与label，前者存放前处理产生数据的bin文件，后者存放前处理生成的label的bin文件

preprocess_Result具体的文件结构如下所示：

```text
├── data
│   ├── data_0.bin
│   └── data_1.bin
│   └── ......
└── label
│   ├── label_0.bin
│   └── label_1.bin
│   └── ......
```

此外，preprocess_Result文件夹为自动生成，相对路径在精度校验的命令中已经给定。
若需要修改相关路径，请具体查看使用该路径的推理脚本中的infer()和cal_acc()函数，如果使用其他路径亦可从中更改。

你也可以忽略上述描述，在准备好数据集和模型文件后直接使用脚本run_infer_310.sh在310机器中进行推理完成上述所有步骤。请切换至scripts目录下，运行如下推理脚本：

```bash
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID]
```

其中[MINDIR_PATH]是先前导出的mindir模型文件的路径，
[DATA_PATH]是三个npz格式数据集存放的data的路径，
[DEVICE_ID]是设备编号，若无特殊指定则默认为0。

在scripts目录运行命令即可进行对应模型的推理。

若按照本文所描述目录结构组织脚本和数据，可得如下实例推理脚本：

```bash
bash run_infer_310.sh ../DCRNN_mindir.mindir ../data 0
```

如果出现文件无法找到情况，请检查文件路径是否正确或尝试绝对路径。

推理结束后可在scripts目录中得到infer.log和acc_info.log两个日志，分别输出推理和精度检验的信息。

以下是310的推理部分结果（可在对应的infer.log和acc_info.log中查询）：

```bash
...
NN inference cost average time: 651.98ms of infer_count 54
eval result:  2.701705907781919
```



