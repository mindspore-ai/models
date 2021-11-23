# 推理

## 模型转换

1. 准备模型文件。

   AIR模型为在昇腾910服务器上导出的模型，导出AIR模型的详细步骤请参见“模型训练”。

2. 执行以下命令，进行模型转换。

   转换详细信息可查看转换脚本，转换命令如下。

   **bash convert/convert.sh** *air_path* *om_path*

   | 参数          | 说明                                              |
   | ------------- | ------------------------------------------------- |
   | air_path      | 转换脚本AIR文件路径。                             |
   | om_path       | 生成的OM文件名，转换脚本会在此基础上添加.om后缀。 |

   转换示例如下所示。

   ```bash

   # 转换模型
   bash convert/convert.sh convert/atae_lstm.air atae_lstm

   ```

3. 将转换好的模型移动到存放模型的目录下。

   ```bash

   # 移动模型
   mv atae_lstm.om data/model/

   ```

## 推理数据集准备

1. 将数据集存放在`data/input/`目录下。

2. 目录格式为：

```text
   └─infer
      ├─data
      │ ├─input
      │ │ ├─00_content
      │ │ ├─01_sen_len
      │ │ ├─02_aspect
      │ │ └─solution_path
      | ├─config
      | └─model
      ├─convert            //  模型转换脚本
      ├─mxbase             // mxbase 推理脚本
      └─sdk                // sdk推理脚本
```

## mxBase推理

1. 编译工程。

   ```bash

   cd infer/mxbase
   bash build.sh

   ```

2. （可选）修改配置文件。

   可根据实际情况修改，配置文件位于“mxbase/src/main.cpp”中，可修改参数如下。

```c++

void InitAtaeLstmParam(InitParam* initParam) {
    initParam->deviceId = 0; // 指定设备ID，默认为0,可根据实际情况调整
    initParam->modelPath = "../data/model/atae_lstm.om"; //om模型路径
}
...

```

3. 运行推理服务。

   运行推理服务。
   **./main**  *dataset_path*
   | 参数       | 说明                           |
   | ---------- | ------------------------------ |
   | dataset_path | 推理数据集路径。如：“../data/input”。 |

4. 观察结果。
   推理结果保存在当前目录下的“mxbase_infer_result”文件夹内，可以通过进入文件夹查看。推理结果以txt格式保存。

```shell
cd mxbase_infer_result
cat predict.txt
```

## MindX SDK推理

1. （可选）修改配置文件。

   1. 可根据实际情况修改pipeline文件。

      ```python

      ├── data
      │   ├── config
      │   │   └── atae_lstm.pipeline # PIPELINE文件

      ```

2. 模型推理。

   1. 执行推理。

      切换到sdk目录下，执行推理脚本。
      **python main.py** *pipeline_file* *data_dir* *res_dir* *infer_mode* *do_eval*
      | 参数        | 说明                                  |
      | ----------- | ------------------------------------- |
      | pipeline_file | 存放pipeline路径。如："../data/config/atae_lstm.pipeline"。 |
      | data_dir  | 推理数据集路径。如：“../data/input”。        |
      | res_dir | 存放推理结果的路径。如："./sdk_infer_result/"。 |
      | do_eval | 是否进行评估，默认为True，可直接得到精度对比 |

   2. 查看推理结果。

      推理结果保存在当前目录下的“sdk_infer_result”文件夹内，可以通过进入文件夹查看。推理结果以txt格式保存。

        ```shell
        cd sdk_infer_result
        cat predict.txt
        ```

    3. 查看性能结果。

        ```bash

        ---accuracy: 0.7543679342240494 ---

        Infer images sum: 973, cost total time: 21.514804 sec.

        ```
