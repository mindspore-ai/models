# 推理

## 模型转换

   ```bash

   cd infer/convert

   ```

1. 准备模型文件。

   AIR模型为在昇腾910服务器上导出的模型，导出AIR模型的详细步骤请参见“模型训练”。

2. 执行以下命令，进行模型转换。

   转换详细信息可查看转换脚本和对应的AIPP配置文件，转换命令如下。

   **bash convert/convert_om.sh** *air_path* *aipp_cfg_path* *om_path*

   | 参数          | 说明                                              |
   | ------------- | ------------------------------------------------- |
   | air_path      | 转换脚本AIR文件路径。                             |
   | aipp_cfg_path | AIPP配置文件路径。                                |
   | om_path       | 生成的OM文件名，转换脚本会在此基础上添加.om后缀。 |

   转换示例如下所示。

   ```bash

   # 转换模型
   bash convert_om.sh fairmot.air aipp_rgb.cfg fairmot

   ```

## 推理数据集下载

1. 下载推理数据集[MOT20](https://motchallenge.net/data/MOT20/)。

2. 将数据集存放在`infer/data/MOT20`目录下。

3. 目录格式为：

```text
   └─fairmot
      ├─infer
        ├─data
        │ ├─MOT20
        │ │ └─train
        │ │    ├─MOT20-01
        │ │    ├─MOT20-02
        │ │    ├─MOT20-03
        │ │    └─MOT20-05
        │ └─data.json
        ├─infer
        │ ├─convert                   //  模型转换脚本
        │ ├─mxbase                   // mxbase 推理脚本
        │ └─sdk                   // sdk推理脚本
```

## mxBase推理

   ```bash

   cd infer/mxbase

   ```

1. 编译工程。

   目前mxBase推理仅实现了基于DVPP方式推理。

   ```bash

   bash build.sh

   ```

2. （可选）修改配置文件。

   可根据实际情况修改，配置文件位于`mxbase/src/main.cpp`中，可修改参数如下。

   ```c++

   namespace {
   const uint32_t DEVICE_ID = 0;
   } // namespace
   ...

   ```

3. 运行推理服务。

   运行推理服务：
   **./build/fairmot_mindspore**  *om_path* *img_path*
   | 参数       | 说明                           |
   | ---------- | ------------------------------ |
   | om_path | om存放路径。如：`../convert/fairmot.om`。 |
   | img_path | 推理图片路径。如：`../../data/MOT20/`。 |

   ```bash

   ./build/fairmot_mindspore ../convert/fairmot.om ../../data/MOT20/

   ```

4. 观察结果。
   推理结果以txt格式保存，路径为`../../data/MOT20/result_Files`.

5. 可视化结果并得到精度。

   运行精度测试以及可视化：
   **python3.7 mx_base_eval.py**  *result_path*
   | 参数       | 说明                           |
   | ---------- | ------------------------------ |
   | result_path | 推理结果路径。如：“../../data/MOT20”。 |

   ```bash

   cd ../../
   python3.7 mx_base_eval.py --data_dir data/MOT20

   ```

6. 查看精度结果以及可视化结果。

   图片保存在`data/MOT20/result`里。精度测试结果在运行完`mxbase_eval.py`以后会在终端显示并以xlsx文件格式保存在`data/MOT20`。

## MindX SDK推理

   ```bash

   cd infer/sdk

   ```

1. （可选）修改配置文件。

   1. 可根据实际情况修改pipeline文件。

      ```python

      ├── config
      │   ├── config.py
      │   └──  fairmot.pipeline # PIPELINE文件

      ```

2. 模型推理。

   1. 执行推理。

      切换到sdk目录下，执行推理脚本。
      **python main.py**  *img_path* *pipeline_path* *infer_result_path* *infer_mode*
      | 参数        | 说明                                  |
      | ----------- | ------------------------------------- |
      | img_path  | 推理图片路径。如：“../data/MOT20”。        |
      | pipeline_path | 存放pipeline路径。如："./config/fairmot.pipeline"。 |
      | infer_result_path | 存放推理结果的路径。如："../data/infer_result"。 |
      | infer_mode | 推理模式，默认为infer，可用eval直接得到精度对比 |

      ```bash

      python3.7 main.py --img_path ../data/MOT20 --pipeline_path ./config/fairmot.pipeline --infer_result_path ../data/infer_result

      ```

   2. 查看推理结果。
      推理结果以bin文件形式保存在`../data/infer_result`目录下。
3. 执行精度测试以及可视化。

   切换到fairmot目录下，执行推理脚本：
      **python3.7 sdk_eval.py**  *img_path*

   ```bash
   cd ../
   python3.7 sdk_eval.py --data_dir ./data/MOT20
   ```

4. 查看精度结果以及可视化结果。

   图片保存在`data/MOT20/results`里。精度测试结果在运行完`sdk_eval.py`以后会在终端显示并以xlsx文件格式保存在`data/MOT20`。
