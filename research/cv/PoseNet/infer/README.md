# 推理

## 模型转换

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
   bash convert/convert_om.sh xxx/posenet.air convert/aipp.cfg posenet

   ```

3. 将转换好的模型移动到存放模型的目录下。

   ```bash

   # 移动模型
   mv posenet.om data/model/

   ```

## mxBase推理

1. 编译工程。

   目前mxBase推理仅实现了基于DVPP方式推理。

   ```bash

   cd xxx/infer/mxbase
   bash build.sh

   ```

2. （可选）修改配置文件。

   可根据实际情况修改，配置文件位于“mxbase/src/main.cpp”中，可修改参数如下。

```c++

namespace {
const uint32_t DEVICE_ID = 0;  // 指定设备ID，默认为0,可根据实际情况调整
std::string RESULT_PATH = "../data/mx_result/";  // 推理结果保存路径
} // namespace
...

```

3. 运行推理服务。

   运行推理服务（确保你的输出路径存在，否则会报错）。
   **./build/midas_mindspore**  *om_path* *img_path* *dataset_name*
   | 参数       | 说明                           |
   | ---------- | ------------------------------ |
   | om_path | om存放路径。如：“../data/model/posenet.om”。 |
   | img_path | 推理图片路径。如：“../dataset/KingsCollege/”。 |
   | dataset_name | 推理数据集名称。如：“seq2”。 |

4. 观察结果。
   推理结果以bin格式保存，路径为“./result_Files/”.

## MindX SDK推理

1. （可选）修改配置文件。

   1. 可根据实际情况修改pipeline文件。

      ```python

      ├── config
      │   ├── config.py
      │   └──  posenet.pipeline # PIPELINE文件

      ```

2. 模型推理。

   1. 执行推理。

      切换到sdk目录下，执行推理脚本。
      **python3.7 main.py**  *--img_path*
      | 参数        | 说明                                  |
      | ----------- | ------------------------------------- |
      | img_path  | 推理图片路径。如：“../dataset/KingsCollege”。        |

3. 执行精度测试。
   推理精度脚本存放在"sdk/eval"目录下。保存为acc.log文件,脚本接受1个参数，依次是数据集路径。脚本示例如下：

   ```python
   python eval_by_sdk.py --result_path ../infer_result >acc.log
   ```

4. 查看性能结果。
   性能结果如下：

   ```bash

   Median error 1.8816218573685832 m and 4.192690626738049 degrees.

   ```
