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
   bash convert/convert_om.sh xxx/ibnnet.air convert/aipp.cfg ./data/model/midas
   ```

   注意model文件夹需要自己创建。

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
   | om_path | om存放路径。如：“../data/model/midas.om”。 |
   | img_path | 推理图片路径。如：“../data/input/”。 |
   | dataset_name | 推理数据集名称。如：“Kitti”。 |

   不同数据集下推理数据集名字不同

4. 观察结果。
   推理结果需要结合测试精度进行查看，测试目录放在../sdk/eval目录下,切换到目录后执行eval_by_sdk.py文件（注意：切换到sdk目录执行而不是直接进入到eval目录）。脚本支持四个参数，依次是数据集路径，输出结果路径，数据集名称，是否可视化.脚本示例如下：

```python
python eval/eval_by_sdk.py --dataset_path ../data/input  --result_path ../data --dataset_name Kitti --visualization True
```

   推理结果以png格式保存，路径为“../data/Kitti/”.
   不同数据集在不同文件夹下，都在data文件夹下面。

## MindX SDK推理

1. （可选）修改配置文件。

   1. 可根据实际情况修改pipeline文件。

      ```python

      ├── config
      │   ├── config.py
      │   └──  midas_ms_test.pipeline # PIPELINE文件

      ```

2. 模型推理。

   1. 执行推理。

      切换到sdk目录下，执行推理脚本。
      **python main.py**  *img_path* *pipeline_path* *infer_result_dir* *dataset_name* *infer_mode*
      | 参数        | 说明                                  |
      | ----------- | ------------------------------------- |
      | img_path  | 推理图片路径。如：“../data/”。        |
      | pipeline_path | 存放pipeline路径。如："../data/config/midas_ms_test.pipeline"。 |
      | infer_result_dir | 存放推理结果的路径。如："./result/"。 |
      | dataset_name | 数据集名称。如："Kitti"。 |
      | infer_mode | 推理模式，默认为infer，可用eval直接得到精度对比 |

      不同数据集下数据集名称会有所不同，Kitti数据集的dataset_name为Kitti,Sintel数据集为Sintel,TUM数据集为TUM。

   2. 查看推理结果。
      推理结果需执行下面的精度测试脚本，visualization参数控制是否需要查看推理结果，结果放在"infer/data/result"目录下，以png格式文件保存，可以直接打开查看。

3. 执行精度测试。

   推理精度脚本存放在"./eval"目录下。保存为result_val.json文件,脚本接受4个参数，依次是数据集路径，输出结果路径，数据集名称，是否可视化。脚本示例如下：

```python
python eval_by_sdk.py --dataset_path ../data/input  --result_path ../data/result  --dataset_name Kitti --visualization True
```

4. 查看性能结果。

```bash

# cat ./eval/result_val.json
{"TUM": 13.319143050564358}

```

(注意：json文件在不同数据集测试后会覆盖)

