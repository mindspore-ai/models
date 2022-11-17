# FastSCNN MindX推理及mx Base推理

<!-- TOC -->

- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [模型转换](#模型转换)
    - [MindX SDK 启动流程](#{mindx\quadsdk\quad启动流程})
    - [mx Base 推理流程](#{mx\quadbase\quad推理流程})

<!-- /TOC -->

## 脚本说明

### 脚本及样例代码

```tex
├── fastscnn
    ├── README_CN.md                           // fastscnn 的说明文件
    ├── infer
        ├── README_CN.md                       // fastscnn 的 MindX SDK 推理及 mx Base 推理的说明文件
        ├── convert
        │   ├──ATC_AIR_2_OM.sh                 // 将 air 模型转换为 om 模型的脚本
        ├── data
        │   ├──config
        │   │   ├──fastscnn.pipeline           // MindX SDK运行所需的 pipline 配置文件
        ├── mxbase                             // mx Base 推理目录（C++）
        │   ├──src
        │   │   ├──FastSCNN.h                  // 头文件
        │   │   ├──FastSCNN.cpp                // 详细实现
        │   │   ├──main.cpp                    // mx Base 主函数
        │   ├──build.sh                        // 代码编译脚本
        │   ├──CMakeLists.txt                  // 代码编译设置
        ├── sdk
        │   ├──main.py                         // MindX SDK 运行脚本
        │   ├──run.sh                          // 启动 main.py 的 sh 文件
    ├── ......                                 // 其他代码文件
```

### 模型转换

1、首先执行 fastscnn 目录下的 export.py 将准备好的权重文件转换为 air 模型文件。

```python
# 此处简要举例
python export.py \
--batch_size=1 \
--image_height=768 \
--image_width=768 \
--ckpt_file=xxx/fastscnn.ckpt \
--file_name=fastscnn \
--file_format='MINDIR' \
--device_target=Ascend \
--device_id=0 \

#转换完成后请将生成的fastscnn.air文件移动至infer目录下。
```

2、然后执行 convert 目录下的 ATC_AIR_2_OM.sh 将刚刚转换好的 air 模型文件转换为 om 模型文件以备后续使用。

```bash
# bash ./ATC_AIR_2_OM.sh -h 或者 bash ./ATC_AIR_2_OM.sh --help 可以查看帮助信息
bash ATC_AIR_2_OM.sh [--model --output]

e.g.
bash ATC_AIR_2_OM.sh --model=../fastscnn.air --output=../data/model/fastscnn
#转换后的模型结果为fastscnn.om，将放在{fastscnn}/infer/data/model/目录下
```

### 数据集准备

此处以原始数据集 Cityscapes 为例。

1、在 infer 目录下或其他文件夹下创建数据集存放文件夹（记为“xxx/dataset/”），并将 Cityscapes 中的 gtFine、leftImg8bit 文件夹上传至此，gtFine 和 leftImg8bit 中的 train、test文件夹可不上传。“xxx/dataset/” 下的数据集组织结构如下：

```tex
├── xxx/dataset/                           // 数据集根目录
    │   ├──gtFine                          // 标注内容
    │   │   ├──val                         // 验证集
    │   │   │   ├── ....
    │   │   │   ├── ....
    │   ├──leftImg8bit                     // 原始图片
    │   │   ├──val                         // 验证集
    │   │   │   ├── ....
    │   │   │   ├── ....
```

2、MindX SDK 直接使用原始数据集，上述数据集准备好后即可执行 “MindX SDK 启动流程” 步骤。

3、mx Base 推理并不直接处理图片和计算语义分割后的 mIoU 值。因此我们需要首先执行 fastscnn 目录下的 preprocess.py 对测试图片进行归一化、对 label 进行 class 映射等操作并以 "bin" 文件形式进行存储。

```python
# 此处简要举例，“xx/preprocess_data” 为处理后的数据存储路径，不存在时会自动创建
python preprocess.py \
--image_path=xx/dataset \
--out_dir=xx/preprocess_data \
--image_height=768 \
--image_width=768
```

数据预处理完毕后即可执行 “mx Base 推理流程” 步骤。

### MindX SDK 启动流程

```shell
# 通过 bash 脚本启动 MindX SDK 推理
# bash ./run.sh -h 或者 bash ./run.sh --help 可以查看帮助信息
bash ./run.sh [--pipeline --image_path --image_width --image_height --save_mask --mask_result_path]
# 注意： data/config/fastscnn.pipeline 中默认 MindX SDK 推理所需的模型文件为 "fastscnn.om"，且放在 data/model/ 目录下，具体可以修改该文件中的 "modelPath" 属性进行配置。

# 通过 python 命令启动 MindX SDK 推理
python main.py \
--pipeline=../data/config/fastscnn.pipeline \
--image_path=xxx/dataset/ \
--image_width=768 \
--image_height=768 \
--save_mask=1 \
--mask_result_path=./mask_result
# 注意： data/config/fastscnn.pipeline 中默认 MindX SDK 推理所需的模型文件为 "fastscnn.om"，且放在 data/model/ 目录下，具体可以修改该文件中的 "modelPath" 属性进行配置。
```

推理结果示例：

```tex
Begin to initialize Log.
The output directory of logs file exist.
Save logs information to specified directory.
Found 500 images in the folder /home/data/FastSCNN/dataset/leftImg8bit/val
Processing --->  frankfurt_000001_029086_leftImg8bit
[EVAL] Sample: 1, pixAcc: 93.023, mIoU: 34.965
Processing --->  frankfurt_000001_064651_leftImg8bit
[EVAL] Sample: 2, pixAcc: 94.525, mIoU: 39.346
Processing --->  frankfurt_000001_023235_leftImg8bit
[EVAL] Sample: 3, pixAcc: 93.609, mIoU: 43.201
........
Processing --->  lindau_000030_000019_leftImg8bit
[EVAL] Sample: 497, pixAcc: 93.535, mIoU: 55.196
Processing --->  lindau_000001_000019_leftImg8bit
[EVAL] Sample: 498, pixAcc: 93.543, mIoU: 55.501
Processing --->  lindau_000025_000019_leftImg8bit
[EVAL] Sample: 499, pixAcc: 93.539, mIoU: 55.485
Processing --->  lindau_000040_000019_leftImg8bit
[EVAL] Sample: 500, pixAcc: 93.546, mIoU: 55.487
End validation pixAcc: 93.546, mIoU: 55.487
Category iou:
 +------------+---------------+----------+
|  class id  |  class name   |   iou    |
+============+===============+==========+
|     0      |     road      | 0.976416 |
+------------+---------------+----------+
|     1      |   sidewalk    | 0.662959 |
+------------+---------------+----------+
|     2      |   building    | 0.866088 |
+------------+---------------+----------+
|     3      |     wall      | 0.320282 |
+------------+---------------+----------+
|     4      |     fence     | 0.248646 |
+------------+---------------+----------+
|     5      |     pole      | 0.31713  |
+------------+---------------+----------+
|     6      | traffic light | 0.360871 |
+------------+---------------+----------+
|     7      | traffic sign  | 0.485951 |
+------------+---------------+----------+
|     8      |  vegetation   | 0.896945 |
+------------+---------------+----------+
|     9      |    terrain    | 0.503119 |
+------------+---------------+----------+
|     10     |      sky      | 0.928662 |
+------------+---------------+----------+
|     11     |    person     | 0.632751 |
+------------+---------------+----------+
|     12     |     rider     | 0.370751 |
+------------+---------------+----------+
|     13     |      car      | 0.851971 |
+------------+---------------+----------+
|     14     |     truck     | 0.496429 |
+------------+---------------+----------+
|     15     |      bus      | 0.565111 |
+------------+---------------+----------+
|     16     |     train     | 0.289486 |
+------------+---------------+----------+
|     17     |  motorcycle   | 0.259988 |
+------------+---------------+----------+
|     18     |    bicycle    | 0.508992 |
+------------+---------------+----------+
Testing finished....
=======================================
The total time of inference is 412.33897733688354 s
=======================================
```

### mx Base 推理流程

1、编译 mx Base

```shell
bash ./build.sh
# 编译后的可执行文件 "fastscnn" 将保存在当前目录下
```

2、执行 mx Base 推理

```tex
./fastscnn [model_path input_data_path output_data_path]
# 按顺序传入模型路径、图像路径(“数据集准备” 中的图片保存路径 “xx/preprocess_data” 下的 images 目录,以"/"结尾)、输出路径（需要提前创建）
例如：
 ./fastscnn ../data/model/fastscnn.om xx/preprocess_data/images ./result/
```

mx Base 推理结果示例：

```tex
I1108 03:06:01.198787 86423 main.cpp:53] =======================================  !!!Parameters setting!!! ========================================
I1108 03:06:01.198843 86423 main.cpp:55] ==========  loading model weights from: ../data/model/fastscnn.om
I1108 03:06:01.198853 86423 main.cpp:58] ==========  input data path = ../preprocess_data/images/
I1108 03:06:01.198858 86423 main.cpp:61] ==========  output data path = ./result/ WARNING: please make sure that this folder is created in advance!!!
I1108 03:06:01.198861 86423 main.cpp:63] ========================================  !!!Parameters setting!!! ========================================
I1108 03:06:01.552381 86423 ModelInferenceProcessor.cpp:22] Begin to ModelInferenceProcessor init
I1108 03:06:01.661561 86423 ModelInferenceProcessor.cpp:69] End to ModelInferenceProcessor init
I1108 03:06:01.661762 86423 main.cpp:82] Processing: 1/500 ---> frankfurt_000001_023769_leftImg8bit_img.bin
I1108 03:06:02.280328 86423 main.cpp:82] Processing: 2/500 ---> frankfurt_000001_067295_leftImg8bit_img.bin
I1108 03:06:02.903029 86423 main.cpp:82] Processing: 3/500 ---> frankfurt_000000_011074_leftImg8bit_img.bin
I1108 03:06:03.528358 86423 main.cpp:82] Processing: 4/500 ---> frankfurt_000000_002196_leftImg8bit_img.bin
I1108 03:06:04.150723 86423 main.cpp:82] Processing: 5/500 ---> frankfurt_000001_073243_leftImg8bit_img.bin
I1108 03:06:04.769243 86423 main.cpp:82] Processing: 6/500 ---> frankfurt_000001_082087_leftImg8bit_img.bin
I1108 03:06:05.391845 86423 main.cpp:82] Processing: 7/500 ---> frankfurt_000001_055172_leftImg8bit_img.bin
........
I1108 03:07:17.471393 86423 main.cpp:91] infer succeed and write the result data with binary file !
I1108 03:07:17.758675 86423 DeviceManager.cpp:83] DestroyDevices begin
I1108 03:07:17.758706 86423 DeviceManager.cpp:85] destroy device:0
I1108 03:07:17.950556 86423 DeviceManager.cpp:91] aclrtDestroyContext successfully!
I1108 03:07:18.839597 86423 DeviceManager.cpp:99] DestroyDevices successfully
I1108 03:07:18.839629 86423 main.cpp:98] Infer images sum 500, cost total time: 256694.6 ms.
I1108 03:07:18.839658 86423 main.cpp:99] The throughput: 1.94784 bin/sec.
I1108 03:07:18.839663 86423 main.cpp:100] ==========  The infer result has been saved in ---> ./result/

```

mx Base 的推理结果为 "图片的语义分割表示"，将以 "bin" 文件的形式存储在指定路径下。

3、计算语义分割的 mIoU 值和将语义分割结果进行可视化展示

如果需要计算语义分割的 mIoU 值和将语义分割结果进行可视化展示，请执行 fastscnn 目录下的 cal_mIoU.py 脚本

```python
# 此处简要举例
python cal_mIoU.py \
--label_path=xx/preprocess_data/labels \
--output_path=xxx/infer/mxbase/result \
--image_height=768 \
--image_width=768 \
--save_mask=1
--mask_result_path=./mask_result
label_path 指第1步处理后的label保存路径，output_path 指 mx Base 推理后的结果保存路径
```

mIoU 计算结果示例：

```tex
[EVAL] Sample: 1, pixAcc: 96.204, mIoU: 32.133
[EVAL] Sample: 2, pixAcc: 96.826, mIoU: 37.931
[EVAL] Sample: 3, pixAcc: 96.307, mIoU: 36.358
[EVAL] Sample: 4, pixAcc: 95.621, mIoU: 39.828
.......
[EVAL] Sample: 497, pixAcc: 93.530, mIoU: 55.492
[EVAL] Sample: 498, pixAcc: 93.536, mIoU: 55.489
[EVAL] Sample: 499, pixAcc: 93.543, mIoU: 55.495
[EVAL] Sample: 500, pixAcc: 93.546, mIoU: 55.487
End validation pixAcc: 93.546, mIoU: 55.487
Category iou:
 +------------+---------------+----------+
|  class id  |  class name   |   iou    |
+============+===============+==========+
|     0      |     road      | 0.976416 |
+------------+---------------+----------+
|     1      |   sidewalk    | 0.662959 |
+------------+---------------+----------+
|     2      |   building    | 0.866088 |
+------------+---------------+----------+
|     3      |     wall      | 0.320282 |
+------------+---------------+----------+
|     4      |     fence     | 0.248646 |
+------------+---------------+----------+
|     5      |     pole      | 0.31713  |
+------------+---------------+----------+
|     6      | traffic light | 0.360871 |
+------------+---------------+----------+
|     7      | traffic sign  | 0.485951 |
+------------+---------------+----------+
|     8      |  vegetation   | 0.896945 |
+------------+---------------+----------+
|     9      |    terrain    | 0.503119 |
+------------+---------------+----------+
|     10     |      sky      | 0.928662 |
+------------+---------------+----------+
|     11     |    person     | 0.632751 |
+------------+---------------+----------+
|     12     |     rider     | 0.370751 |
+------------+---------------+----------+
|     13     |      car      | 0.851971 |
+------------+---------------+----------+
|     14     |     truck     | 0.496429 |
+------------+---------------+----------+
|     15     |      bus      | 0.565111 |
+------------+---------------+----------+
|     16     |     train     | 0.289486 |
+------------+---------------+----------+
|     17     |  motorcycle   | 0.259988 |
+------------+---------------+----------+
|     18     |    bicycle    | 0.508992 |
+------------+---------------+----------+
Time cost:188.06731748580933 seconds!
```
