# R(2+1)D MindX推理及mx Base推理

<!-- TOC -->

- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [数据集准备](#数据集准备)
    - [模型转换](#模型转换)
    - [MindX SDK 启动流程](#mindx-sdk-启动流程)
    - [mx Base 推理流程](#mx-base-推理流程)

<!-- /TOC -->

## 脚本说明

### 脚本及样例代码

```tex
├── R(2+1)D
    ├── README_CN.md                           // R(2+1)D 的说明文件
    ├── infer
        ├── README_CN.md                       // R(2+1)D 的 MindX SDK 推理及 mx Base 推理的说明文件
        ├── convert
        │   ├──ATC_AIR_2_OM.sh                 // 将 air 模型转换为 om 模型的脚本
        ├── data
        │   ├──config
        │   │   ├──r2p1d.pipeline              // MindX SDK运行所需的 pipline 配置文件
        ├── mxbase                             // mx Base 推理目录（C++）
        │   ├──src
        │   │   ├──R2P1D.h                     // 头文件
        │   │   ├──R2P1D.cpp                   // 详细实现
        │   │   ├──main.cpp                    // mx Base 主函数
        │   ├──build.sh                        // 代码编译脚本
        │   ├──CMakeLists.txt                  // 代码编译设置
        ├── sdk
        │   ├──main.py                         // MindX SDK 运行脚本
        │   ├──run.sh                          // 启动 main.py 的 sh 文件
    ├── ......                                 // 其他代码文件
```

### 数据集准备

此处以 UCF101 原始数据集为例。

**注意：数据集预处理需要的 decord 包在 aarch64 架构的机器上无法安装，请使用 x86 架构的机器。**

1、在 infer 目录下或其他文件夹下创建数据集存放文件夹（记为“xxx/ucf101/”），并将 UCF101 视频数据集上传到该文件夹下。“xxx/ucf101/” 下的数据集组织结构如下：

```tex
├── xxx/ucf101/                           // 数据集根目录
    │   ├──ApplyEyeMakeup
    │   ├──ApplyLipstick
    │   ├──....
```

然后将 “/home/r2p1d”(假设为代码解压目录) 文件夹下的 default_config.yaml 文件中的 “splited” ，“source_data_dir”，“output_dir” 参数依次设置为 0、“xxx/ucf101/”、“./infer/ucf101_img”，最后在该目录下执行 python3.7 dataset_process.py ，等待执行完毕即可。执行完毕后，我们会得到如下结构的数据集：

  ```tex
  ├── /home/r2p1d/infer/ucf101_img           // 数据集根目录
      │   ├──train                           // 训练集
      │   │   ├── ApplyEyeMakeup
      │   │   ├── ApplyLipstick
      │   │   ├── ....
      │   ├──val                             // 验证集
      │   │   ├── ApplyEyeMakeup
      │   │   ├── ApplyLipstick
      │   │   ├── ....
  ```

2、MindX SDK 直接使用原始数据集，上述数据集准备好后即可执行 “MindX SDK 启动流程” 步骤。

3、mx Base 推理并不直接处理图片和计算行为识别的准确率。因此我们需要首先执行  “/home/r2p1d” (假设为代码解压目录) 文件夹下的 preprocess.py 对测试图片进行随机缩放、归一化等操作并以 "bin" 文件形式进行存储。

```python
# 此处简要举例，“/home/r2p1d/infer/preprocess_data” 为处理后的数据存储路径，不存在时会自动创建
python3.7 preprocess.py \
--dataset_root_path=/home/r2p1d/infer/ucf101_img \
--output_path=/home/r2p1d/infer/preprocess_data
```

数据预处理完毕后即可执行 “mx Base 推理流程” 步骤。

### 模型转换

1、首先执行 r2plus1d 根目录下的 export.py 将准备好的权重文件转换为 air 模型文件。

```python
# 此处简要举例
python export.py \
--export_batch_size=1 \
--image_height=112 \
--image_width=112 \
--ckpt_file=xxx/r2plus1d_best_map.ckpt \
--file_name=r2plus1d \
--file_format='MINDIR' \
--device_target=Ascend
```

2、然后执行 convert 目录下的 ATC_AIR_2_OM.sh 将刚刚转换好的 air 模型文件转换为 om 模型文件以备后续使用。

```bash
cd /home/r2p1d/infer/convert
bash ATC_AIR_2_OM.sh [model] [output]
e.g.
bash ATC_AIR_2_OM.sh ../r2plus1d.air ../data/model/r2p1d
#转换后的模型结果为r2p1d.om，将放在/home/r2p1d/infer/data/model/目录下
```

### MindX SDK 启动流程

```bash
# 通过 bash 脚本启动 MindX SDK 推理
cd /home/r2p1d/infer/sdk
bash ./run.sh [pipeline] [dataset_root_path]
例如：
bash ./run.sh ../data/config/r2p1d.pipeline /home/r2p1d/infer/ucf101_img/
# 注意： ../data/config/r2p1d.pipeline 中默认 MindX SDK 推理所需的模型文件为 "r2p1d.om"，且放在 ../data/model/ 目录下，具体可以修改该文件中的 "modelPath" 属性进行配置。

# 通过 python 命令启动 MindX SDK 推理
python3.7 main.py \
--pipeline=../data/config/r2p1d.pipeline \
--dataset_root_path=/home/r2p1d/infer/ucf101_img/
# 注意： ../data/config/r2p1d.pipeline 中默认 MindX SDK 推理所需的模型文件为 "r2p1d.om"，且放在 ../data/model/ 目录下，具体可以修改该文件中的 "modelPath" 属性进行配置。
```

**表 1**  参数说明

|        参数        |     说明     |
| ----------------- | ------------ |
| pipeline          | pipeline路径 |
| dataset_root_path | 数据集根目录  |

推理结果示例：

```tex
Begin to initialize Log.
The output directory of logs file exist.
Save logs information to specified directory.
Category:  ApplyEyeMakeup
The predicted category of: v_ApplyEyeMakeup_g01_c02 ---> ApplyEyeMakeup
The predicted category of: v_ApplyEyeMakeup_g03_c02 ---> ApplyEyeMakeup
The predicted category of: v_ApplyEyeMakeup_g03_c03 ---> ApplyEyeMakeup
The predicted category of: v_ApplyEyeMakeup_g04_c05 ---> ApplyEyeMakeup
The predicted category of: v_ApplyEyeMakeup_g04_c07 ---> ApplyEyeMakeup
The predicted category of: v_ApplyEyeMakeup_g05_c02 ---> ApplyEyeMakeup
The predicted category of: v_ApplyEyeMakeup_g05_c03 ---> ApplyEyeMakeup
......
The predicted category of: v_YoYo_g20_c04 ---> YoYo
The predicted category of: v_YoYo_g21_c01 ---> YoYo
The predicted category of: v_YoYo_g22_c01 ---> YoYo
The predicted category of: v_YoYo_g22_c03 ---> YoYo
The predicted category of: v_YoYo_g23_c01 ---> YoYo
The predicted category of: v_YoYo_g24_c01 ---> YoYo
The predicted category of: v_YoYo_g25_c04 ---> YoYo
The predicted category of: v_YoYo_g25_c05 ---> YoYo
Accuracy: 0.9774774774774775
```

### mx Base 推理流程

1、编译 mx Base

```bash
cd /home/r2p1d/infer/mxbase
bash build.sh
```

> ![输入图片说明](https://images.gitee.com/uploads/images/2021/0719/172222_3c2963f4_923381.gif "icon-note.gif") **说明：**
> 编译成功后将在该路径（“/home/r2p1d/infer/mxbase”)下生成“build”目录，其中包含了编译好的可执行文件“r2p1d”。“/home/r2p1d/infer/mxbase”目录下的“r2p1d”也可直接运行，二者是一致的。

2、执行 mx Base 推理

```tex
./r2p1d model_path input_data_path output_data_path
# 按顺序传入模型路径、图像路径(“数据集准备”中的数据存储路径 “/home/r2p1d/infer/preprocess_data”,以"/"结尾)、输出路径（需要提前创建）
例如：
 ./r2p1d ../data/model/r2p1d.om /home/r2p1d/infer/preprocess_data/ ./result/
```

**表 1**  参数说明

| 参数             | 说明                                       |
| ---------------- | ------------------------------------------ |
| model_path       | om模型路径                                 |
| input_data_path  | 本节步骤1处理后的图片保存路径              |
| output_data_path | 模型推理完成后的结果保存路径，请提前创建好 |

> ![输入图片说明](https://images.gitee.com/uploads/images/2021/0719/172222_3c2963f4_923381.gif "icon-note.gif") **说明：**
> 视频行为识别的结果将保存在output_data_path参数指定的目录下，仍是以二进制文件的形式保存。如果需要计算行为识别的准确率，请执行 /home/r2p1d 目录下的 postprocess.py 脚本。

```python
# 此处简要举例
python3.7 postprocess.py \
--result_path=/home/r2p1d/infer/mxbase/result
result_path 指 mx Base 推理后的结果保存路径。
```

mx Base 推理结果示例：

如 The predicted category of:82_SkyDivingv_SkyDiving_g14_c04.bin--->SkyDiving 则是将视频 “82_SkyDivingv_SkyDiving_g14_c04.bin” 预测为 SkyDiving 类。

```tex
I1215 14:21:03.449173 11422 main.cpp:53] =======================================  !!!Parameters setting!!!========================================
I1215 14:21:03.449221 11422 main.cpp:56] ==========  loading model weights from: ../data/model/r2p1d.om
I1215 14:21:03.449230 11422 main.cpp:59] ==========  input data path = ../outputs/
I1215 14:21:03.449234 11422 main.cpp:62] ==========  output data path = ./result/ WARNING: please make sure that this folder is created in advance!!!
I1215 14:21:03.449237 11422 main.cpp:65] ========================================  !!!Parameters setting!!! ========================================
I1215 14:21:03.831578 11422 ModelInferenceProcessor.cpp:22] Begin to ModelInferenceProcessor init
I1215 14:21:04.376271 11422 ModelInferenceProcessor.cpp:69] End to ModelInferenceProcessor init
I1215 14:21:04.382838 11422 main.cpp:85] Processing: 1/2664 ---> 82_SkyDivingv_SkyDiving_g14_c04.bin
The predicted category of:82_SkyDivingv_SkyDiving_g14_c04.bin--->SkyDiving
I1215 14:21:04.460315 11422 main.cpp:85] Processing: 2/2664 ---> 90_TaiChiv_TaiChi_g11_c04.bin
The predicted category of:90_TaiChiv_TaiChi_g11_c04.bin--->TaiChi
I1215 14:21:04.536715 11422 main.cpp:85] Processing: 3/2664 ---> 42_HulaHoopv_HulaHoop_g01_c05.bin
The predicted category of:42_HulaHoopv_HulaHoop_g01_c05.bin--->HulaHoop
I1215 14:21:04.612809 11422 main.cpp:85] Processing: 4/2664 ---> 100_YoYov_YoYo_g14_c01.bin
The predicted category of:100_YoYov_YoYo_g14_c01.bin--->YoYo
I1215 14:21:04.688663 11422 main.cpp:85] Processing: 5/2664 ---> 60_PlayingDholv_PlayingDhol_g17_c05.bin
The predicted category of:60_PlayingDholv_PlayingDhol_g17_c05.bin--->PlayingDhol
I1215 14:21:04.764374 11422 main.cpp:85] Processing: 6/2664 ---> 53_Mixingv_Mixing_g04_c06.bin
The predicted category of:53_Mixingv_Mixing_g04_c06.bin--->Mixing
I1215 14:21:04.839991 11422 main.cpp:85] Processing: 7/2664 ---> 56_ParallelBarsv_ParallelBars_g23_c04.bin
The predicted category of:56_ParallelBarsv_ParallelBars_g23_c04.bin--->ParallelBars
I1215 14:21:04.915774 11422 main.cpp:85] Processing: 8/2664 ---> 62_PlayingGuitarv_PlayingGuitar_g18_c01.bin
The predicted category of:62_PlayingGuitarv_PlayingGuitar_g18_c01.bin--->PlayingGuitar
I1215 14:21:04.991299 11422 main.cpp:85] Processing: 9/2664 ---> 94_Typingv_Typing_g09_c02.bin
The predicted category of:94_Typingv_Typing_g09_c02.bin--->Typing
I1215 14:21:05.067085 11422 main.cpp:85] Processing: 10/2664 ---> 67_PoleVaultv_PoleVault_g04_c03.bin
The predicted category of:67_PoleVaultv_PoleVault_g04_c03.bin--->PoleVault
I1215 14:21:05.143180 11422 main.cpp:85] Processing: 11/2664 ---> 74_RopeClimbingv_RopeClimbing_g15_c02.bin
The predicted category of:74_RopeClimbingv_RopeClimbing_g15_c02.bin--->RopeClimbing
I1215 14:21:05.219604 11422 main.cpp:85] Processing: 12/2664 ---> 10_Bikingv_Biking_g05_c05.bin
The predicted category of:10_Bikingv_Biking_g05_c05.bin--->Biking
......
I1215 14:24:28.462285 11422 main.cpp:85] Processing: 2661/2664 ---> 77_ShavingBeardv_ShavingBeard_g16_c06.bin
The predicted category of:77_ShavingBeardv_ShavingBeard_g16_c06.bin--->ShavingBeard
I1215 14:24:28.539448 11422 main.cpp:85] Processing: 2662/2664 ---> 15_Bowlingv_Bowling_g24_c07.bin
The predicted category of:15_Bowlingv_Bowling_g24_c07.bin--->Bowling
I1215 14:24:28.616636 11422 main.cpp:85] Processing: 2663/2664 ---> 70_Punchv_Punch_g24_c02.bin
The predicted category of:70_Punchv_Punch_g24_c02.bin--->Punch
I1215 14:24:28.693912 11422 main.cpp:85] Processing: 2664/2664 ---> 52_MilitaryParadev_MilitaryParade_g10_c02.bin
The predicted category of:52_MilitaryParadev_MilitaryParade_g10_c02.bin--->MilitaryParade
I1215 14:24:28.771168 11422 main.cpp:94] infer succeed and write the result data with binary file !
I1215 14:24:28.989598 11422 DeviceManager.cpp:83] DestroyDevices begin
I1215 14:24:28.989636 11422 DeviceManager.cpp:85] destroy device:0
I1215 14:24:29.324473 11422 DeviceManager.cpp:91] aclrtDestroyContext successfully!
I1215 14:24:31.227154 11422 DeviceManager.cpp:99] DestroyDevices successfully
I1215 14:24:31.227205 11422 main.cpp:101] Infer images sum 2664, cost total time: 193126 ms.
I1215 14:24:31.227236 11422 main.cpp:102] The throughput: 13.7941 bin/sec.
I1215 14:24:31.227241 11422 main.cpp:103] ==========  The infer result has been saved in ---> ./result/
```

行为识别准确率计算结果示例：

```tex
Accuracy: 0.9774774774774775
```
