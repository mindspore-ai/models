# BRDNet MindX推理及mx Base推理

<!-- TOC -->

- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [模型转换](#模型转换)
    - [MindX SDK 启动流程](#mindx-sdk-启动流程)
    - [mx Base 推理流程](#mx-base-推理流程)

<!-- /TOC -->

## 脚本说明

### 脚本及样例代码

```tex
├── brdnet
    ├── README_CN.md                           // brdnet 的说明文件
    ├── infer
        ├── README_CN.md                       // brdnet 的 MindX SDK 推理及 mx Base 推理的说明文件
        ├── convert
        │   ├──ATC_AIR_2_OM.sh                 // 将 air 模型转换为 om 模型的脚本
        ├── data
        │   ├──config
        │   │   ├──brdnet.pipeline             // MindX SDK运行所需的 pipline 配置文件
        ├── mxbase                             // mx Base 推理目录（C++）
        │   ├──src
        │   │   ├──BRDNet.h                    // 头文件
        │   │   ├──BRDNet.cpp                  // 详细实现
        │   │   ├──main.cpp                    // mx Base 主函数
        │   ├──build.sh                        // 代码编译脚本
        │   ├──CMakeLists.txt                  // 代码编译设置
        ├── sdk
        │   ├──main.py                         // MindX SDK 运行脚本
        │   ├──run.sh                          // 启动 main.py 的 sh 文件
    ├── ......                                 // 其他代码文件
```

### 模型转换

1、首先执行 brdnet 目录下的 export.py 将准备好的权重文件转换为 air 模型文件。

```python
# 此处简要举例
python export.py \
--batch_size=1 \
--channel=3 \
--image_height=500 \
--image_width=500 \
--ckpt_file=xxx/brdnet.ckpt \
--file_name=brdnet \
--file_format='AIR' \
--device_target=Ascend \
--device_id=0 \
```

2、然后执行 convert 目录下的 ATC_AIR_2_OM.sh 将刚刚转换好的 air 模型文件转换为 om 模型文件以备后续使用。

```bash
# bash ./ATC_AIR_2_OM.sh -h 或者 bash ./ATC_AIR_2_OM.sh --help 可以查看帮助信息
bash ATC_AIR_2_OM.sh [--model --output --soc_version --input_shape]
```

### MindX SDK 启动流程

```shell
# 通过 bash 脚本启动 MindX SDK 推理
# bash ./run.sh -h 或者 bash ./run.sh --help 可以查看帮助信息
bash ./run.sh [--pipeline --clean_image_path --image_width --image_height --channel --sigma]
# 注意： data/config/brdnet.pipeline 中默认 MindX SDK 推理所需的模型文件为 "channel_3_sigma_15.om"，且放在 data/model/ 目录下，具体可以修改该文件中的 "modelPath" 属性进行配置。

# 通过 python 命令启动 MindX SDK 推理
python main.py \
--pipeline=../data/config/brdnet.pipeline \
--clean_image_path=../Test/Kodak24 \
--image_width=500 \
--image_height=500 \
--channel=3 \
--sigma=15
# 注意： data/config/brdnet.pipeline 中默认 MindX SDK 推理所需的模型文件为 "channel_3_sigma_15.om"，且放在 data/model/ 目录下，具体可以修改该文件中的 "modelPath" 属性进行配置。
```

推理结果示例：

```tex
Begin to initialize Log.
The output directory of logs file exist.
Save logs information to specified directory.
Denosing image: ../../BRDNet/Test_dataset/Kodak24/kodim01.png
../../BRDNet/Test_dataset/Kodak24/kodim01.png : psnr_denoised:    32.28033733366724
Denosing image: ../../BRDNet/Test_dataset/Kodak24/kodim02.png
../../BRDNet/Test_dataset/Kodak24/kodim02.png : psnr_denoised:    35.018032807200164
Denosing image: ../../BRDNet/Test_dataset/Kodak24/kodim03.png
../../BRDNet/Test_dataset/Kodak24/kodim03.png : psnr_denoised:    37.80273057933442
Denosing image: ../../BRDNet/Test_dataset/Kodak24/kodim04.png
../../BRDNet/Test_dataset/Kodak24/kodim04.png : psnr_denoised:    35.60892146774801
Denosing image: ../../BRDNet/Test_dataset/Kodak24/kodim05.png
../../BRDNet/Test_dataset/Kodak24/kodim05.png : psnr_denoised:    33.336266095083175
Denosing image: ../../BRDNet/Test_dataset/Kodak24/kodim06.png
../../BRDNet/Test_dataset/Kodak24/kodim06.png : psnr_denoised:    33.738780427944974
Denosing image: ../../BRDNet/Test_dataset/Kodak24/kodim07.png
../../BRDNet/Test_dataset/Kodak24/kodim07.png : psnr_denoised:    37.10481992981783
Denosing image: ../../BRDNet/Test_dataset/Kodak24/kodim08.png
../../BRDNet/Test_dataset/Kodak24/kodim08.png : psnr_denoised:    33.126510144521
Denosing image: ../../BRDNet/Test_dataset/Kodak24/kodim09.png
../../BRDNet/Test_dataset/Kodak24/kodim09.png : psnr_denoised:    37.23759544848104
Denosing image: ../../BRDNet/Test_dataset/Kodak24/kodim10.png
../../BRDNet/Test_dataset/Kodak24/kodim10.png : psnr_denoised:    36.954513882215366
Denosing image: ../../BRDNet/Test_dataset/Kodak24/kodim11.png
../../BRDNet/Test_dataset/Kodak24/kodim11.png : psnr_denoised:    33.961228532687855
Denosing image: ../../BRDNet/Test_dataset/Kodak24/kodim12.png
../../BRDNet/Test_dataset/Kodak24/kodim12.png : psnr_denoised:    35.90416546419448
Denosing image: ../../BRDNet/Test_dataset/Kodak24/kodim13.png
../../BRDNet/Test_dataset/Kodak24/kodim13.png : psnr_denoised:    31.199819472546324
Denosing image: ../../BRDNet/Test_dataset/Kodak24/kodim14.png
../../BRDNet/Test_dataset/Kodak24/kodim14.png : psnr_denoised:    33.176099577560706
Denosing image: ../../BRDNet/Test_dataset/Kodak24/kodim15.png
../../BRDNet/Test_dataset/Kodak24/kodim15.png : psnr_denoised:    34.62721601846573
Denosing image: ../../BRDNet/Test_dataset/Kodak24/kodim16.png
../../BRDNet/Test_dataset/Kodak24/kodim16.png : psnr_denoised:    35.10364930038219
Denosing image: ../../BRDNet/Test_dataset/Kodak24/kodim17.png
../../BRDNet/Test_dataset/Kodak24/kodim17.png : psnr_denoised:    35.14010929192525
Denosing image: ../../BRDNet/Test_dataset/Kodak24/kodim18.png
../../BRDNet/Test_dataset/Kodak24/kodim18.png : psnr_denoised:    33.19858405097709
Denosing image: ../../BRDNet/Test_dataset/Kodak24/kodim19.png
../../BRDNet/Test_dataset/Kodak24/kodim19.png : psnr_denoised:    34.92669369486534
Denosing image: ../../BRDNet/Test_dataset/Kodak24/kodim20.png
../../BRDNet/Test_dataset/Kodak24/kodim20.png : psnr_denoised:    36.63018276423998
Denosing image: ../../BRDNet/Test_dataset/Kodak24/kodim21.png
../../BRDNet/Test_dataset/Kodak24/kodim21.png : psnr_denoised:    34.170664959208786
Denosing image: ../../BRDNet/Test_dataset/Kodak24/kodim22.png
../../BRDNet/Test_dataset/Kodak24/kodim22.png : psnr_denoised:    34.182998599615985
Denosing image: ../../BRDNet/Test_dataset/Kodak24/kodim23.png
../../BRDNet/Test_dataset/Kodak24/kodim23.png : psnr_denoised:    36.84838635349549
Denosing image: ../../BRDNet/Test_dataset/Kodak24/kodim24.png
../../BRDNet/Test_dataset/Kodak24/kodim24.png : psnr_denoised:    34.334034305678436
Average PSNR: 34.81718085424403
Testing finished....
=======================================
The total time of inference is 2.731436252593994 s
=======================================
```

### mx Base 推理流程

mx Base 推理并不直接处理图片和计算去噪后的 PSNR 值。

1、首先执行 brdnet 目录下的 preprocess.py 为测试图片添加噪声并以 "bin" 文件形式进行存储

```python
# 此处简要举例
python preprocess.py \
--out_dir=xx/noise_data_Kodak24 \
--image_path=xx/Kodak24 \
--image_height=500 \
--image_width=500 \
--channel=3 \
--sigma=15
```

2、编译 mx Base

```shell
bash ./build.sh
# 编译后的可执行文件 "brdnet" 将保存在当前目录下
```

3、执行 mx Base 推理

```tex
./brdnet [model_path input_data_path output_data_path]
# 按顺序传入模型路径、噪声图像路径、输出路径（需要提前创建）
例如：
 ./brdnet ../data/model/channel_3_sigma_15.om ../noise_data_Kodak24/ ./result/
```

mx Base 推理结果示例：

```tex
I1106 10:09:26.438470 79291 main.cpp:53] =======================================  !!!Parameters setting!!! ========================================
I1106 10:09:26.438516 79291 main.cpp:55] ==========  loading model weights from: ../data/model/channel_3_sigma_15.om
I1106 10:09:26.438527 79291 main.cpp:58] ==========  input data path = ../noise_data_Kodak24/
I1106 10:09:26.438536 79291 main.cpp:61] ==========  output data path = ./result/ WARNING: please make sure that this folder is created in advance!!!
I1106 10:09:26.438544 79291 main.cpp:63] ========================================  !!!Parameters setting!!! ========================================
I1106 10:09:26.798825 79291 ModelInferenceProcessor.cpp:22] Begin to ModelInferenceProcessor init
I1106 10:09:26.863025 79291 ModelInferenceProcessor.cpp:69] End to ModelInferenceProcessor init
I1106 10:09:26.863147 79291 main.cpp:82] Processing: 1/24 ---> kodim24_noise.bin
I1106 10:09:26.980234 79291 main.cpp:82] Processing: 2/24 ---> kodim18_noise.bin
I1106 10:09:27.096143 79291 main.cpp:82] Processing: 3/24 ---> kodim07_noise.bin
I1106 10:09:27.213531 79291 main.cpp:82] Processing: 4/24 ---> kodim19_noise.bin
I1106 10:09:27.328680 79291 main.cpp:82] Processing: 5/24 ---> kodim02_noise.bin
I1106 10:09:27.444927 79291 main.cpp:82] Processing: 6/24 ---> kodim20_noise.bin
I1106 10:09:27.558817 79291 main.cpp:82] Processing: 7/24 ---> kodim12_noise.bin
I1106 10:09:27.675061 79291 main.cpp:82] Processing: 8/24 ---> kodim21_noise.bin
I1106 10:09:27.791473 79291 main.cpp:82] Processing: 9/24 ---> kodim14_noise.bin
I1106 10:09:27.906719 79291 main.cpp:82] Processing: 10/24 ---> kodim16_noise.bin
I1106 10:09:28.023947 79291 main.cpp:82] Processing: 11/24 ---> kodim01_noise.bin
I1106 10:09:28.140027 79291 main.cpp:82] Processing: 12/24 ---> kodim23_noise.bin
I1106 10:09:28.255630 79291 main.cpp:82] Processing: 13/24 ---> kodim17_noise.bin
I1106 10:09:28.369719 79291 main.cpp:82] Processing: 14/24 ---> kodim05_noise.bin
I1106 10:09:28.485267 79291 main.cpp:82] Processing: 15/24 ---> kodim22_noise.bin
I1106 10:09:28.600522 79291 main.cpp:82] Processing: 16/24 ---> kodim13_noise.bin
I1106 10:09:28.716308 79291 main.cpp:82] Processing: 17/24 ---> kodim09_noise.bin
I1106 10:09:28.830880 79291 main.cpp:82] Processing: 18/24 ---> kodim06_noise.bin
I1106 10:09:28.945564 79291 main.cpp:82] Processing: 19/24 ---> kodim03_noise.bin
I1106 10:09:29.061424 79291 main.cpp:82] Processing: 20/24 ---> kodim04_noise.bin
I1106 10:09:29.176980 79291 main.cpp:82] Processing: 21/24 ---> kodim10_noise.bin
I1106 10:09:29.292285 79291 main.cpp:82] Processing: 22/24 ---> kodim11_noise.bin
I1106 10:09:29.406962 79291 main.cpp:82] Processing: 23/24 ---> kodim15_noise.bin
I1106 10:09:29.521801 79291 main.cpp:82] Processing: 24/24 ---> kodim08_noise.bin
I1106 10:09:29.637691 79291 main.cpp:91] infer succeed and write the result data with binary file !
I1106 10:09:29.771848 79291 DeviceManager.cpp:83] DestroyDevices begin
I1106 10:09:29.771868 79291 DeviceManager.cpp:85] destroy device:0
I1106 10:09:29.954421 79291 DeviceManager.cpp:91] aclrtDestroyContext successfully!
I1106 10:09:31.532470 79291 DeviceManager.cpp:99] DestroyDevices successfully
I1106 10:09:31.532511 79291 main.cpp:98] Infer images sum 24, cost total time: 2535.4 ms.
I1106 10:09:31.532536 79291 main.cpp:99] The throughput: 9.46598 bin/sec.
I1106 10:09:31.532541 79291 main.cpp:100] ==========  The infer result has been saved in ---> ./result/
```

mx Base 的推理结果为 "去噪后的图片"，将以 "bin" 文件的形式存储在指定路径下。

4、计算去噪后的 PSNR 值

如果需要计算去噪后的 PSNR 值，请执行 brdnet 目录下的 cal_psnr.py 脚本

```python
# 此处简要举例
python cal_psnr.py \
--image_path=xx/Kodak24 \
--output_path=xx/Kodak24 \
--image_height=500 \
--image_width=500 \
--channel=3
image_path 指原不含噪声的图片路径，output_path 指 mx Base 推理后的结果保存路径
```

PSNR 计算结果示例：

```tex
kodim01 : psnr_denoised:    32.28033733366724
kodim02 : psnr_denoised:    35.018032807200164
kodim03 : psnr_denoised:    37.80273057933442
kodim04 : psnr_denoised:    35.60892146774801
kodim05 : psnr_denoised:    33.336266095083175
kodim06 : psnr_denoised:    33.738780427944974
kodim07 : psnr_denoised:    37.10481992981783
kodim08 : psnr_denoised:    33.126510144521
kodim09 : psnr_denoised:    37.23759544848104
kodim10 : psnr_denoised:    36.954513882215366
kodim11 : psnr_denoised:    33.961228532687855
kodim12 : psnr_denoised:    35.90416546419448
kodim13 : psnr_denoised:    31.199819472546324
kodim14 : psnr_denoised:    33.176099577560706
kodim15 : psnr_denoised:    34.62721601846573
kodim16 : psnr_denoised:    35.10364930038219
kodim17 : psnr_denoised:    35.14010929192525
kodim18 : psnr_denoised:    33.19858405097709
kodim19 : psnr_denoised:    34.92669369486534
kodim20 : psnr_denoised:    36.63018276423998
kodim21 : psnr_denoised:    34.170664959208786
kodim22 : psnr_denoised:    34.182998599615985
kodim23 : psnr_denoised:    36.84838635349549
kodim24 : psnr_denoised:    34.334034305678436
Average PSNR: 34.81718085424403
Testing finished....
Time cost:1.0340161323547363 seconds!
```
