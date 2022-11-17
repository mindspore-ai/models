# EAST MindX推理及mx Base推理

<!-- TOC -->

- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [准备工作](#准备工作)
    - [模型转换](#模型转换)
    - [MindX SDK 启动流程](#mindx-sdk-启动流程)
    - [mx Base 推理流程](#mx-base-推理流程)

<!-- /TOC -->

## 脚本说明

### 脚本及样例代码

```tex
├── east
    ├── README.md                           // east 的说明文件
    ├── infer
        ├── README_CN.md                       // east 的 MindX SDK 推理及 mx Base 推理的说明文件
        ├── convert
        │   ├──act_model_convert.sh                 // 将 air 模型转换为 om 模型的脚本
        │   ├──aipp.config                      // aipp 配置文件
        ├── data
        │   ├──image                           // 用于推理的图片
        │   ├──models                          // 转换后的模型文件
        ├── mxbase                             // mx Base 推理目录（C++）
        │   ├──results                         //推理结果存放目录
        │   ├──src
        │   │   ├──PostProcess                                  // 后处理目录
        │   │   │   ├──EASTMindSporePost.cpp                   // 后处理文件
        │   │   │   ├──EASTMindSporePost.h                    // 后处理头文件
        │   │   │   ├──lanmsUtils.h                               // nms头文件
        │   │   │   ├──lanmsUtils.cpp                               // nms文件
        │   │   │   ├──clipper                              // 第三方库文件目录
        │   │   ├──EASTDetection.h                    // 头文件
        │   │   ├──EASTDetection.cpp                  // 详细实现
        │   │   ├──main.cpp                    // mx Base 主函数
        │   ├──build.sh                        // 代码编译脚本
        │   ├──CMakeLists.txt                  // 代码编译设置
        ├── sdk
        │   ├──eavl                            // 精度计算目录
        │   ├──mxpi                            // 后处理插件编译目录
        │   │   ├──build.sh                        // 代码编译脚本
        │   │   ├──CMakeLists.txt                  // 代码编译设置
        │   ├──pipeline                            // 流程管理目录
        │   │   ├──east.pipeline                  // 流程配置文件
        │   ├──result                            // 推理结果保存路径
        │   ├──main.py                         // MindX SDK 运行脚本
        │   ├──run.sh                          // 启动 main.py 的 sh 文件
    ├── ......                                 // 其他代码文件
```

### 准备工作

1、下载需要用到的后处理第三方库文件下载地址[lanms](#https://github.com/argman/EAST), 下载这个仓库里面lanms/include/目录下的clipper文件，然后把这个文件放到对应得mxbase推理的后处理目录下。目录结构如下所示：

```tex
├──PostProcess                                  // 后处理目录
   ├──EASTMindSporePost.cpp                   // 后处理文件
   ├──EASTMindSporePost.h                    // 后处理头文件
   ├──lanms.h                               // nms头文件
   ├──clipper                              // 下载的库文件
```

2、从[here](#https://rrc.cvc.uab.es/?ch=4&com=downloads)下载推理数据集。注意所下载的内容是Task 4.1: Text Localization里面的test数据集。下载完成后把图片解压到data下的image目录，文件结构如下所示：

```tex
├──image                           // 用于推理的图片
    ├──img1.jpg
    ├──.........jpg
```

3、下载[evaluation tool](#https://rrc.cvc.uab.es/?ch=4&com=mymethods&task=1)用于计算精度，注意下载要Evaluation Scripts这个压缩包。下载完成后把这个压缩包解压后的文件放入到eval目录下。最后的文件结构如下所示：

```tex
├──eavl                           // 精度计算目录
    ├──gt.zip                     //标签文件
    ├──script.py                  //启动文件
    ├──rrc_evaluation_funcs_1_1.py    //精度计算
```

### 模型转换

1、首先执行 easxt目录下的 export.py 将准备好的权重文件转换为 air 模型文件。转换完成之后把air模型文件上传到convert目录下

```python
# 此处简要举例
python export.py \
--batch_size=1 \
--channel=3 \
--image_height=704 \
--image_width=1280 \
--ckpt_file=xxx/east.ckpt \
--file_name=east \
--file_format='MINDIR' \
--device_target=Ascend \
--device_id=0 \
```

2、aipp配置，详细内容请参照[这里](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/51RC2alpha003/infacldevg/atctool/atlasatc_16_0021.html)配置

```tex
aipp_op {
    aipp_mode: static
    input_format: RGB888_U8
    src_image_size_w: 1280
    src_image_size_h: 704
    csc_switch: false
    rbuv_swap_switch: true
    min_chn_0: 127.5
    min_chn_1: 127.5
    min_chn_2: 127.5
    var_reci_chn_0: 0.0078431
    var_reci_chn_1: 0.0078431
    var_reci_chn_2: 0.0078431
}
```

3、然后执行 convert 目录下的 act_model_convert.sh 将刚刚转换好的 air 模型文件转换为 om 模型文件以备后续使用。把转换好的.om模型移动到data/models目录下

```bash
# bash
bash act_model_convert.sh
```

### MindX SDK 启动流程

1、在SDK推理之前首先要进行后处理插件的编译，进入sdk下的mxpi目录执行编译命令

```shell
bash build.sh
```

2、编译成功后到sdk目录执行一下命令的一种。**注意：image_path 只支持传入目录**

```shell
# 通过 bash 脚本启动 MindX SDK 推理
# bash ./run.sh -h 或者 bash ./run.sh --help 可以查看帮助信息
bash ./run.sh

# 通过 python 命令启动 MindX SDK 推理
python3.7 main.py --pipeline=./pipeline/east.pipeline \
                  --image_path=../data/image/ \
                  --result_path=./result
```

推理结果示例：

```tex
Begin to initialize Log.
The output directory of logs file exist.
Save logs information to specified directory.
W1111 02:23:48.410840 100187 PostProcessBase.cpp:113] Get key "labelPath" failed. No postprocess label will be read.
W1111 02:23:48.411120 100187 TextObjectPostProcessBase.cpp:36] [1016][Object, file or other resource doesn't exist] Fail to read CHECK_MODEL_FLAG from config, default is true
img_106.jpg
{'confidence': 40.6416, 'text': '', 'x0': 308.271606, 'x1': 434.742188, 'x2': 440.63089, 'x3': 314.1604, 'y0': 562.562378, 'y1': 541.582458, 'y2': 578.963928, 'y3': 599.944}
{'confidence': 31.9189453, 'text': '', 'x0': 895.227905, 'x1': 952.64447, 'x2': 946.173828, 'x3': 888.757385, 'y0': 356.842, 'y1': 376.62912, 'y2': 396.293823, 'y3': 376.506714}
{'confidence': 28.8994141, 'text': '', 'x0': 950.151123, 'x1': 1005.21, 'x2': 998.957, 'x3': 943.897888, 'y0': 408.857605, 'y1': 431.902, 'y2': 447.486755, 'y3': 424.442139}
{'confidence': 21.8564453, 'text': '', 'x0': 1006.36627, 'x1': 1051.67029, 'x2': 1045.82385, 'x3': 1000.51978, 'y0': 433.338226, 'y1': 451.17746, 'y2': 466.758698, 'y3': 448.919769}
{'confidence': 12.8408203, 'text': '', 'x0': 909.687317, 'x1': 949.671082, 'x2': 941.05011, 'x3': 901.066528, 'y0': 384.727905, 'y1': 401.26355, 'y2': 423.071686, 'y3': 406.536}
{'confidence': 6, 'text': '', 'x0': 950.03418, 'x1': 993.991, 'x2': 986.013428, 'x3': 942.056519, 'y0': 374.645325, 'y1': 390.603149, 'y2': 413.549683, 'y3': 397.59198}
{'confidence': 1.99804688, 'text': '', 'x0': 426.164093, 'x1': 466.883789, 'x2': 468.632812, 'x3': 427.913208, 'y0': 563.343506, 'y1': 556.763794, 'y2': 568.097778, 'y3': 574.677551}
img_453.jpg
{'confidence': 62.8164062, 'text': '', 'x0': 677.410217, 'x1': 706.760681, 'x2': 760.731323, 'x3': 731.381104, 'y0': 205.048019, 'y1': 192.932785, 'y2': 329.775024, 'y3': 341.89035}
{'confidence': 40.7939453, 'text': '', 'x0': 439.500793, 'x1': 587.444397, 'x2': 589.486694, 'x3': 441.542786, 'y0': 172.410126, 'y1': 158.592773, 'y2': 181.46402, 'y3': 195.281067}
{'confidence': 28.8886719, 'text': '', 'x0': 463.8414, 'x1': 580.352173, 'x2': 582.576294, 'x3': 466.065094, 'y0': 149.893875, 'y1': 138.46785, 'y2': 162.129044, 'y3': 173.555069}
{'confidence': 17.9580078, 'text': '', 'x0': 196.948502, 'x1': 281.261597, 'x2': 283.435211, 'x3': 199.122, 'y0': 167.025681, 'y1': 157.331238, 'y2': 177.086136, 'y3': 186.780579}
{'confidence': 10.9892578, 'text': '', 'x0': 373.159393, 'x1': 462.566, 'x2': 468.618591, 'x3': 379.211792, 'y0': 415.599121, 'y1': 387.391052, 'y2': 407.458618, 'y3': 435.666687}
{'confidence': 10.7431641, 'text': '', 'x0': 464.439301, 'x1': 539.860901, 'x2': 545.179199, 'x3': 469.75769, 'y0': 387.505615, 'y1': 364.229584, 'y2': 382.317322, 'y3': 405.593262}
{'confidence': 8.99804688, 'text': '', 'x0': 809.273376, 'x1': 907.554077, 'x2': 920.235229, 'x3': 821.954407, 'y0': 410.712646, 'y1': 361.992767, 'y2': 388.751, 'y3': 437.470978}
{'confidence': 1.97070312, 'text': '', 'x0': 350.595215, 'x1': 374.936096, 'x2': 380.998688, 'x3': 356.657898, 'y0': 423.200439, 'y1': 415.167206, 'y2': 434.410858, 'y3': 442.44397}
{'confidence': 1.88867188, 'text': '', 'x0': 664.795, 'x1': 696.308, 'x2': 734.68512, 'x3': 703.17218, 'y0': 124.782127, 'y1': 114.388451, 'y2': 235.968628, 'y3': 246.362305}
{'confidence': 0.9453125, 'text': '', 'x0': 701.094482, 'x1': 731.254578, 'x2': 781.022705, 'x3': 750.86261, 'y0': 304.947601, 'y1': 292.779816, 'y2': 421.810852, 'y3': 433.978668}
img_253.jpg
...

```

3、计算精度，首先进入到result/目录下执行`zip -q submit.zip *.txt`命令压缩文件，然后把`submit.zip`移动到eval目录下，进入eval目录执行`python3.7 script.py -g=gt.zip -s=submit.zip`命令。得到的精度结果如下:

```tex
Calculated!{"precision": 0.8151016456921588, "recall": 0.810784785748676, "hmean": 0.8129374849143133, "AP": 0}
```

### mx Base 推理流程

1、进入mxbase 目录编译 mxBase

```shell
bash ./build.sh
# 编译后的可执行文件 "east" 将保存在build目录
```

2、执行 mx Base 推理，进入build目录执行

```shell
# 例如：
 ./east --dir /home/data/xdm_mindx/east/infer/data/image/
# 也可以传入单张图片
 ./east --image /home/data/xdm_mindx/east/infer/data/image/img_1.jpg
```

mx Base 推理结果示例：

```tex
I1111 05:56:51.626052 101407 EASTMindSporePost.cpp:211] EASTMindsporePost start to write results.
I1111 05:56:51.626081 101407 EASTMindSporePost.cpp:170] Begin to GetValidDetBoxes.
I1111 05:56:51.636149 101407 EASTMindSporePost.cpp:220] EASTMindsporePost write results succeeded.
I1111 05:56:51.636189 101407 EASTMindSporePost.cpp:204] End to Process EASTPostProcess.
I1111 05:56:51.882103 101407 EASTDetection.cpp:219] topkIndex:1, x0:263, y0:162.112, x1:383.379, y1:188.056, x2:379.33, y2:207.695, x3:258.951, y3:181.751, confidence:79.5898
I1111 05:56:51.882128 101407 EASTDetection.cpp:219] topkIndex:2, x0:785.5, y0:337.433, x1:861.418, y1:331.185, x2:863.054, y2:351.848, x3:787.136, y3:358.096, confidence:13.7822
I1111 05:56:51.882140 101407 EASTDetection.cpp:219] topkIndex:3, x0:596.172, y0:248.123, x1:628.555, y1:249.972, x2:617.488, y2:458.109, x3:585.105, y3:456.261, confidence:7.67285
I1111 05:56:51.882740 101407 main.cpp:100] read image path /home/data/xdm_mindx/east/infer/data/image//img_209.jpg
I1111 05:56:51.994958 101407 EASTMindSporePost.cpp:193] Start to Process EASTPostProcess.
I1111 05:56:51.995965 101407 EASTMindSporePost.cpp:211] EASTMindsporePost start to write results.
I1111 05:56:51.995988 101407 EASTMindSporePost.cpp:170] Begin to GetValidDetBoxes.
I1111 05:56:52.019093 101407 EASTMindSporePost.cpp:220] EASTMindsporePost write results succeeded.
I1111 05:56:52.019115 101407 EASTMindSporePost.cpp:204] End to Process EASTPostProcess.
I1111 05:56:52.019124 101407 EASTDetection.cpp:219] topkIndex:1, x0:1004.55, y0:299.899, x1:1242.32, y1:257.828, x2:1252.08, y2:315.913, x3:1014.31, y3:357.984, confidence:191.563
I1111 05:56:52.019150 101407 EASTDetection.cpp:219] topkIndex:2, x0:484.167, y0:201.98, x1:548.425, y1:209.763, x2:546.084, y2:230.061, x3:481.826, y3:222.278, confidence:32.9082
I1111 05:56:52.019161 101407 EASTDetection.cpp:219] topkIndex:3, x0:583.307, y0:447.53, x1:627.895, y1:450.115, x2:626.642, y2:472.642, x3:582.054, y3:470.056, confidence:26.876
I1111 05:56:52.019172 101407 EASTDetection.cpp:219] topkIndex:4, x0:232.693, y0:171.145, x1:285.5, y1:178.455, x2:282.724, y2:199.516, x3:229.917, y3:192.206, confidence:22.9932
I1111 05:56:52.019184 101407 EASTDetection.cpp:219] topkIndex:5, x0:127.671, y0:157.862, x1:180.407, y1:165.025, x2:177.193, y2:189.812, x3:124.456, y3:182.649, confidence:12.9141
I1111 05:56:52.019196 101407 EASTDetection.cpp:219] topkIndex:6, x0:548.813, y0:210.405, x1:607.722, y1:216.165, x2:605.923, y2:235.468, x3:547.014, y3:229.708, confidence:11.9863
I1111 05:56:52.019204 101407 EASTDetection.cpp:219] topkIndex:7, x0:608.578, y0:219.822, x1:636.833, y1:222.49, x2:635.324, y2:239.207, x3:607.069, y3:236.539, confidence:8.98926
I1111 05:56:52.019215 101407 EASTDetection.cpp:219] topkIndex:8, x0:178.828, y0:165.117, x1:228.741, y1:172.023, x2:225.698, y2:194.963, x3:175.785, y3:188.057, confidence:7.99121
I1111 05:56:52.019227 101407 EASTDetection.cpp:219] topkIndex:9, x0:338.25, y0:485.366, x1:380.249, y1:489.525, x2:378.864, y2:504.246, x3:336.865, y3:500.087, confidence:3.93262
I1111 05:56:52.019237 101407 EASTDetection.cpp:219] topkIndex:10, x0:337.985, y0:470.431, x1:372.175, y1:474.496, x2:370.503, y2:489.214, x3:336.312, y3:485.149, confidence:1.99219
I1111 05:56:52.019873 101407 main.cpp:100] read image path /home/data/xdm_mindx/east/infer/data/image//img_359.jpg
I1111 05:56:52.132947 101407 EASTMindSporePost.cpp:193] Start to Process EASTPostProcess.
I1111 05:56:52.133891 101407 EASTMindSporePost.cpp:211] EASTMindsporePost start to write results.
I1111 05:56:52.133913 101407 EASTMindSporePost.cpp:170] Begin to GetValidDetBoxes.
I1111 05:56:52.149281 101407 EASTMindSporePost.cpp:220] EASTMindsporePost write results succeeded.
I1111 05:56:52.149307 101407 EASTMindSporePost.cpp:204] End to Process EASTPostProcess.
I1111 05:56:52.149322 101407 EASTDetection.cpp:219] topkIndex:1, x0:312.294, y0:206.896, x1:400.949, y1:212.115, x2:399.396, y2:239.679, x3:310.741, y3:234.459, confidence:30.8506
I1111 05:56:52.149348 101407 EASTDetection.cpp:219] topkIndex:2, x0:217.487, y0:198.594, x1:312.641, y1:207.292, x2:310.441, y2:232.737, x3:215.287, y3:224.04, confidence:20.835
I1111 05:56:52.149359 101407 EASTDetection.cpp:219] topkIndex:3, x0:567.676, y0:75.7584, x1:605.072, y1:80.1993, x2:603.072, y2:97.7185, x3:565.676, y3:93.2774, confidence:17.8516
I1111 05:56:52.149371 101407 EASTDetection.cpp:219] topkIndex:4, x0:363.787, y0:274.921, x1:412.805, y1:277.657, x2:412.113, y2:290.609, x3:363.095, y3:287.873, confidence:11.8955
I1111 05:56:52.149382 101407 EASTDetection.cpp:219] topkIndex:5, x0:136.926, y0:205.203, x1:189.642, y1:201.242, x2:190.928, y2:219.163, x3:138.212, y3:223.123, confidence:10.9795
I1111 05:56:52.149392 101407 EASTDetection.cpp:219] topkIndex:6, x0:325.01, y0:273.149, x1:365.931, y1:277.034, x2:364.797, y2:289.503, x3:323.875, y3:285.619, confidence:8.99805
I1111 05:56:52.149402 101407 EASTDetection.cpp:219] topkIndex:7, x0:104.004, y0:208.026, x1:142.921, y1:205.533, x2:144.017, y2:223.546, x3:105.1, y3:226.039, confidence:7.87305
I1111 05:56:52.149413 101407 EASTDetection.cpp:219] topkIndex:8, x0:402.916, y0:217.85, x1:445.805, y1:218.186, x2:445.652, y2:238.64, x3:402.763, y3:238.304, confidence:0.984375
I1111 05:56:52.150015 101407 main.cpp:100] read image path /home/data/xdm_mindx/east/infer/data/image//img_229.jpg
I1111 05:56:52.262957 101407 EASTMindSporePost.cpp:193] Start to Process EASTPostProcess.
I1111 05:56:52.263942 101407 EASTMindSporePost.cpp:211] EASTMindsporePost start to write results.
I1111 05:56:52.263962 101407 EASTMindSporePost.cpp:170] Begin to GetValidDetBoxes.
I1111 05:56:52.281702 101407 EASTMindSporePost.cpp:220] EASTMindsporePost write results succeeded.
I1111 05:56:52.281719 101407 EASTMindSporePost.cpp:204] End to Process EASTPostProcess.
...
111 02:43:06.449703 100345 EASTDetection.cpp:219] topkIndex:1, x0:1040.04, y0:371.62, x1:1248.31, y1:452.319, x2:1225.77, y2:513.321, x3:1017.5, y3:432.623, confidence:253.249
I1111 02:43:06.449728 100345 EASTDetection.cpp:219] topkIndex:2, x0:1065.93, y0:239.424, x1:1251.01, y1:260.472, x2:1244, y2:324.953, x3:1058.93, y3:303.905, confidence:78.918
I1111 02:43:06.449738 100345 EASTDetection.cpp:219] topkIndex:3, x0:955.403, y0:352.865, x1:1039.23, y1:384.983, x2:1022.59, y2:430.332, x3:938.763, y3:398.214, confidence:62.8047
I1111 02:43:06.449748 100345 EASTDetection.cpp:219] topkIndex:4, x0:708.932, y0:155.135, x1:802.985, y1:163.863, x2:800.917, y2:187.103, x3:706.865, y3:178.375, confidence:54.7998
I1111 02:43:06.449759 100345 EASTDetection.cpp:219] topkIndex:5, x0:797.816, y0:165.43, x1:848.87, y1:169.733, x2:846.886, y2:194.352, x3:795.833, y3:190.05, confidence:10.9268
I1111 02:43:06.449767 100345 EASTDetection.cpp:219] topkIndex:6, x0:716.588, y0:581.089, x1:765.467, y1:601.002, x2:756.536, y2:623.94, x3:707.657, y3:604.028, confidence:3.88086
I1111 02:43:06.449777 100345 EASTDetection.cpp:219] topkIndex:7, x0:638.906, y0:538.447, x1:713.984, y1:581.08, x2:701.962, y2:603.22, x3:626.884, y3:560.586, confidence:1.9375
I1111 02:43:06.784284 100345 DeviceManager.cpp:83] DestroyDevices begin
I1111 02:43:06.784310 100345 DeviceManager.cpp:85] destroy device:0
I1111 02:43:06.980532 100345 DeviceManager.cpp:91] aclrtDestroyContext successfully!
I1111 02:43:07.878551 100345 DeviceManager.cpp:99] DestroyDevices successfully
I1111 02:43:07.878587 100345 main.cpp:109] Infer images sum 500, cost total time: 49370.5 ms.
I1111 02:43:07.878602 100345 main.cpp:110] The throughput: 10.1275 images/sec.
```

3、mxbase查看精度的步骤和sdk的一样，这里不再赘述。