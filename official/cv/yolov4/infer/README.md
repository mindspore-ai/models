# 交付件基本信息

**发布者（Publisher）**：Huawei

**应用领域（Application Domain）**：Object Detection

**版本（Version）**：1.1

**修改时间（Modified）**：2022.3.29

**大小（Size）**：251.52 MB (air) / 126.24 MB (om) / 503.62 MB (ckpt)

**框架（Framework）**：MindSpore\_1.3.0

**模型格式（Model Format）**：ckpt/air/om

**精度（Precision）**：Mixed/FP16

**处理器（Processor）**：昇腾910/昇腾310

**应用级别（Categories）**：Released

**描述（Description）**：基于MindSpore框架的YOLOv4网络模型训练并保存模型，通过ATC工具转换，可在昇腾AI设备上运行，支持使用MindX SDK及MxBase进行推理

# 概述

## 简述

YOLOv4作为先进的检测器，它比所有可用的替代检测器更快（FPS）并且更准确（MS COCO AP50 ... 95和AP50）。

本文已经验证了大量的特征，并选择使用这些特征来提高分类和检测的精度。

这些特性可以作为未来研究和开发的最佳实践。

* [参考论文](https://arxiv.org/pdf/2004.10934.pdf): Bochkovskiy A, Wang C Y, Liao H Y M. YOLOv4: Optimal Speed and Accuracy of Object Detection[J]. arXiv preprint arXiv:2004.10934, 2020.

通过Git获取对应commit_id的代码方法如下：

```shell
git clone {repository_url}     # 克隆仓库的代码
cd {repository_name}           # 切换到模型的代码仓目录
git checkout  {branch}         # 切换到对应分支
git reset --hard ｛commit_id｝  # 代码设置到对应的commit_id
cd ｛code_path｝                # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
```

## 默认配置

1. 网络结构  

选择CSPDarknet53主干、SPP附加模块、PANet路径聚合网络和YOLOv4（基于锚点）头作为YOLOv4架构。  

2. 预训练模型  

YOLOv4需要CSPDarknet53主干来提取图像特征进行检测。  

可以从[这里](https://gitee.com/link?target=https%3A%2F%2Fdownload.mindspore.cn%2Fmodel_zoo%2Fr1.2%2Fcspdarknet53_ascend_v120_imagenet2012_official_cv_bs64_top1acc7854_top5acc9428%2Fcspdarknet53_ascend_v120_imagenet2012_official_cv_bs64_top1acc7854_top5acc9428.ckpt) 获取到在ImageNet2012上训练的预训练模型。

3. 训练参数

```SHELL
lr_scheduler：cosine_annealing
lr：0.1
training_shape: 416
max_epochs：320
warmup_epochs: 4
```

## 支持特性

### 支持特性

支持的特性包括：1、分布式并行训练。2、混合精度训练。

### 分布式并行训练

MindSpore支持数据并行及自动并行。自动并行是MindSpore融合了数据并行、模型并行及混合并行的一种分布式并行模式，可以自动建立代价模型，为用户选择一种并行模式。相关代码示例。

```shell
context.set_auto_parallel_context(parallel_mode = ParallelMode.DATA_PARALLEL, device_num = device_num)
```

### 混合精度训练

混合精度训练方法是通过混合使用单精度和半精度数据格式来加速深度神经网络训练的过程，同时保持了单精度训练所能达到的网络精度。混合精度训练能够加速计算过程，同时减少内存使用和存取，并使得在特定的硬件上可以训练更大的模型或batch size。

对于FP16的算子，若给定的数据类型是FP32，MindSpore框架的后端会进行降精度处理。用户可以开启INFO日志，并通过搜索关键字“Reduce precision”查看降精度处理的算子。

# 准备工作

## 训练环境准备

1. 硬件环境准备请参见各硬件产品[“驱动和固件安装升级指南”](https://support.huawei.com/enterprise/zh/category/ai-computing-platform-pid-1557196528909) 。需要在硬件设备上安装与CANN版本配套的固件与驱动。

2. 宿主机上需要安装Python3和Docker，并登录[Ascend Hub中心](https://ascend.huawei.com/ascendhub/#/home) 获取镜像。

   当前模型支持的镜像列表如下表所示。  
   **表 1** 镜像列表  

    | 镜像名称 | 镜像版本 | 配套CANN版本 |  
    | ------- | ------------ | --------------------- |  
    | ARM/x86架构：[mindspore-modelzoo](https://ascendhub.huawei.com/#/detail/mindspore-modelzoo) | 21.0.4   | [5.0.2](https://www.hiascend.com/software/cann/commercial)  |  

## 推理环境准备

1. 硬件环境、开发环境和运行环境准备请参见[《CANN 软件安装指南》](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=installation-upgrade) 。

2. 宿主机上需要安装Docker并登录[Ascend Hub中心](https://ascendhub.huawei.com/#/home) 获取镜像。

   当前模型支持的镜像列表如下表所示。

   **表 1** 镜像列表

   | 镜像名称 | 镜像版本 | 配套CANN版本 |  
   | ------- | ------------ | --------------------- |  
   | ARM/x86架构：[infer-modelzoo](https://ascendhub.huawei.com/#/detail/infer-modelzoo) | 21.0.4   | [5.0.2](https://www.hiascend.com/software/cann/commercial) |

## 源码介绍

1. 脚本目录结构如下：

 ```shell
 infer
 ├──README.md              # 离线推理文档  
 ├──convert  
 │    ├──aipp.config       # aipp配置文件  
 │    └──air2om.sh         # om模型转换脚本  
 │──data  
 │    ├──models            # 模型文件  
 │    │   ├──yolov4_coco2017_acc_Test.cfg  #infer的超参数设置  
 │    │   ├──yolov4.om     # 生成的om模型  
 │    │   ├──yolov4.air    # modelarts训练后生成的air模型  
 │    │   ├──trainval.txt  # 为infer准备的验证数据
 │    │   ├──object_task_metric.py  # 将infer后的结果转换为coco模式
 │    │   └──coco2017.names  # coco数据集样本的label
 │    └──images            # 模型输入数据集, 将数据集中的val2017中的内容拷贝过来
 │───mxbase                # 基于mxbase推理脚本
 │    ├──src
 │    │   ├──PostProcess   # 前处理
 │    │   │   ├──Yolov4MindsporePost.cpp
 │    │   │   └──Yolov4MindsporePost.h
 │    │   ├──Yolov4Detection.h
 │    │   ├──Yolov4Detection.cpp
 │    │   └──main.cpp
 │    ├──CMakeLists.txt  
 │    ├──build.sh          # 编译
 │    └──infermxbase.sh    # 验证推理结果精度
 │──sdk                    # 基于sdk包推理脚本
 │    ├──mxpi
 │    │   ├──CMakeLists.txt
 │    │   └──build.sh  
 │    ├──config
 │    │   └──yolov4.pipeline
 │    ├──run.sh  
 │    └──infersdk.sh      # 验证推理结果精度
 └── docker_start_infer.sh # 启动容器脚本
 ```

# 训练

## 数据集准备

1. 请用户自行准备好数据集，使用的数据集：[COCO2017](https://gitee.com/link?target=https%3A%2F%2Fcocodataset.org%2F%23download)  

* 支持的数据集：[COCO2017](https://gitee.com/link?target=https%3A%2F%2Fcocodataset.org%2F%23download) 或与MS COCO格式相同的数据集  

* 支持的标注：[COCO2017](https://gitee.com/link?target=https%3A%2F%2Fcocodataset.org%2F%23download) 或与MS COCO相同格式的标注

2. 数据准备

* 将数据集放到任意路径，文件夹应该包含如下文件

      ```SHELL
      .
      └── datasets
        ├── annotations
          │   ├─ instances_train2017.json
          │   └─ instances_val2017.json
          ├─ train2017  
          │   ├─picture1.jpg
          │   ├─ ...
          │   └─picturen.jpg
          ├─ val2017
              ├─picture1.jpg
              ├─ ...
              └─picturen.jpg
      ```

* 为数据集生成TXT格式推理文件。

      ```shell
      # 导出txt推理数据
      python coco_trainval_anns.py --data_url=./datasets/ --train_url=./infer/data/models/ --val_url=./infer/data/images/
      #data_url参数为数据集datasets存储路径，train_url参数为存储txt路径，val_url参数为推理数据集存放的路径
      ```

      每行如下所示：

      ```  SHELL
      0 ../infer/data/images/000000289343.jpg 529 640 16 473 395 511 423 0 204 235 264 412 13 0 499 339 605 1 204 304 256 456
      ```  

      每行是按空间分割的图像标注，第一列数是序号，第二列是推理使用的图像的绝对路径，其余为[xmin,ymin,xmax,ymax,class]格式的框和类信息。

## 高级参考

### 脚本参数

1. 训练和测试部分重要参数如下：

   ```SHELL
   usage: modelarts.py  [--data_url DATA_URL] [--train_url TRAIN_URL] [--checkpoint_url CHECKPOINT_URL]  
   options:
      --train_url    The path model saved
      --data_url   Dataset directory
      --checkpoint_url   The path pre-model saved
   ```

2. 参数意义如下：

   ```SHELL
    # Train options
    data_dir: "Train dataset directory."
    per_batch_size: "Batch size for Training."
    pretrained_backbone: "The ckpt file of CspDarkNet53."
    resume_yolov4: "The ckpt file of YOLOv4, which used to fine tune."
    pretrained_checkpoint: "The ckpt file of YoloV4CspDarkNet53."
    filter_weight: "Filter the last weight parameters"
    lr_scheduler: "Learning rate scheduler, options: exponential, cosine_annealing."
    lr: "Learning rate."
    lr_epochs: "Epoch of changing of lr changing, split with ','."
    lr_gamma: "Decrease lr by a factor of exponential lr_scheduler."
    eta_min: "Eta_min in cosine_annealing scheduler."
    t_max: "T-max in cosine_annealing scheduler."
    max_epoch: "Max epoch num to train the model."
    warmup_epochs: "Warmup epochs."
    weight_decay: "Weight decay factor."
    momentum: "Momentum."
    loss_scale: "Static loss scale."
    label_smooth: "Whether to use label smooth in CE."
    label_smooth_factor: "Smooth strength of original one-hot."
    log_interval: "Logging interval steps."
    ckpt_path: "Checkpoint save location."
    ckpt_interval: "Save checkpoint interval."
    is_save_on_master: "Save ckpt on master or all rank, 1 for master, 0 for all ranks."
    is_distributed: "Distribute train or not, 1 for yes, 0 for no."
    rank: "Local rank of distributed."
    group_size: "World size of device."
    need_profiler: "Whether use profiler. 0 for no, 1 for yes."
    training_shape: "Fix training shape."
    resize_rate: "Resize rate for multi-scale training."
    run_eval: "Run evaluation when training."
    save_best_ckpt: "Save best checkpoint when run_eval is True."
    eval_start_epoch: "Evaluation start epoch when run_eval is True."
    eval_interval: "Evaluation interval when run_eval is True"
    ann_file: "path to annotation"
    each_multiscale: "Apply multi-scale for each scale"
    detect_head_loss_coff: "the loss coefficient of detect head.
                           The order of coefficients is large head, medium head and small head"
    bbox_class_loss_coff: "bbox and class loss coefficient.
                           The order of coefficients is ciou loss, confidence loss and class loss"
    labels: "the label of train data"
    mosaic: "use mosaic data augment"
    multi_label: "use multi label to nms"
    multi_label_thresh: "multi label thresh"

    # Eval options
    pretrained: "model_path, local pretrained model to load"
    log_path: "checkpoint save location"
    ann_val_file: "path to annotation"

    # Export options
    device_id: "Device id for export"
    batch_size: "batch size for export"
    testing_shape: "shape for test"
    ckpt_file: "Checkpoint file path for export"
    file_name: "output file name for export"
    file_format: "file format for export"
    keep_detect: "keep the detect module or not, default: True"
    img_id_file_path: 'path of image dataset'
    result_files: 'path to 310 infer result floder'
   ```

# 推理

## 准备推理数据

1. 下载源码包。

   单击“下载模型脚本”和“下载模型”，并下载所需MindX SDK开发套件（mxManufacture）。

2. 将源码上传至推理服务器任意目录并解压（如：“/home/data/wwq“）。

3. 编译镜像。

   **docker build -t** *infer_image* **--build-arg FROM_IMAGE_NAME=** *base_image:tag* **--build-arg SDK_PKG=** *sdk_pkg* **.**

   **表 1**  参数说明

   <table><thead align="left"><tr id="zh-cn_topic_0304403934_row9243114772414"><th class="cellrowborder" valign="top" width="40%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0304403934_p524364716241"><a name="zh-cn_topic_0304403934_p524364716241"></a><a name="zh-cn_topic_0304403934_p524364716241"></a>参数</p>
   </th>
   <th class="cellrowborder" valign="top" width="60%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0304403934_p172431247182412"><a name="zh-cn_topic_0304403934_p172431247182412"></a><a name="zh-cn_topic_0304403934_p172431247182412"></a>说明</p>
   </th>
   </tr>
   </thead>
   <tbody><tr id="zh-cn_topic_0304403934_row52431473244"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p144312172333"><a name="p144312172333"></a><a name="p144312172333"></a><em id="i290520133315"><a name="i290520133315"></a><a name="i290520133315"></a>infer_image</em></p>
   </td>
   <td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0304403934_p10243144712410"><a name="zh-cn_topic_0304403934_p10243144712410"></a><a name="zh-cn_topic_0304403934_p10243144712410"></a>推理镜像名称，根据实际写入。</p>
   </td>
   </tr>
   <tr id="zh-cn_topic_0304403934_row1624394732415"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0304403934_p92434478242"><a name="zh-cn_topic_0304403934_p92434478242"></a><a name="zh-cn_topic_0304403934_p92434478242"></a><em id="i78645182347"><a name="i78645182347"></a><a name="i78645182347"></a>base_image</em></p>
   </td>
   <td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0304403934_p324313472240"><a name="zh-cn_topic_0304403934_p324313472240"></a><a name="zh-cn_topic_0304403934_p324313472240"></a>基础镜像，可从Ascend Hub上下载。</p>
   </td>
   </tr>
   <tr id="row2523459163416"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p55241359203412"><a name="p55241359203412"></a><a name="p55241359203412"></a><em id="i194517711355"><a name="i194517711355"></a><a name="i194517711355"></a>tag</em></p>
   </td>
   <td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1952435919341"><a name="p1952435919341"></a><a name="p1952435919341"></a>镜像tag，请根据实际配置，如：21.0.1。</p>
   </td>
   </tr>
   <tr id="zh-cn_topic_0304403934_row132436473240"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0304403934_p1824319472242"><a name="zh-cn_topic_0304403934_p1824319472242"></a><a name="zh-cn_topic_0304403934_p1824319472242"></a>sdk_pkg</p>
   </td>
   <td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0304403934_p7243144712249"><a name="zh-cn_topic_0304403934_p7243144712249"></a><a name="zh-cn_topic_0304403934_p7243144712249"></a>下载的mxManufacture包名称，如Ascend-mindxsdk-mxmanufacture_<em id="i061383054119"><a name="i061383054119"></a><a name="i061383054119"></a>{version}</em>_linux-<em id="i1055956194514"><a name="i1055956194514"></a><a name="i1055956194514"></a>{arch}</em>.run。</p>
   </td>
   </tr>
   </tbody>
   </table>

   > ![输入图片说明](https://images.gitee.com/uploads/images/2021/0719/172222_3c2963f4_923381.gif "icon-note.gif") **说明：**  
   > 不要遗漏命令结尾的“.“。

4. 准备数据。

   执行位于/infer/data/models下的coco_trainval_anns.py脚本，导出准备用于推理的数据。

      ```shell
      # 导出txt推理数据
      python coco_trainval_anns.py --data_url=./datasets/ --train_url=./infer/data/models/ --val_url=./infer/data/images/
      #data_url参数为数据集datasets存储路径，train_url参数为存储txt路径,val_url为推理数据集存放的路径
      ```

   AIR模型可通过“模型训练”后转换生成。

   将生成的推理数据拷贝到 infer/data/models、infer/mxbase 和 infer/sdk 目录下。

5. 启动容器。

   进入“infer“目录，执行以下命令，启动容器。  

   ```shell
   bash docker_start_infer.sh docker_image:tag model_dir
   ```

   > ![输入图片说明](https://images.gitee.com/uploads/images/2021/0926/181445_0077d606_8725359.gif) **说明：**
   > MindX SDK开发套件（mxManufacture）已安装在基础镜像中，安装路径：“/usr/local/sdk_home“。

   **表 2** 参数说明

   | 参数           | 说明                                  |
   | -------------- | ------------------------------------- |
   | *docker_image* | 推理镜像名称及镜像tag，根据实际写入。 |
   | tag | 镜像tag，请根据实际配置，如：21.0.2。 |
   | data_path      | 代码路径。                            |

   启动容器时会将推理芯片和数据路径挂载到容器中。可根据需要通过修改**docker_start_infer.sh**的device来指定挂载的推理芯片。

## 模型转换

   1. 准备模型文件。

* 将ModelArts训练之后导出的 **.air 模型文件放入 infer/data/models 目录下

   2. 模型转换。

* 执行 infer/convert/air2om.sh， 转换命令如下 。

      ```SHELL
      cd ./infer/convert
      #bash air2om.sh air_path(转换脚本AIR文件路径) om_path(生成的OM文件名，转换脚本会在此基础上添加.om后缀)
      bash air2om.sh ../data/models/yolov4.air ../data/models/yolov4
      ```

      执行完成后会在 infer/data/model 目录下生成 **.om 模型文件，注意此处 om 文件名需与 pipeline 中的保持一致。

## MxBase推理

   1. 配置环境变量

      ```SHELL
      export ASCEND_HOME=/usr/local/Ascend
      export ASCEND_VERSION=ascend-toolkit/latest
      export ARCH_PATTERN=.
      export LD_LIBRARY_PATH=${MX_SDK_HOME}/lib/modelpostprocessors:$LD_LIBRARY_PATH
      ```

   2. （可选）修改配置文件
      可根据实际情况修改，配置文件位于“mxbase/src/main.cpp”中，可修改参数如下:

      ```SHELL
      initParam.deviceId = 0;
      initParam.labelPath = "../data/models/coco2017.names";#实际使用的标签名表
      initParam.checkTensor = true;
      initParam.modelPath = "../data/models/yolov4.om";#实际的推理模型文件
      initParam.classNum = 80;#实际数据集类别数
      initParam.biasesNum = 18;
      initParam.biases = "12,16,19,36,40,28,36,75,76,55,72,146,142,110,192,243,459,401";
      initParam.objectnessThresh = "0.001";
      initParam.iouThresh = "0.6";#nms用到的IOU阈值，可调整
      initParam.scoreThresh = "0.001";
      initParam.yoloType = 3;
      initParam.modelType = 0;
      initParam.inputType = 0;
      initParam.anchorDim = 3;
      ```

      根据实际情况修改"mxbase/src/Yolov4Detection.cpp"中的图片缩放尺寸：

      ```SHELL
      APP_ERROR Yolov4TinyDetectionOpencv::Resize(cv::Mat &srcImageMat, cv::Mat &dstImageMat) {
         static constexpr uint32_t resizeHeight = 608; #模型输入高度
         static constexpr uint32_t resizeWidth = 608; #模型输入宽度
         cv::resize(srcImageMat, dstImageMat, cv::Size(resizeWidth, resizeHeight));
         return APP_ERR_OK;
      }
      ```

   3. 编译工程。

      ```
      cd ./infer/mxbase
      bash build.sh
      ```

   4. 运行推理服务。

      a. 确保验证集图片的权限为640

      ```shell
      #可以通过以下命令确保验证集图片的权限为640
      chmod 640 ../data/images/. -R #此处为验证集图片地址
      ```

      b. 确保result文件夹为空，或者不存在

      ```shell
      #可以通过以下命令确保结果文件夹为空，或者不存在
      rm -rf ./result/result.txt #删除结果文件
      rm -rf ./result #删除结果文件夹
      rm -rf ./result.json #删除结果转换文件
      ```

      c. 执行推理脚本，确保记录推理图片路径的文件在/infer/mxbase文件夹下，命令如下 。

      ```shell
      #./build/Yolov4_mindspore image_path_txt(记录推理图片路径的txt文件。如：trainval.txt)
      ./build/Yolov4_mindspore ./trainval.txt
      ```

      推理结果保存在“./result/result.txt”。

   5. 观察结果。

      拷贝infer/data/models/object_task_metric.py 和coco2017的验证集标签instances_val2017.json文件到“mxbase”目录下.  
      根据实际情况修改object_task_metric.py代码

      ```shell
      if __name__ == "__main__":
        ban_path = './trainval.txt' # 修改为实际的推理数据集路径的文件
        input_file = './result/result.txt'
        if not os.path.exists(ban_path):
            print('The infer text file does not exist.')
        if not os.path.exists(input_file):
            print('The result text file does not exist.')

        image_id_list = get_image_id(ban_path)
        result_dict = get_dict_from_file(input_file, image_id_list)
        json_file_name = './result.json'
        with open(json_file_name, 'w') as f:
            json.dump(result_dict, f)

        # set iouType to 'segm', 'bbox' or 'keypoints'
        ann_type = ('segm', 'bbox', 'keypoints')
        # specify type here
        ann_type = ann_type[1]
        coco_gt_file = './instances_val2017.json' # 修改为真实标签文件
      ```

   6. 查看精度  

      执行以下命令计算精度。

      ```shell
      bash infermxbase.sh
      ```

      推理结果以json格式保存，路径为“./result.json”。  
      精度信息示例如下所示：

      ```shell
       Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.455
       Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.646
       Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.495
       Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.278
       Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.481
       Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.565
       Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.358
       Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.575
       Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.605
       Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.424
       Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.632
       Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.710
      ```

## MindX SDK推理

   1. 编译后处理代码

      基于MindX SDK推理的后处理代码编译时直接编译 “mxbase/src/PostProcess”。

      ```shell
      cd infer/sdk/mxpi
      bash build.sh
      ```

   2. 修改配置文件。

      a.根据实际情况修改config中的pipeline文件。

      ```shell
      {
       "im_yolov4": {
        "stream_config": {
            "deviceId": "0"
        },
        "appsrc0": {
            "props": {
                "blocksize": "409600"
            },
            "factory": "appsrc",
            "next": "mxpi_imagedecoder0"
        },
        "mxpi_imagedecoder0": {
            "props": {
                "handleMethod": "opencv"
            },
            "factory": "mxpi_imagedecoder",
            "next": "mxpi_imageresize0"
        },
        "mxpi_imageresize0": {
            "props": {
                "parentName": "mxpi_imagedecoder0",
                "handleMethod": "opencv",
                "resizeHeight": "608",#模型输入高度
                "resizeWidth": "608",#模型输入宽度
                "resizeType": "Resizer_Stretch"
            },
            "factory": "mxpi_imageresize",
            "next": "mxpi_tensorinfer0"
        },
        "mxpi_tensorinfer0": {
            "props": {
                "dataSource": "mxpi_imageresize0",
                "modelPath": "../data/models/yolov4.om",#推理模型路径
                "waitingTime": "3000",
                "outputDeviceId": "-1"
            },
            "factory": "mxpi_tensorinfer",
            "next": "mxpi_objectpostprocessor0"
        },
        "mxpi_objectpostprocessor0": {
            "props": {
                "dataSource": "mxpi_tensorinfer0",
                "postProcessConfigPath": "../data/models/yolov4_coco2017_acc_test.cfg",#推理后处理相关参数配置文件路径
                "labelPath": "../data/models/coco2017.names",#推理数据集类别标签文件，需自行添加到对应目录
                "postProcessLibPath": "./mxpi/build/libyolov4_mindspore_post.so"#编译后处理so文件
            },
            "factory": "mxpi_objectpostprocessor",
            "next": "mxpi_dataserialize0"
        },
        "mxpi_dataserialize0": {
            "props": {
                "outputDataKeys": "mxpi_objectpostprocessor0"
            },
            "factory": "mxpi_dataserialize",
            "next": "appsink0"
        },
        "appsink0": {
            "props": {
                "blocksize": "4096000"
            },
            "factory": "appsink"
        }
        }
      }
      ```

      b. 可根据实际情况修改后处理配置文件  

      其配置文件yolov4_coco2017_acc_test.cfg在“../data/models/”目录下.

      ```shell
      # hyper-parameter
      CLASS_NUM=80 #推理数据集类别数
      BIASES_NUM=18
      BIASES=12,16,19,36,40,28,36,75,76,55,72,146,142,110,192,243,459,401
      SCORE_THRESH=0.001
      OBJECTNESS_THRESH=0.001
      IOU_THRESH=0.6 #nms用到的IOU阈值，可调整
      YOLO_TYPE=3
      ANCHOR_DIM=3
      MODEL_TYPE=0
      RESIZE_FLAG=0
      ```

   3. 运行推理服务。  

      a. 确保trainval.txt文件在sdk目录下。

      b. 修改main.py中记录推理图片路径的文件路径。  

      ```shell
      infer_file = './trainval.txt' #根据实际情况进行修改
      ```

      c. 确保验证集图片和编译后生成的后处理文件/sdk/mxpi/build/libyolov4_mindspore_post.so的权限为640

      ```shell
      #可以通过以下命令确保验证集图片和后处理文件的权限为640
      chmod 640 ../data/images/. -R #此处为验证集图片地址
      chmod 640 ./mxpi/build/libyolov4_mindspore_post.so #此处为后处理文件路径
      ```

      d. 确保result文件夹为空，或者不存在

      ```shell
      #可以通过以下命令确保结果文件夹为空，或者不存在
      rm -rf ./result/result.txt #删除结果文件
      rm -rf ./result #删除结果文件夹
      rm -rf ./result.json #删除结果转换文件
      ```

      e. 执行推理

      ```shell
      cd infer/sdk
      bash run.sh
      ```

   4. 观察结果。

      拷贝infer/data/models/object_task_metric.py和coco2017的验证集标签instances_val2017.json文件到“sdk”目录下。  
      根据实际情况修改object_task_metric.py代码。  

      ```shell
      ...
      if __name__ == "__main__":
        ban_path = './trainval.txt' # 修改为实际的推理数据集路径的文件
        input_file = './result/result.txt'
        if not os.path.exists(ban_path):
            print('The infer text file does not exist.')
        if not os.path.exists(input_file):
            print('The result text file does not exist.')

        image_id_list = get_image_id(ban_path)
        result_dict = get_dict_from_file(input_file, image_id_list)
        json_file_name = './result.json'
        with open(json_file_name, 'w') as f:
            json.dump(result_dict, f)

        # set iouType to 'segm', 'bbox' or 'keypoints'
        ann_type = ('segm', 'bbox', 'keypoints')
        # specify type here
        ann_type = ann_type[1]
        coco_gt_file = './instances_val2017.json' # 修改为真实标签文件
      ...
      ```

   5. 查看精度

      执行以下命令计算精度。

      ```shell
      bash infersdk.sh
      ```

      推理结果以json格式保存，路径为“./result.json”。  
      精度信息示例如下所示：

      ```shell
       Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.455
       Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.646
       Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.495
       Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.278
       Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.481
       Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.565
       Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.358
       Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.575
       Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.605
       Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.424
       Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.632
       Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.710
      ```

   6. 性能测试。

         开启性能统计开关，在sdk.conf配置文件中，设置 enable_ps=true，开启性能统计开关。

         调整性能统计时间间隔，设置ps_interval_time=2，每隔2秒，进行一次性能统计。

         进入infer/sdk目录，执行推理命令脚本，启动SDK推理服务 。

   7. 查看性能结果。  

         在日志目录"~/MX_SDK_HOME/logs"查看性能统计结果。

         ```shell
         performance-statistics.log.e2e.xx×
         performance-statistics.log.plugin.xx×
         performance-statistics.log.tpr.xxx
         ```

         其中e2e日志统计端到端时间，plugin日志统计单插件时间。

# 在ModelArts上应用

## 创建OBS桶

1. 创建桶。

* 登录[OBS管理控制台](https://storage.huaweicloud.com/obs)，创建OBS桶，具体请参见[“创建桶”](https://support.huaweicloud.com/usermanual-obs/obs_03_0306.html)章节。
* ”区域“选择”华北-北京四“
* ”存储类别“选取”标准存储“
* ”桶ACL“选取”私有“
* 关闭”多AZ“
* 输入全局唯一桶名称, 例如 “S3"
* 点击”确定“

2. 创建文件夹存放数据。

   在创建的桶中创建以下文件夹：

* code：存放训练脚本
* datasets: 存放数据集
* preckpt：存放预训练模型
* output: 存放训练生成ckpt模型
* logs：存放训练日志目录

3. 上传代码

* 进入 yolov4 代码文件根目录
 * 将 yolov4 目录下的文件全部上传至 obs://S3/yolov4 文件夹下

## 创建算法

1. 使用华为云帐号登录[ModelArts管理控制台](https://console.huaweicloud.com/modelarts)，在左侧导航栏中选择“算法管理”。
2. 在“我的算法管理”界面，单击左上角“创建”，进入“创建算法”页面。
3. 在“创建算法”页面，填写相关参数，然后单击“提交”。
4. 设置算法基本信息如下。

```text
   # ==================================创建算法==========================================
   # (1) 上传你的代码和数据集到 S3 桶上
   # (2) 创建方式: 自定义脚本
         AI引擎：Ascend-Powered-Engine mindspore_1.3.0-cann_5.0.2-py_3.7-euler_2.8.3-aarch64
         代码目录： /S3/yolov4/
         启动文件： /S3/yolov4/modelarts.py
   # (3) 超参：
         名称               类型            必需
         data_url         String          是
         train_url        String          是
         checkpoint_url   String          是
   # (4) 自定义超参：支持
   # (5) 输入数据配置:  "映射名称 = '数据来源2'", "代码路径参数 = 'data_url'","映射名称 = '数据来源3'", "代码路径参数 = 'checkpoint_url'"
   # (6) 输出数据配置:  "映射名称 = '输出数据1'", "代码路径参数 = 'train_url'"
   # (7) 添加训练约束： 否
```

## 创建训练作业

1. 登录ModelArts。

2. 创建训练作业。

    训练作业参数配置说明如下。

   ```text
   # ==================================创建训练作业=======================================
   # (1) 算法： 在我的算法中选择前面创建的算法
   # (2) 训练输入： '/S3/yolov4/datasets/'
   # 在OBS桶/S3/gat/目录下新建output文件夹
   # (3) 训练输出： '/S3/yolov4/output/'
   # (4) 超参：
            "data_dir = 'obs://S3/yolov4/datasets/'"
            "train_dir='obs://S3/yolov4/output/'"
            "checkpoint_url='obs://S3/yolov4/preckpt/'"
   # (5) 设置作业日志路径
            "log='obs://S3/yolov4/log/'"
   ```

3. 单击“提交”，完成训练作业的创建。

   训练作业一般需要运行一段时间，根据您选择的数据量和资源不同，训练时间将耗时几分钟左右。训练结果模型将保存在 obs://S3/gat/results/model/ 文件夹下。

## 查看训练任务日志

1. 训练完成后进入logs文件夹，点击对应当次训练作业的日志文件即可。

2. logs文件夹内生成日志文件，您可在  /logs 文件夹下的日志文件中找到如下结果：

      ```text
      2022-03-29 13:36:59,826:INFO:epoch[0], iter[117199], loss:495.129946, per step time: 45.80 ms, fps: 21.83, lr:0.011993246152997017
      ...
      2022-03-29 13:53:04,842:INFO:Calculating mAP...
      2022-03-29 14:24:23,597:INFO:result file path: /home/ma-user/modelarts/outputs/train_url_0/2022-03-29_time_11_31_12/predict_2022_03_29_14_22_47.json
     ...
     Accumulating evaluation results...
     DONE (t=14.87s).
     2022-03-29 14:27:32,440:INFO:epoch: 1, mAP:
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.000
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.000
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.000
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.000
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.000
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.000
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.000
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.001
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.001
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.002
      ```
