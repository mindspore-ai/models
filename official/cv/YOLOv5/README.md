# Contents

- [Contents](#contents)
- [YOLOv5 Description](#yolov5-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Training](#training)
        - [Distributed Training](#distributed-training)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
    - [Inference Process](#inference-process)
        - [Export MindIR](#export-mindir)
        - [Infer](#infer)
        - [result](#result)
        - [Export ONNX](#export-onnx)
        - [Run ONNX evaluation](#run-onnx-evaluation)
        - [result](#result-1)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Evaluation Performance](#evaluation-performance)
        - [Inference Performance](#inference-performance)
        - [Transfer Learning](#transfer-learning)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [YOLOv5 Description](#contents)

Published in April 2020, YOLOv5 achieved state of the art performance on the COCO dataset for object detection. It is an important improvement of YoloV3, the implementation of a new architecture in the **Backbone** and the modifications in the **Neck** have improved the **mAP**(mean Average Precision) by **10%** and the number of **FPS**(Frame per Second) by **12%**.

[code](https://github.com/ultralytics/yolov5)

# [Model Architecture](#contents)

The YOLOv5 network is mainly composed of CSP and Focus as a backbone, spatial pyramid pooling(SPP) additional module, PANet path-aggregation neck and YOLOv3 head. [CSP](https://arxiv.org/abs/1911.11929) is a novel backbone that can enhance the learning capability of CNN. The [spatial pyramid pooling](https://arxiv.org/abs/1406.4729) block is added over CSP to increase the receptive field and separate out the most significant context features. Instead of Feature pyramid networks (FPN) for object detection used in YOLOv3, the PANet is used as the method for parameter aggregation for different detector levels. To be more specifical, CSPDarknet53 contains 5 CSP modules which use the convolution **C** with kernel size k=3x3, stride s = 2x2; Within the PANet and SPP, **1x1, 5x5, 9x9, 13x13 max poolings are applied.

# [Dataset](#contents)

Dataset used: [COCO2017](<https://cocodataset.org/#download>)

Note that you can run the scripts with **COCO2017 **or any other datasets with the same format as MS COCO Annotation. But we do suggest user to use MS COCO dataset to experience our model.

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

```bash
#run training example(1p) on Ascend/GPU by python command
python train.py \
    --device_target="Ascend" \ # Ascend or GPU
    --data_dir=xxx/dataset \
    --is_distributed=0 \
    --yolov5_version='yolov5s' \
    --lr=0.01 \
    --max_epoch=320 \
    --warmup_epochs=4 > log.txt 2>&1 &
```

```bash
# run 1p by shell script, please change `device_target` in config file to run on Ascend/GPU, and change `T_max`, `max_epoch`, `warmup_epochs` refer to contents of notes
bash run_standalone_train.sh [DATASET_PATH] [DEVICE_ID]

# For Ascend device, distributed training example(8p) by shell script
bash run_distribute_train.sh [DATASET_PATH] [RANK_TABLE_FILE]

# For GPU device, distributed training example(8p) by shell script
bash run_distribute_train_gpu.sh [DATASET_PATH] [RANK_SIZE]
```

```bash
# run evaluation on Ascend/GPU by python command
python eval.py \
    --device_target="Ascend" \ # Ascend or GPU
    --data_dir=xxx/dataset \
    --yolov5_version='yolov5s' \
    --pretrained="***/*.ckpt" \
    --eval_shape=640 > log.txt 2>&1 &
```

```bash
# run evaluation by shell script, please change `device_target` in config file to run on Ascend/GPU
bash run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH] [DEVICE_ID]
```

Note the default_config.yaml is the default parameters for yolov5s on 8p. The `batchsize` and `lr` are different on Ascend and GPU, see the settings in `scripts/run_distribute_train.sh` or `scripts/run_distribute_train_gpu.sh`.

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```text
├── model_zoo
    ├── README.md                              // descriptions about all the models
    ├── yolov5
        ├── README.md                          // descriptions about yolov5
        ├── scripts
        │   ├──docker_start.sh                 // shell script for docker start
        |   ├──run_distribute_eval.sh          // launch distributed evaluation in ascend
        │   ├──run_distribute_train.sh         // launch distributed training(8p) in ascend
        │   ├──run_distribute_train_gpu.sh     // launch distributed training(8p) in GPU
        │   ├──run_standalone_train.sh         // launch 1p training
        │   ├──run_infer_310.sh                // shell script for evaluation on 310
        │   ├──run_eval.sh                     // shell script for evaluation
        │   ├──run_eval_onnx.sh                // shell script for onnx evaluation
        ├──model_utils
        │   ├──config.py                       // getting config parameters
        │   ├──device_adapter.py               // getting device info
        │   ├──local_adapter.py                // getting device info
        │   ├──moxing_adapter.py               // Decorator
        ├── src
        │   ├──backbone.py                     // backbone of network
        │   ├──distributed_sampler.py          // iterator of dataset
        │   ├──initializer.py                  // initializer of parameters
        │   ├──logger.py                       // log function
        │   ├──loss.py                         // loss function
        │   ├──lr_scheduler.py                 // generate learning rate
        │   ├──transforms.py                   // Preprocess data
        │   ├──util.py                         // util function
        │   ├──yolo.py                         // yolov5 network
        │   ├──yolo_dataset.py                 // create dataset for YOLOV5
        ├── default_config.yaml                // parameter configuration(yolov5s 8p)
        ├── train.py                           // training script
        ├── eval.py                            // evaluation script
        ├── eval_onnx.py                       // ONNX evaluation script
        ├── export.py                          // export script
```

## [Script Parameters](#contents)

```text
Major parameters in train.py are:

optional arguments:

  --device_target       device where the code will be implemented: "Ascend", default is "Ascend"
  --data_dir            Train dataset directory.
  --per_batch_size      Batch size for Training. Default: 32(1p) 16(Ascend 8p) 32(GPU 8p).
  --resume_yolov5       The ckpt file of YOLOv5, which used to fine tune.Default: ""
  --lr_scheduler        Learning rate scheduler, options: exponential,cosine_annealing.
                        Default: cosine_annealing
  --lr                  Learning rate. Default: 0.01(1p) 0.02(Ascend 8p) 0.025(GPU 8p)
  --lr_epochs           Epoch of changing of lr changing, split with ",". Default: '220,250'
  --lr_gamma            Decrease lr by a factor of exponential lr_scheduler. Default: 0.1
  --eta_min             Eta_min in cosine_annealing scheduler. Default: 0.
  --t_max               T-max in cosine_annealing scheduler. Default: 300(8p)
  --max_epoch           Max epoch num to train the model. Default: 300(8p)
  --warmup_epochs       Warmup epochs. Default: 20(8p)
  --weight_decay        Weight decay factor. Default: 0.0005
  --momentum            Momentum. Default: 0.9
  --loss_scale          Static loss scale. Default: 64
  --label_smooth        Whether to use label smooth in CE. Default:0
  --label_smooth_factor Smooth strength of original one-hot. Default: 0.1
  --log_interval        Logging interval steps. Default: 100
  --ckpt_path           Checkpoint save location. Default: outputs/
  --is_distributed      Distribute train or not, 1 for yes, 0 for no. Default: 0
  --rank                Local rank of distributed. Default: 0
  --group_size          World size of device. Default: 1
  --need_profiler       Whether use profiler. 0 for no, 1 for yes. Default: 0
  --training_shape      Fix training shape. Default: ""
  --resize_rate         Resize rate for multi-scale training. Default: 10
  --bind_cpu            Whether bind cpu when distributed training. Default: True
  --device_num          Device numbers per server. Default: 8
  --save_ckpt_interval  Epoch interval to save checkpoint. Default: 1
  --save_ckpt_max_num   Max number of checkpoints to save. If the number of saved checkpoints is larger than this value,
                        the oldest chekpoint would be removed. Default: 10
```

## [Training Process](#contents)

### Training

For Ascend device, standalone training can be started like this:

```shell
#run training example(1p) by python command
python train.py \
    --data_dir=xxx/dataset \
    --yolov5_version='yolov5s' \
    --is_distributed=0 \
    --lr=0.01 \
    --T_max=320
    --max_epoch=320 \
    --warmup_epochs=4 \
    --per_batch_size=32 \
    --lr_scheduler=cosine_annealing > log.txt 2>&1 &
```

You should fine tune the params when run training 1p on GPU

The python command above will run in the background, you can view the results through the file `log.txt`.

After training, you'll get some checkpoint files under the **outputs** folder by default. The loss value will be achieved as follows:

```text
# grep "loss:" log.txt
2021-08-06 15:30:15,798:INFO:epoch[0], iter[600], loss:296.308071, fps:44.44 imgs/sec, lr:0.00010661844862625003
2021-08-06 15:31:21,119:INFO:epoch[0], iter[700], loss:276.071959, fps:48.99 imgs/sec, lr:0.00012435863027349114
2021-08-06 15:32:26,185:INFO:epoch[0], iter[800], loss:266.955208, fps:49.18 imgs/sec, lr:0.00014209879736881703
2021-08-06 15:33:30,507:INFO:epoch[0], iter[900], loss:252.610914, fps:49.75 imgs/sec, lr:0.00015983897901605815
2021-08-06 15:34:42,176:INFO:epoch[0], iter[1000], loss:243.106683, fps:44.65 imgs/sec, lr:0.00017757914611138403
2021-08-06 15:35:47,429:INFO:epoch[0], iter[1100], loss:240.498834, fps:49.04 imgs/sec, lr:0.00019531932775862515
2021-08-06 15:36:48,945:INFO:epoch[0], iter[1200], loss:245.711473, fps:52.02 imgs/sec, lr:0.00021305949485395104
2021-08-06 15:37:51,293:INFO:epoch[0], iter[1300], loss:231.388255, fps:51.33 imgs/sec, lr:0.00023079967650119215
2021-08-06 15:38:55,680:INFO:epoch[0], iter[1400], loss:238.904242, fps:49.70 imgs/sec, lr:0.00024853984359651804
2021-08-06 15:39:57,419:INFO:epoch[0], iter[1500], loss:232.161600, fps:51.83 imgs/sec, lr:0.00026628002524375916
2021-08-06 15:41:03,808:INFO:epoch[0], iter[1600], loss:227.844698, fps:48.20 imgs/sec, lr:0.00028402020689100027
2021-08-06 15:42:06,155:INFO:epoch[0], iter[1700], loss:226.668858, fps:51.33 imgs/sec, lr:0.00030176035943441093
...
```

### Distributed Training

Distributed training example(8p) by shell script:

```bash
# For Ascend device, distributed training example(8p) by shell script
bash run_distribute_train.sh [DATASET_PATH] [RANK_TABLE_FILE]

# For GPU device, distributed training example(8p) by shell script
bash run_distribute_train_gpu.sh [DATASET_PATH] [RANK_SIZE]
```

The above shell script will run distribute training in the background. You can view the results through the file train_parallel[X]/log.txt(Ascend) or distribute_train/nohup.out(GPU). The loss value will be achieved as follows:

```text
# distribute training result(8p, dynamic shape)
...
2021-08-05 16:01:34,116:INFO:epoch[0], iter[200], loss:415.453676, fps:580.07 imgs/sec, lr:0.0002742903889156878
2021-08-05 16:01:57,588:INFO:epoch[0], iter[300], loss:273.358383, fps:545.96 imgs/sec, lr:0.00041075327317230403
2021-08-05 16:02:26,247:INFO:epoch[0], iter[400], loss:244.621502, fps:446.64 imgs/sec, lr:0.0005472161574289203
2021-08-05 16:02:55,532:INFO:epoch[0], iter[500], loss:234.524876, fps:437.10 imgs/sec, lr:0.000683679012581706
2021-08-05 16:03:25,046:INFO:epoch[0], iter[600], loss:235.185213, fps:434.08 imgs/sec, lr:0.0008201419259421527
2021-08-05 16:03:54,585:INFO:epoch[0], iter[700], loss:228.878598, fps:433.48 imgs/sec, lr:0.0009566047810949385
2021-08-05 16:04:23,932:INFO:epoch[0], iter[800], loss:219.259134, fps:436.29 imgs/sec, lr:0.0010930676944553852
2021-08-05 16:04:52,707:INFO:epoch[0], iter[900], loss:225.741833, fps:444.84 imgs/sec, lr:0.001229530549608171
2021-08-05 16:05:21,872:INFO:epoch[1], iter[1000], loss:218.811336, fps:438.91 imgs/sec, lr:0.0013659934047609568
2021-08-05 16:05:51,216:INFO:epoch[1], iter[1100], loss:219.491889, fps:436.50 imgs/sec, lr:0.0015024563763290644
2021-08-05 16:06:20,546:INFO:epoch[1], iter[1200], loss:219.895906, fps:436.57 imgs/sec, lr:0.0016389192314818501
2021-08-05 16:06:49,521:INFO:epoch[1], iter[1300], loss:218.516680, fps:441.79 imgs/sec, lr:0.001775382086634636
2021-08-05 16:07:18,303:INFO:epoch[1], iter[1400], loss:209.922935, fps:444.79 imgs/sec, lr:0.0019118449417874217
2021-08-05 16:07:47,702:INFO:epoch[1], iter[1500], loss:210.997816, fps:435.60 imgs/sec, lr:0.0020483077969402075
2021-08-05 16:08:16,482:INFO:epoch[1], iter[1600], loss:210.678421, fps:444.88 imgs/sec, lr:0.002184770768508315
2021-08-05 16:08:45,568:INFO:epoch[1], iter[1700], loss:203.285874, fps:440.07 imgs/sec, lr:0.0023212337400764227
2021-08-05 16:09:13,947:INFO:epoch[1], iter[1800], loss:203.014775, fps:451.11 imgs/sec, lr:0.0024576964788138866
2021-08-05 16:09:42,954:INFO:epoch[2], iter[1900], loss:194.683969, fps:441.28 imgs/sec, lr:0.0025941594503819942
...
```

## [Evaluation Process](#contents)

### Evaluation

Before running the command below, please check the checkpoint path used for evaluation. The file **yolov5.ckpt** used in the  follow script is the last saved checkpoint file, but we renamed it to "yolov5.ckpt".

```shell
# run evaluation by python command
python eval.py \
    --data_dir=xxx/dataset \
    --pretrained=xxx/yolov5.ckpt \
    --eval_shape=640 > log.txt 2>&1 &
OR
# run evaluation by shell script
bash run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH] [DEVICE_ID]
```

The above python command will run in the background. You can view the results through the file "log.txt". The mAP of the test dataset will be as follows:

```text
# log.txt
=============coco eval reulst=========
Average Precision (AP) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.369
Average Precision (AP) @[ IoU=0.50 | area= all | maxDets=100 ] = 0.573
Average Precision (AP) @[ IoU=0.75 | area= all | maxDets=100 ] = 0.395
Average Precision (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.218
Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.418
Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.482
Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets= 1 ] = 0.298
Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets= 10 ] = 0.501
Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.557
Average Recall (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.395
Average Recall (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.619
Average Recall (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.677
2020-12-21 17:16:40,322:INFO:testing cost time 0.35h
```

### Distributed Evaluation

The script `run_distribute_eval.sh` in `scripts` folder can be used to do evaluation in parallel.

The following is an usage example:

```bash
bash run_distribute_eval.sh /path/to/dataset /path/to/hccl_config.json
```

## Inference Process

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

### [Export MindIR](#contents)

```shell
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

The ckpt_file parameter is required,
`file_format` should be in ["AIR", "MINDIR"]

### Infer

Before performing inference, the mindir file must be exported by `export.py` script. We only provide an example of inference using MINDIR model.
Current batch_Size can only be set to 1.

```shell
bash run_infer_cpp.sh [MINDIR_PATH] [DATA_PATH] [ANN_FILE] [DVPP] [DEVICE_TYPE] [DEVICE_ID]
```

- `DVPP` is mandatory, and must choose from ["DVPP", "CPU"], it's case-insensitive. The DVPP hardware restricts width 16-alignment and height even-alignment. Therefore, the network needs to use the CPU operator to process images.
- `DATA_PATH` is mandatory, path of the dataset containing images.
- `ANN_FILE` is mandatory, path to annotation file.
- `DEVICE_ID` is optional, default value is 0.

### result

Inference result is saved in current path, you can find result like this in acc.log file.

```text
Average Precision (AP) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.369
Average Precision (AP) @[ IoU=0.50 | area= all | maxDets=100 ] = 0.573
Average Precision (AP) @[ IoU=0.75 | area= all | maxDets=100 ] = 0.395
Average Precision (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.218
Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.418
Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.482
Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets= 1 ] = 0.298
Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets= 10 ] = 0.501
Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.557
Average Recall (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.395
Average Recall (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.619
Average Recall (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.677
```

### [Export ONNX](#contents)

- Export your model to ONNX:  

  ```shell
  python export.py --ckpt_file /path/to/yolov5.ckpt --file_name /path/to/yolov5.onnx --file_format ONNX
  ```

### Run ONNX evaluation

- Run ONNX evaluation from yolov5 directory:

  ```shell
  bash scripts/run_eval_onnx.sh <DATA_DIR> <ONNX_MODEL_PATH> [<DEVICE_TARGET>]
  ```

### result

- You can view the results through the file eval.log. The mAP of the validation dataset will be as follows:

  ```text
  =============coco eval result=========
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.366
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.569
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.397
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.213
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.415
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.474
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.299
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.501
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.557
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.399
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.611
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.677
  ```

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

YOLOv5 on 118K images(The annotation and data format must be the same as coco2017)

| Parameters                 | YOLOv5s                                                      | YOLOv5s                                                      |
| -------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Resource                   | Ascend 910 ；CPU 2.60GHz，192cores; Memory, 755G               | GPU NV SMX2 V100-32G                                         |
| uploaded Date              | 7/12/2021 (month/day/year)                                   | 9/15/2021 (month/day/year)                                   |
| MindSpore Version          | 1.2.0                                                        | 1.3.0                                                        |
| Dataset                    | 118K images                                                  | 118K images                                                  |
| Training Parameters        | epoch=300, batch_size=8, lr=0.02,momentum=0.9,warmup_epoch=20| epoch=300, batch_size=32, lr=0.025, warmup_epoch=20, 8p      |
| Optimizer                  | Momentum                                                     | Momentum                                                     |
| Loss Function              | Sigmoid Cross Entropy with logits, Giou Loss                 | Sigmoid Cross Entropy with logits, Giou Loss                 |
| outputs                    | boxes and label                                              | boxes and label                                              |
| Loss                       | 111.970097                                                   | 85                                                           |
| Speed                      | 8p about 450 FPS                                             | 8p about 290 FPS                                             |
| Total time                 | 8p 21h28min                                                  | 8p 35h                                                       |
| Checkpoint for Fine tuning | 53.62M (.ckpt file)                                          | 58.87M (.ckpt file)                                          |
| Scripts                    | https://gitee.com/mindspore/models/tree/r2.0/official/cv/YOLOv5 | https://gitee.com/mindspore/models/tree/r2.0/official/cv/YOLOv5 |

### Inference Performance

| Parameters          | YOLOv5s                                        | YOLOv5s                                      |
| ------------------- | -----------------------------------------------| ---------------------------------------------|
| Resource            | Ascend 910 ；CPU 2.60GHz，192cores; Memory, 755G | GPU NV SMX2 V100-32G                         |
| Uploaded Date       | 7/12/2021 (month/day/year)                     | 9/15/2021 (month/day/year)                   |
| MindSpore Version   | 1.2.0                                          | 1.3.0                                        |
| Dataset             | 20K images                                     | 20K images                                   |
| batch_size          | 1                                              | 1                                            |
| outputs             | box position and sorces, and probability       | box position and sorces, and probability     |
| Accuracy            | mAP >= 36.7%(shape=640)                        | mAP >= 36.7%(shape=640)                      |
| Model for inference | 56.67M (.ckpt file)                            | 58.87M (.ckpt file)                          |

### Transfer Learning

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py.

# [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/models).  
