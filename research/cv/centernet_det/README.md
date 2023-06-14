# Contents

- [CenterNet Description](#centernet-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Distributed Training](#distributed-training)
    - [Testing Process](#testing-process)
        - [Testing and Evaluation](#testing-and-evaluation)
    - [Ascend Inference Process](#ascend-inference-process)
        - [Convert](#convert)
        - [Infer on Ascend310](#infer-on-Ascend310)
        - [Result](#result)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance On Ascend 910](#training-performance-on-ascend-910)
        - [Inference Performance On Ascend 910](#inference-performance-on-ascend-910)
        - [Inference Performance On Ascend 310](#inference-performance-on-ascend-310)
- [ModelZoo Homepage](#modelzoo-homepage)

# [CenterNet Description](#contents)

CenterNet is a novel practical anchor-free method for object detection, 3D detection, and pose estimation, which detect identifies objects as axis-aligned boxes in an image. The detector uses keypoint estimation to find center points and regresses to all other object properties, such as size, 3D location, orientation, and even pose. In nature, it's a one-stage method to simultaneously predict center location and bboxes with real-time speed and higher accuracy than corresponding bounding box based detectors.
We support training and evaluation on Ascend910.

[Paper](https://arxiv.org/pdf/1904.07850.pdf): Objects as Points. 2019.
Xingyi Zhou(UT Austin) and Dequan Wang(UC Berkeley) and Philipp Krahenbuhl(UT Austin)

# [Model Architecture](#contents)

The stacked Hourglass Network downsamples the input by 4×,followed by two sequential hourglass modules.Each hourglass module is a symmetric 5-layer down-and up-convolutional network with skip connections .This network is quite large ,but generally yields the best keypoint estimation performance.

# [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

Dataset used: [COCO2017](https://cocodataset.org/)

- Dataset size：26G
    - Train：19G，118000 images
    - Val：0.8G，5000 images
    - Test: 6.3G, 40000 images
    - Annotations：808M，instances，captions etc
- Data format：image and json files

- Note：Data will be processed in dataset.py

- The directory structure is as follows, name of directory and file is user defined:

    ```text
    .
    ├── dataset
        ├── centernet
            ├── annotations
            │   ├─ train.json
            │   └─ val.json
            └─ images
                ├─ train
                │    └─images
                │       ├─class1_image_folder
                │       ├─ ...
                │       └─classn_image_folder
                └─ val
                │    └─images
                │       ├─class1_image_folder
                │       ├─ ...
                │       └─classn_image_folder
                └─ test
                      └─images
                        ├─class1_image_folder
                        ├─ ...
                        └─classn_image_folder
    ```

# [Environment Requirements](#contents)

- Hardware（Ascend）

    - Prepare hardware environment with Ascend processor.
- Framework

    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore tutorials](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)
- Download the dataset COCO2017.
- We use COCO2017 as training dataset in this example by default, and you can also use your own datasets.

    1. If coco dataset is used. **Select dataset to coco when run script.**
        Install Cython and pycocotool, and you can also install mmcv to process data.

        ```bash
        pip install Cython

        pip install pycocotools

        pip install mmcv==0.2.14
        ```

        And change the COCO_ROOT and other settings you need in `config.py`. The directory structure is as follows:

        ```text
        .
        └─cocodataset
          ├─annotations
            ├─instance_train2017.json
            └─instance_val2017.json
          ├─val2017
          └─train2017

        ```

    2. If your own dataset is used. **Select dataset to other when run script.**
        Organize the dataset information the same format as COCO.

# [Quick Start](#contents)

## Running on local (Ascend or GPU)

After installing MindSpore via the official website, you can start training and evaluation as follows:

**Notes:**

1. the first run of training will generate the mindrecord file, which will take a long time.
2. MINDRECORD_DATASET_PATH is the mindrecord dataset directory.
3. For `train.py`, LOAD_CHECKPOINT_PATH is the optional pretrained checkpoint file, if no just set "".
4. For `eval.py`, LOAD_CHECKPOINT_PATH is the checkpoint to be evaluated.
5. RUN_MODE argument support validation and testing, set to be "val"/"test"

### Ascend

The training configuration is in `default_config.yaml`.

```bash
# create dataset in mindrecord format
bash scripts/convert_dataset_to_mindrecord.sh [COCO_DATASET_DIR] [MINDRECORD_DATASET_DIR]

# standalone training on Ascend
bash scripts/run_standalone_train_ascend.sh [DEVICE_ID] [MINDRECORD_DATASET_PATH] [LOAD_CHECKPOINT_PATH](optional)

# distributed training on Ascend
bash scripts/run_distributed_train_ascend.sh [MINDRECORD_DATASET_PATH] [RANK_TABLE_FILE] [LOAD_CHECKPOINT_PATH](optional)

# eval on Ascend
bash scripts/run_standalone_eval_ascend.sh [DEVICE_ID] [RUN_MODE] [DATA_DIR] [LOAD_CHECKPOINT_PATH]
```

### GPU

**Note:** the training configuration is in `centernetdet_gpu_config.yaml`.

```bash
# create dataset in mindrecord format
bash scripts/convert_dataset_to_mindrecord.sh [COCO_DATASET_DIR] [MINDRECORD_DATASET_DIR]

# standalone training on GPU
bash scripts/run_standalone_train_gpu.sh [DEVICE_ID] [MINDRECORD_DIR] [LOAD_CHECKPOINT_PATH](optional)

# distributed training on GPU
bash scripts/run_distributed_train_gpu.sh [MINDRECORD_DIR] [NUM_DEVICES] [LOAD_CHECKPOINT_PATH](optional)

# eval on GPU
bash scripts/run_standalone_eval_gpu.sh [DEVICE_ID] [RUN_MODE] [DATA_DIR] [LOAD_CHECKPOINT_PATH]
```

## Running on ModelArts

If you want to run in modelarts, please check the official documentation of modelarts, and you can start training as follows

- Creating mindrecord dataset with single cards on ModelArts

    ```text
    # (1) Upload the code folder to S3 bucket.
    # (2) Upload the COCO2017 dataset to S3 bucket.
    # (2) Click to "create task" on the website UI interface.
    # (3) Set the code directory to "/{path}/centernet_det" on the website UI interface.
    # (4) Set the startup file to /{path}/centernet_det/dataset.py" on the website UI interface.
    # (5) Perform a or b.
    #     a. setting parameters in /{path}/centernet_det/default_config.yaml.
    #         1. Set ”enable_modelarts: True“
    #     b. adding on the website UI interface.
    #         1. Add ”enable_modelarts=True“
    # (7) Check the "data storage location" on the website UI interface and set the "Dataset path" path.
    # (8) Set the "Output file path" and "Job log path" to your path on the website UI interface.
    # (9) Under the item "resource pool selection", select the specification of single cards.
    # (10) Create your job.
    ```

- Training with single cards on ModelArts

   ```text
    # (1) Upload the code folder to S3 bucket.
    # (2) Click to "create task" on the website UI interface.
    # (3) Set the code directory to "/{path}/centernet_det" on the website UI interface.
    # (4) Set the startup file to /{path}/centernet_det/train.py" on the website UI interface.
    # (5) Perform a or b.
    #     a. setting parameters in /{path}/centernet_det/default_config.yaml.
    #         1. Set ”enable_modelarts: True“
    #         2. Set “epoch_size: 130”
    #         3. Set “distribute: 'true'”
    #         4. Set “save_checkpoint_path: ./checkpoints”
    #     b. adding on the website UI interface.
    #         1. Add ”enable_modelarts=True“
    #         2. Add “epoch_size=130”
    #         3. Add “distribute=true”
    #         4. Add “save_checkpoint_path=./checkpoints”
    # (6) Upload the mindrecord dataset to S3 bucket.
    # (7) Check the "data storage location" on the website UI interface and set the "Dataset path" path.
    # (8) Set the "Output file path" and "Job log path" to your path on the website UI interface.
    # (9) Under the item "resource pool selection", select the specification of single cards.
    # (10) Create your job.
   ```

- evaluating with single card on ModelArts

   ```text
    # (1) Upload the code folder to S3 bucket.
    # (2) Git clone https://github.com/xingyizhou/CenterNet.git on local, and put the folder 'CenterNet' under the folder 'centernet' on s3 bucket.
    # (3) Click to "create task" on the website UI interface.
    # (4) Set the code directory to "/{path}/centernet_det" on the website UI interface.
    # (5) Set the startup file to /{path}/centernet_det/eval.py" on the website UI interface.
    # (6) Perform a or b.
    #     a. setting parameters in /{path}/centernet_det/default_config.yaml.
    #         1. Set ”enable_modelarts: True“
    #         2. Set “run_mode: 'val'”
    #         3. Set "load_checkpoint_path='/cache/checkpoint_path/model.ckpt'" on yaml file.
    #         4. Set "checkpoint_url=/The path of checkpoint in S3/" on yaml file.
    #     b. adding on the website UI interface.
    #         1. Add ”enable_modelarts=True“
    #         2. Add “run_mode=val”
    #         3. Add "load_checkpoint_path='/cache/checkpoint_path/model.ckpt'" on the website UI interface.
    #         4. Add "checkpoint_url=/The path of checkpoint in S3/" on the website UI interface.
    # (7) Upload the dataset(not mindrecord format) to S3 bucket.
    # (8) Check the "data storage location" on the website UI interface and set the "Dataset path" path.
    # (9) Set the "Output file path" and "Job log path" to your path on the website UI interface.
    # (10) Under the item "resource pool selection", select the specification of a single card.
    # (11) Create your job.
    ```

After installing MindSpore via the official website, you can start training and evaluation as follows:

Note: 1.the first run of training will generate the mindrecord file, which will take a long time.
      2.MINDRECORD_DATASET_PATH is the mindrecord dataset directory.
      3.LOAD_CHECKPOINT_PATH is the pretrained checkpoint file directory, if no just set ""
      4.RUN_MODE support validation and testing, set to be "val"/"test"

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```text
.
├── cv
    ├── centernet_det
        ├── train.py                     // training scripts
        ├── eval.py                      // testing and evaluation outputs
        ├── export.py                    // convert mindspore model to mindir model
        ├── README.md                    // descriptions about centernet_det
        ├── default_config.yaml          // Ascend parameter configuration
        ├── centernetdet_gpu_config.yaml // GPU parameter configuration
        ├── ascend310_infer              // application for 310 inference
        ├── preprocess.py                // preprocess scripts
        ├── postprocess.py               // postprocess scripts
        ├── scripts
        │   ├── ascend_distributed_launcher
        │   │    ├── __init__.py
        │   │    ├── hyper_parameter_config.ini         // hyper parameter for distributed training
        │   │    ├── get_distribute_train_cmd.py        // script for distributed training
        │   │    └── README.md
        │   ├── convert_dataset_to_mindrecord.sh        // shell script for converting coco type dataset to mindrecord
        │   ├── run_standalone_train_ascend.sh          // shell script for standalone training on ascend
        │   ├── run_standalone_train_gpu.sh             // shell script for standalone training on GPU
        │   ├── run_infer_310.sh                        // shell script for 310 inference on ascend
        │   ├── run_distributed_train_ascend.sh         // shell script for distributed training on ascend
        │   ├── run_distributed_train_gpu.sh            // shell script for distributed training on GPU
        │   ├── run_standalone_eval_ascend.sh           // shell script for standalone evaluation on ascend
        │   └── run_standalone_eval_gpu.sh              // shell script for standalone evaluation on GPU
        └── src
            ├── model_utils
            │   ├── config.py            // parsing parameter configuration file of "*.yaml"
            │   ├── device_adapter.py    // local or ModelArts training
            │   ├── local_adapter.py     // get related environment variables on local
            │   └── moxing_adapter.py    // get related environment variables abd transfer data on ModelArts
            ├── __init__.py
            ├── centernet_det.py          // centernet networks, training entry
            ├── dataset.py                // generate dataloader and data processing entry
            ├── decode.py                 // decode the head features
            ├── hourglass.py              // hourglass backbone
            ├── image.py                  // image preprocess functions
            ├── post_process.py           // post-process functions after decode in inference
            ├── utils.py                  // auxiliary functions for train, to log and preload
            └── visual.py                 // visualization image, bbox, score and keypoints
```

## [Script Parameters](#contents)

### Create MindRecord type dataset

```text
usage: dataset.py  [--coco_data_dir COCO_DATA_DIR]
                   [--mindrecord_dir MINDRECORD_DIR]
                   [--mindrecord_prefix MINDRECORD_PREFIX]

options:
    --coco_data_dir            path to coco dataset directory: PATH, default is ""
    --mindrecord_dir           path to mindrecord dataset directory: PATH, default is ""
    --mindrecord_prefix        prefix of MindRecord dataset filename: STR, default is "coco_det.train.mind"
```

### Training

```text
usage: train.py  [--device_target DEVICE_TARGET] [--distribute DISTRIBUTE]
                 [--need_profiler NEED_PROFILER] [--profiler_path PROFILER_PATH]
                 [--epoch_size EPOCH_SIZE] [--train_steps TRAIN_STEPS]  [device_id DEVICE_ID]
                 [--device_num DEVICE_NUM] [--do_shuffle DO_SHUFFLE]
                 [--enable_data_sink ENABLE_DATA_SINK] [--data_sink_steps N]
                 [--enable_save_ckpt ENABLE_SAVE_CKPT]
                 [--save_checkpoint_path SAVE_CHECKPOINT_PATH]
                 [--load_checkpoint_path LOAD_CHECKPOINT_PATH]
                 [--save_checkpoint_steps N] [--save_checkpoint_num N]
                 [--mindrecord_dir MINDRECORD_DIR]
                 [--mindrecord_prefix MINDRECORD_PREFIX]
                 [--save_result_dir SAVE_RESULT_DIR]

options:
    --device_target            device where the code will be implemented: "Ascend"
    --distribute               training by several devices: "true"(training by more than 1 device) | "false", default is "true"
    --need profiler            whether to use the profiling tools: "true" | "false", default is "false"
    --profiler_path            path to save the profiling results: PATH, default is ""
    --epoch_size               epoch size: N, default is 1
    --train_steps              training Steps: N, default is -1
    --device_id                device id: N, default is 0
    --device_num               number of used devices: N, default is 1
    --do_shuffle               enable shuffle: "true" | "false", default is "true"
    --enable_lossscale         enable lossscale: "true" | "false", default is "true"
    --enable_data_sink         enable data sink: "true" | "false", default is "true"
    --data_sink_steps          set data sink steps: N, default is 1
    --enable_save_ckpt         enable save checkpoint: "true" | "false", default is "true"
    --save_checkpoint_path     path to save checkpoint files: PATH, default is ""
    --load_checkpoint_path     path to load checkpoint files: PATH, default is ""
    --save_checkpoint_steps    steps for saving checkpoint files: N, default is 1000
    --save_checkpoint_num      number for saving checkpoint files: N, default is 1
    --mindrecord_dir           path to mindrecord dataset directory: PATH, default is ""
    --mindrecord_prefix        prefix of MindRecord dataset filename: STR, default is "coco_det.train.mind"
    --save_result_dir          path to save the visualization results: PATH, default is ""
```

### Evaluation

```text
usage: eval.py  [--device_target DEVICE_TARGET] [--device_id N]
                [--load_checkpoint_path LOAD_CHECKPOINT_PATH]
                [--data_dir DATA_DIR] [--run_mode RUN_MODE]
                [--visual_image VISUAL_IMAGE]
                [--enable_eval ENABLE_EVAL] [--save_result_dir SAVE_RESULT_DIR]
options:
    --device_target              device where the code will be implemented: "Ascend"
    --device_id                  device id to run task, default is 0
    --load_checkpoint_path       initial checkpoint (usually from a pre-trained CenterNet model): PATH, default is ""
    --data_dir                   validation or test dataset dir: PATH, default is ""
    --run_mode                   inference mode: "val" | "test", default is "val"
    --visual_image               whether visualize the image and annotation info: "true" | "false", default is "false"
    --save_result_dir            path to save the visualization and inference results: PATH, default is ""
```

### Options and Parameters

Parameters for training and evaluation can be set in file `config.py`.

#### Options

```text
train_config.
    batch_size: 12                  // batch size of input dataset: N, default is 12
    loss_scale_value: 1024          // initial value of loss scale: N, default is 1024
    optimizer: 'Adam'               // optimizer used in the network: Adam, default is Adam
    lr_schedule: 'MultiDecay'       // schedules to get the learning rate
```

```text
config for evaluation.
    SOFT_NMS: True                  // nms after decode: True | False, default is True
    keep_res: True                  // keep original or fix resolution: True | False, default is True
    multi_scales: [1.0]             // use multi-scales of image: List, default is [1.0]
    K: 100                          // number of bboxes to be computed by TopK, default is 100
    score_thresh: 0.3               // threshold of score when visualize image and annotation info,default is 0.3
```

#### Parameters

```text
Parameters for dataset (Training/Evaluation):
    num_classes                     number of categories: N, default is 80
    max_objs                        maximum numbers of objects labeled in each image,default is 128
    input_res                       input resolution, default is [512, 512]
    output_res                      output resolution, default is [128, 128]
    rand_crop                       whether crop image in random during data augmenation: True | False, default is True
    shift                           maximum value of image shift during data augmenation: N, default is 0.1
    scale                           maximum value of image scale times during data augmenation: N, default is 0.4
    aug_rot                         properbility of image rotation during data augmenation: N, default is 0.0
    rotate                          maximum value of rotation angle during data augmentation: N, default is 0.0
    flip_prop                       properbility of image flip during data augmenation: N, default is 0.5
    color_aug                       color augmentation of RGB image, default is True
    coco_classes                    name of categories in COCO2017
    mean                            mean value of RGB image
    std                             variance of RGB image
    eig_vec                         eigenvectors of RGB image
    eig_val                         eigenvalues of RGB image

Parameters for network (Training/Evaluation):
    down_ratio                      the ratio of input and output resolution during training,default is 4
    num_stacks　　　　　　　　　　　　 the number of stacked hourglass network, default is 2
    n                               the number of stacked hourglass modules, default is 5
    heads                           the number of heatmap,width and height,offset, default is {'hm': 80, 'wh': 2, 'reg': 2}
    cnv_dim                         the convolution of dimension, default is 256
    modules                         the number of stacked residual networks, default is [2, 2, 2, 2, 2, 4]
    dims                            residual network input and output dimensions, default is [256, 256, 384, 384, 384, 512]
    dense_hp                        whether apply weighted pose regression near center point: True | False, default is True
    dense_wh                        apply weighted regression near center or just apply regression on center point
    cat_spec_wh                     category specific bounding box size
    reg_offset                      regress local offset or not: True | False, default is True
    hm_weight                       loss weight for keypoint heatmaps: N, default is 1.0
    off_weight                      loss weight for keypoint local offsets: N, default is 1
    wh_weight                       loss weight for bounding box size: N, default is 0.1
    mse_loss                        use mse loss or focal loss to train keypoint heatmaps: True | False, default is False
    reg_loss                        l1 or smooth l1 for regression loss: 'l1' | 'sl1', default is 'l1'

Parameters for optimizer and learning rate:
    Adam:
    weight_decay                    weight decay: Q
    decay_filer                     lamda expression to specify which param will be decayed

    PolyDecay:
    learning_rate                   initial value of learning rate: Q,default is 2.4e-4
    end_learning_rate               final value of learning rate: Q,default is 2.4e-7
    power                           learning rate decay factor,default is 5.0
    eps                             normalization parameter,default is 1e-7
    warmup_steps                    number of warmup_steps,default is 2000

    MultiDecay:
    learning_rate                   initial value of learning rate: Q,default is 2.4e-4
    eps                             normalization parameter,default is 1e-7
    warmup_steps                    number of warmup_steps,default is 2000
    multi_epochs                    list of epoch numbers after which the lr will be decayed,default is [105, 125]
    factor                          learning rate decay factor,default is 10
```

## [Training Process](#contents)

Before your first training, convert coco type dataset to mindrecord files is needed to improve performance on host.

```bash
bash scripts/convert_dataset_to_mindrecord.sh /path/coco_dataset_dir /path/mindrecord_dataset_dir
```

The command above will run in the background, after converting mindrecord files will be located in path specified by yourself.

### Distributed Training

#### Running on Ascend

```bash
bash scripts/run_distributed_train_ascend.sh /path/mindrecord_dataset /path/hccl.json /path/load_ckpt(optional)
```

The command above will run in the background, you can view training logs in LOG*/training_log.txt and LOG*/ms_log/. After training finished, you will get some checkpoint files under the LOG*/ckpt_0 folder by default. The loss value will be displayed as follows:

```text
# grep "epoch" training_log.txt
epoch: 128, current epoch percent: 1.000, step: 157509, outputs are (Tensor(shape=[], dtype=Float32, value= 1.54529), Tensor(shape=[], dtype=Bool, value= False), Tensor(shape=[], dtype=Float32, value= 1024))
epoch time: 1211875.286 ms, per step time: 992.527 ms
epoch: 129, current epoch percent: 1.000, step: 158730, outputs are (Tensor(shape=[], dtype=Float32, value= 1.57337), Tensor(shape=[], dtype=Bool, value= False), Tensor(shape=[], dtype=Float32, value= 1024))
epoch time: 1214703.313 ms, per step time: 994.843 ms
...
```

#### Running on GPU

```bash
bash scripts/run_distributed_train_gpu.sh /path/mindrecord_dataset 8
```

The command above will run in the background, you can view training logs in Train_parallel/training_log.txt. After training finished, you will get some checkpoint files under the Train_parallel/checkpointsckpt_0 folder by default. The loss value will be displayed as follows:

```text
epoch: 120, current epoch percent: 0.287, step: 195827, outputs are 1.628212
epoch: 120, current epoch percent: 0.287, step: 195827, outputs are 2.0098093
epoch: 120, current epoch percent: 0.287, step: 195827, outputs are 2.2687669
epoch: 120, current epoch percent: 0.287, step: 195827, outputs are 2.0722423
epoch: 120, current epoch percent: 0.287, step: 195827, outputs are 1.3634725
epoch: 120, current epoch percent: 0.287, step: 195827, outputs are 1.933584
epoch: 120, current epoch percent: 0.287, step: 195827, outputs are 1.5815283
epoch: 120, current epoch percent: 0.287, step: 195827, outputs are 1.4993639
epoch: 120, current epoch percent: 0.287, step: 195828, outputs are 2.063395
epoch: 120, current epoch percent: 0.287, step: 195828, outputs are 1.83587
epoch: 120, current epoch percent: 0.287, step: 195828, outputs are 1.8395581
epoch: 120, current epoch percent: 0.287, step: 195828, outputs are 1.5058619
epoch: 120, current epoch percent: 0.287, step: 195828, outputs are 2.1509295
epoch: 120, current epoch percent: 0.287, step: 195828, outputs are 2.082848
epoch: 120, current epoch percent: 0.287, step: 195828, outputs are 1.7891021
epoch: 120, current epoch percent: 0.287, step: 195828, outputs are 2.2131023
```

## [Testing Process](#contents)

### Testing and Evaluation

#### Ascend results

```bash
# Evaluation base on validation dataset will be done automatically, while for test or test-dev dataset, the accuracy should be upload to the CodaLab official website(https://competitions.codalab.org).
# On Ascend
bash scripts/run_standalone_eval_ascend.sh device_id val(or test) /path/coco_dataset /path/load_ckpt
```

you can see the MAP result below as below:

```text
overall performance on coco2017 validation dataset
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.415
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.604
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.447
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.248
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.457
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.536
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.338
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.566
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.599
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.394
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.656
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.764
```

#### GPU results

```bash
# example
bash scripts/run_standalone_eval_gpu.sh 0 val /path/coco_dataset /path/load_ckpt
```

you can see the MAP result below as below:

```text
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.409
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.600
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.441
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.235
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.450
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.536
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.338
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.556
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.589
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.387
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.644
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.755
```

## [Ascend Inference Process](#contents)

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

### Convert

If you want to infer the network on Ascend 310, you should convert the model to MINDIR:

- Export on local

  ```text
  python export.py --device_id [DEVICE_ID] --export_format MINDIR --export_load_ckpt [CKPT_FILE__PATH] --export_name [EXPORT_FILE_NAME]
  ```

- Export on ModelArts (If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start as follows)

  ```text
  # (1) Upload the code folder to S3 bucket.
  # (2) Click to "create training task" on the website UI interface.
  # (3) Set the code directory to "/{path}/centernet_det" on the website UI interface.
  # (4) Set the startup file to /{path}/centernet_det/export.py" on the website UI interface.
  # (5) Perform a or b.
  #     a. setting parameters in /{path}/centernet_det/default_config.yaml.
  #         1. Set ”enable_modelarts: True“
  #         2. Set “export_load_ckpt: ./{path}/*.ckpt”('export_load_ckpt' indicates the path of the weight file to be exported relative to the file `export.py`, and the weight file must be included in the code directory.)
  #         3. Set ”export_name: centernet_det“
  #         4. Set ”export_format：MINDIR“
  #     b. adding on the website UI interface.
  #         1. Add ”enable_modelarts=True“
  #         2. Add “export_load_ckpt=./{path}/*.ckpt”('export_load_ckpt' indicates the path of the weight file to be exported relative to the file `export.py`, and the weight file must be included in the code directory.)
  #         3. Add ”export_name=centernet_det“
  #         4. Add ”export_format=MINDIR“
  # (7) Check the "data storage location" on the website UI interface and set the "Dataset path" path (This step is useless, but necessary.).
  # (8) Set the "Output file path" and "Job log path" to your path on the website UI interface.
  # (9) Under the item "resource pool selection", select the specification of a single card.
  # (10) Create your job.
  # You will see centernet.mindir under {Output file path}.
  ```

### Infer on Ascend310

Before performing inference, the mindir file must be exported by export.py script. We only provide an example of inference using MINDIR model. Current batch_size can only be set to 1.

  ```shell
  #Ascend310 inference
  bash run_infer_310.sh [MINDIR_PATH] [DATASET_PATH] [PREPROCESS_IMAGES] [DEVICE_ID]
  ```

- `PREPROCESS_IMAGES` Weather need preprocess or not, it's value must be in [y, n]

### Result

Inference result is saved in current path, you can find result like this in acc.log file.Since the input images are fixed shape on Ascend 310, all accuracy will be lower than that on Ascend 910.

```log
 #acc.log
 =============coco2017 310 infer reulst=========
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.410
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.600
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.440
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.213
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.437
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.567
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.339
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.543
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.572
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.342
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.620
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.764
```

# [Model Description](#contents)

## [Performance](#contents)

### Training Performance

CenterNet on 11.8K images(The annotation and data format must be the same as coco)

| Parameters                 | CenterNet_Hourglass (Ascend)                                 | CenterNet_Hourglass (GPU)                                    |
| -------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory, 755G              | GPU RTX 3090 24GB                                            |
| uploaded Date              | 3/27/2021 (month/day/year)                                   | 09/22/2022 (month/day/year)                                  |
| MindSpore Version          | 1.1.0                                                        | 1.8.1                                                        |
| Dataset                    | COCO2017                                                     | COCO2017                                                     |
| Training Parameters        | 8p, epoch=130, steps=158730, batch_size = 12, lr=2.4e-4      | 8p, epoch=130, batch_size = 9, lr=2.4e-4                     |
| Optimizer                  | Adam                                                         | Adam                                                         |
| Loss Function              | Focal Loss, L1 Loss, RegLoss                                 | Focal Loss, L1 Loss, RegLoss                                 |
| outputs                    | detections                                                   | detections                                                   |
| Loss                       | 1.5-2.5                                                      | 1.5-2.5                                                      |
| Speed                      | 8p 20 img/s                                                  | 8p 20 img/s                                                  |
| Total time: training       | 8p: 44 h                                                     | 8p: 95 h                                                     |
| Total time: evaluation     | keep res: test 1h, val 0.25h; fix res: test 40 min, val 8 min| keep res: test 1h, val 0.25h; fix res: test 40 min, val 8 min|
| Checkpoint                 | 2.3G (.ckpt file)                                            | 2.3G (.ckpt file)                                            |
| Scripts                    | [centernet_det script](https://gitee.com/mindspore/models/tree/r2.0/research/cv/centernet_det) |

### Inference Performance

CenterNet on validation(5K images)

| Parameters                 | CenterNet_Hourglass (Ascend)                                     | CenterNet_Hourglass (GPU)                                        |
| -------------------------- | ---------------------------------------------------------------- | ---------------------------------------------------------------- |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory, 755G                  | GPU RTX 3090 24GB                                                |
| uploaded Date              | 3/27/2021 (month/day/year)                                       | 09/22/2022 (month/day/year)                                      |
| MindSpore Version          | 1.1.0                                                            | 1.8.1                                                            |
| Dataset                    | COCO2017                                                         | COCO2017                                                         |
| batch_size                 | 1                                                                | 1                                                                |
| outputs                    | mAP                                                              | mAP                                                              |
| Accuracy(validation)       | MAP: 41.5%, AP50: 60.4%, AP75: 44.7%, Medium: 45.7%, Large: 53.6%| MAP: 40.9%, AP50: 60.0%, AP75: 44.1%, Medium: 45.0%, Large: 53.6%|

### Inference Performance On Ascend 310

CenterNet on validation(5K images)

| Parameters                 | CenterNet_Hourglass                                                       |
| -------------------------- | ----------------------------------------------------------------|
| Resource                   | Ascend 310; CentOS 3.10                |
| uploaded Date              | 8/31/2021 (month/day/year)                                     |
| MindSpore Version          | 1.4.0                                                           |
| Dataset                    | COCO2017                            |
| batch_size                 | 1                                                               |
| outputs                    | mAP                         |
| Accuracy(validation)       | MAP: 41.0%, AP50: 60.0%, AP75: 44.0%, Medium: 43.7%, Large: 56.7%|

# [Description of Random Situation](#contents)

In run_distributed_train_ascend.sh, we set do_shuffle to True to shuffle the dataset by default.
In train.py, we set a random seed to make sure that each node has the same initial weight in distribute training.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).

# FAQ

First refer to [ModelZoo FAQ](https://gitee.com/mindspore/models#FAQ) to find some common public questions.
