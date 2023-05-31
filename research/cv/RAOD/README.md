# Contents

- [Contents](#contents)
- [Model Description](#model-description)
    - [\[CVPR 2023\] Toward RAW Object Detection: A New Benchmark and A New Model](#cvpr-2023-toward-raw-object-detection-a-new-benchmark-and-a-new-model)
    - [Abstract](#abstract)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
- [Performance](#performance)
    - [Platform(#platform)](#platformplatform)
    - [Results](#results)
- [Contributors](#contributors)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)
- [ModeZoo Homepage](#modezoo-homepage)

# [Model Description](#model-description)

## [CVPR 2023] Toward RAW Object Detection: A New Benchmark and A New Model

Ruikang Xu*, Chang Chen*, Jingyang Peng*, Cheng Li, Yibin Huang, Fenglong Song, Youliang Yan, and [Zhiwei Xiong#](http://staff.ustc.edu.cn/~zwxiong)

(*Equal contribution, #Corresponding author)

This is the official MindSpore evaluation code of RAOD.

## Abstract

In many computer vision applications (e.g., robotics and
autonomous driving), high dynamic range (HDR) data is
necessary for object detection algorithms to handle a variety of lighting conditions, such as strong glare. In this paper, we aim to achieve object detection on RAW sensor data, which naturally saves the HDR information from image sensors without extra equipment costs. We build a novel RAW sensor dataset, named ROD, for Deep Neural Networks (DNNs)-based object detection algorithms to be applied to HDR data. The ROD dataset contains a large amount of annotated instances of day and night driving scenes in 24-bit dynamic range. Based on the dataset, we first investigate the impact of dynamic range for DNNs-based detectors and demonstrate the importance of dynamic range adjustment for detection on RAW sensor data. Then, we propose a simple and effective adjustment method for object detection on HDR RAW sensor data, which is image adaptive and jointly optimized with the downstream detector in an end-to-end scheme. Extensive experiments demonstrate that the performance of detection on RAW sensor data is significantly superior to standard dynamic range (SDR) data in different situations. Moreover, we analyze the influence of texture information and pixel distribution of input data on the performance of the DNNs-based detector.

# [Dataset](#dataset)

Please download the ROD dataset from [this link](https://openi.pcl.ac.cn/innovation_contest/innov202305091731448/datasets?lang=en-US). Please arrange the dataset as follows:

```text
datasets/
        /file_list.txt
        /raws/
             /file1.raw
             /file2.raw
             /...
        /anno/
             /file1.json
             /file2.json
             /...
```

[Here](https://openi.pcl.ac.cn/innovation_contest/innov202305091731448) provides sample codes for how to load our dataset.

# [Environment Requirements](#environment-requirements)

- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

- Dependencies
    - [See requirements](./requirements.txt).

# [Quick Start](#quick-start)

[Here](./scripts/eval_template.sh) provides a template of starting an evaluation. [Here](./scripts/eval_example.sh) is an example.

# [Script Description](#script-description)

## [Script and Sample Code](#script-and-sample-code)

```text
└──ROD
    ├── dataset
        ├── __init__.py
        ├── data_utils.py                   # modules used for data pre-processing
        └── dataset.py                      # definition of dataloader
    ├── model
        ├── __init__.py
        ├── adaptive_module.py              # our proposed global / local adjustment modules
        ├── boxes.py                        # yolox dependencies
        ├── config.py                       # yolox dependencies
        ├── darknet.py                      # yolox dependencies
        ├── network_blocks.py               # yolox dependencies
        ├── util.py                         # yolox dependencies
        ├── yolo_fpn.py                     # yolox dependencies
        ├── yolo_pafpn.py                   # yolox dependencies
        ├── yolox_darknet53.yaml            # yolox dependencies
        └── yolox.py                        # yolox dependencies
    ├── scripts
        ├── eval_template.sh                # template of starting evaluation
        └── eval_example.sh                 # example of starting evaluation
    ├── main.py                             # main interface
    ├── README.md
    └── requirements.txt                    # project dependencies
```

# [Performance](#performance)

## Platform(#platform)

| Parameters          | GPU                         |
| ------------------- | --------------------------- |
| Resource            | CUDA 11.0                   |
| Uploaded Date       | 04/20/2023                  |
| MindSpore Version   | 1.9.0                       |

## Results

| scenes | AP   | AR   | AP50 | AP75 |
| ------ | ---- | ---- | ---- | ---- |
| day    | 58.7 | 63.9 | 85.3 | 61.3 |
| night  | 54.2 | 61.7 | 83.0 | 58.2 |

# [Citation](#citation)

If you find our work helpful, please cite the following paper.

```text
@InProceedings{CVPR2023_ROD,
    author    = {Xu, Ruikang and Chen, Chang and Peng, Jingyang and Li, Cheng and Huang, Yibin and Song, Fenglong and Yan, Youliang and Xiong, Zhiwei},
    title     = {Toward RAW Object Detection: A New Benchmark and A New Model},
    booktitle = {CVPR},
    year      = {2023},
}
```

# [Acknowledgement](#acknowledgement)

Our code is modified based on [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX).

# [ModeZoo Homepage](#modelzoo)

Please check the official [homepage](https://gitee.com/mindspore/models).
