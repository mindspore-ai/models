
# Contents

- [Contents](#contents)
- [HMR Description](#hmr-description)
- [Dataset](#dataset)
    - [dataset processing](#dataset-processing)
    - [dataset sample code](#dataset-sample-code)
- [Script description](#script-description)
    - [Scripts and sample code](#scripts-and-sample-code)
- [Quick start](#quick-start)
    - [Training process](#training-process)
    - [Eval process](#eval-process)
    - [Inference](#inference)
        - [Inference](#inference)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
            - [HMR train on Dataset](#hmr-train-on-dataset)
        - [Inference Performance](#inference-performance)
            - [HMR infer on Human3.6m](#hmr-infer-on-human36m)
- [ModelZoo Homepage](#modelzoo-homepage)

# [HMR Description](#contents)  

Mainly for three-dimensional reconstruction of the human body. In this paper, the encoder module is used for feature extraction, the regressor is used for multiple iterative feature output, the SMPL module is used for reprojection algorithm to obtain its joint point coordinates, and finally it is judged to remove its ambiguous parameters.
Available open source code:  
<https://github.com/MandyMo/pytorch_HMR>  
Paper download address:  
<http://arxiv.org/abs/1712.06584>  
# [Dataset](#contents)  

The dataset used in this model mainly consists of two parts, you can download [All Datasets](https://pan.baidu.com/s/1aLfKrNNkHdoUXSa6hKu-Jg?pwd=fzcn) from here.

1. Contains four two-dimensional datasets of [Coco2017](https://cocodataset.org/#home), [Lsp](http://sam.johnson.io/research/lsp_dataset.zip), [Lsp-extend](http://sam.johnson.io/research/lspet_dataset.zip), and [Mpii](http://human-pose.mpi-inf.mpg.de/#download). We filter images that are too small Or have less than 6 visible keypoints and get training set sizes of 1k, 10k, 20k and 80k images.

2. Contains two 3D datasets, [MPI-INF-3D](http://gvv.mpi-inf.mpg.de/3dhp-dataset/) and [Human3.6m](http://vision.imar.ro/human3.6m/description.php) We exclude the sequence from training MPI-INF-3DHP topic 8 as a validation set hyperparameter to be adjusted, and use the complete training set for Final experiment. Both datasets are captured in a controlled environment and provide 150,000 3D joint training image annotations.  

## dataset processing  

```bash  

cat x*>HMR_DATA.tar.gz

tar xzvf HMR_DATA.tar.gz

# Running dataprocess  
python  dataprocess.py --data_path
# python  dataprocess.py --data_path  /mass_store/dataset/HMR_DATA

```  

## dataset sample code  

```bash  
├── HMR_DATA  
    ├── neutral_smpl_mean_params.h5  
    ├── neutral_smpl_with_cocoplus_reg.pkl  
    ├── lsp  
    │   ├──images
    │   ├──joints.mat  
    │   ├──jReadME.txt  
    │   ├──jvisualized  
    ├── coco2017  
    │   ├──annotations
    │   ├──test2017  
    │   ├──train2017  
    │   ├──val2017  
    ├── lsp_exd  
    │   ├──images  
    │   ├──joints.mat  
    │   ├──README.txt  
    ├── mpii  
    │   ├──images  
    │   ├──annotations  
    │   ├──eft_annots.npz  
    │   ├──mpii_human_pose_v1_u12_2
    │   ├──annot  
    ├── human3.6m  
    │   ├──images
    │   ├──annot.h5  
    │   ├──annots.npz  
    ├── mpii_3d  
    │   ├──images
    │   ├──annot.h5  
    │   ├──annots.npz  

```  

# [Script description](#contents)  

## Scripts and sample code  

```bash  
├── model_zoo  
    ├── README.md  
    ├── HMR  
        ├──config.yaml
        ├──eval.py  
        ├──export.py  
        ├──postprocess.py  
        ├──preprocess.py  
        ├──run_infer_310.sh  
        ├──trainer_hmr.py  
        ├──util.py  
        ├── README.md  
        ├── ascend310_infer  
        ├── scripts  
        │   ├──train_distribute.sh  
        │   ├──train_singe_npu.sh  
        │   ├──run_infer_310.sh  
        ├── model_utils  
        │   ├──config.py
        ├── src  
        │   ├──util.py  
        │   ├──model.py  
        │   ├──dataset.py  
        │   ├──cal_loss.py  
```  

# [Quick start](#contents)  

After installing MindSpore through the official website, you can follow the steps below for training and evaluation:  

## Training process  

- Ascend processor environment running, add checkpoint path parameter before training  

```bash

# Running the standalone training example
bash ./scripts/run_standalone_train.sh [DATASET_PATH]
# bash ./scripts/run_standalone_train.sh  /mass_store/dataset/HMR_DATA

# Running the distributed training example
bash ./scripts/run_distributed_train.sh [RANK_TABLE_FILE] [DATASET_PATH]
# bash ./scripts/run_distributed_train.sh ./hccl_2p_01_127.0.0.1.json  /mass_store/dataset/HMR_DATA

#续训过程：
# Running the standalone training example
bash ./scripts/run_standalone_train.sh [DATASET_PATH] [PRETRAINED_CKPT_PATH]
# bash ./scripts/run_standalone_train.sh  /mass_store/dataset/HMR_DATA  /mass_store/zjc/models/research/cv/HMR/generator-20000.ckpt

# Running the distributed training example
bash ./scripts/run_distributed_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [PRETRAINED_CKPT_PATH]
# bash ./scripts/run_distribute_train.sh  ./hccl_tools/hccl_8p_01234567_127.0.0.1.json  /mass_store/dataset/HMR_DATA   /mass_store/zjc/models/research/cv/HMR/generator-20000.ckpt

```  

## Eval process  

```bash
# Run the evaluation example  
bash ./scripts/run_eval.sh [DATASET_PATH] [PRETRAINED_CKPT_PATH]  
# bash ./scripts/run_eval.sh  /mass_store/dataset/HMR_DATA  /mass_store/zjc/models/research/cv/HMR/generator-20000.ckpt  
```

## Inference  

### Inference  

Before we can run inference we need to export the model first. The Air model can only be exported in the Shengteng 910 environment, mindir can be exported in any environment, and the batch_size only supports 32.  

Inference using Human3.6m dataset on Ascend 310  

Before executing the following command, we need to modify the configuration file of Human3.6m. Modified items include batch_size and val_data_path. The LABEL_FILE parameter is only useful for the ImageNet dataset and can be passed any value.  

```bash
#在导出之前需要修改数据集对应的配置文件，需要修改的配置项为 batch_size 和  batch_size_3d , 以及ckpt_file.
python export.py --batch_size=1 --batch_3d_size=0 --data_path  --checkpoint_file_path
#python export.py  --data_path=/mass_store/dataset/HMR_DATA  --batch_size=1 --batch_3d_size=0 --checkpoint_file_path=/mass_store/zjc/models/research/cv/HMR/generator-20000.ckpt

# Run the inference example
bash ./scripts/run_infer_310.sh [MINDIR_PATH] [DATASET] [DATA_PATH] [DEVICE_ID]
# bash ./scripts/run_infer_310.sh ./hmr.mindir Human3.6m /home/HMR_DATA  0
# MINDIR_PATH   mindir模型路径
# DATASET       数据集名称,默认为  Human3.6m
# DATA_PATH     数据集路径
# LABEL_FILE    310推理得到的结果保存路径
# DEVICE_ID     310设备ID

```

The results of inference are saved in the current directory, and results similar to the following can be found in the acc.log log file.

# [Model Description](#contents)

## [Performance](#contents)

### Training Performance

#### HMR train on Dataset

| Parameters                 | Ascend                                                      |  
| -------------------------- | ----------------------------------------------------------- |  
| Model Version              | HMR                                                |  
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8             |  
| uploaded Date              | 08/04/2022 (month/day/year)                                 |  
| MindSpore Version          | 1.5.0                                                       |  
| Dataset                    |     HMR                                                |  
| Training Parameters        | batch_size = 32, lr=0.0001              |  
| Optimizer                  | Adam                                                   |  
| outputs                    | probability                                                 |  
| Speed                      | 1pc: 1800 ms/step;                      | 1pc: 1700 ms/step;      |
| Checkpoint for Fine tuning | 176.07M (.ckpt file)                                         |

### Inference Performance

#### HMR infer on Human3.6m

| Parameters          | Ascend                      |  
| ------------------- | --------------------------- |  
| Model Version       | HMR                |  
| Resource            | Ascend 910; OS Euler2.8                  |  
| Uploaded Date       | 08/04/2022 (month/day/year) |  
| MindSpore Version   | 1.5.0                       |
| Dataset             | Human3.6m     |  
| batch_size          | 32                         |  
| outputs             | probability                 |  
| Reconst. Error            | 55   |  

# [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/models).
