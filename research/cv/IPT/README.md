<TOC>

# Pre-Trained Image Processing Transformer (IPT)

This repository is an official implementation of the paper "Pre-Trained Image Processing Transformer" from CVPR 2021.

We study the low-level computer vision task (e.g., denoising, super-resolution and deraining) and develop a new pre-trained model, namely, image processing transformer (IPT). To maximally excavate the capability of transformer, we present to utilize the well-known ImageNet benchmark for generating a large amount of corrupted image pairs. The IPT model is trained on these images with multi-heads and multi-tails. In addition, the contrastive learning is introduced for well adapting to different image processing tasks. The pre-trained model can therefore efficiently employed on desired task after fine-tuning. With only one pre-trained model, IPT outperforms the current state-of-the-art methods on various low-level benchmarks.

If you find our work useful in your research or publication, please cite our work:
[1] Hanting Chen, Yunhe Wang, Tianyu Guo, Chang Xu, Yiping Deng, Zhenhua Liu, Siwei Ma, Chunjing Xu, Chao Xu, and Wen Gao. **"Pre-trained image processing transformer"**. <i>**CVPR 2021**.</i> [[arXiv](https://arxiv.org/abs/2012.00364)]

    @inproceedings{chen2020pre,
      title={Pre-trained image processing transformer},
      author={Chen, Hanting and Wang, Yunhe and Guo, Tianyu and Xu, Chang and Deng, Yiping and Liu, Zhenhua and Ma, Siwei and Xu, Chunjing and Xu, Chao and Gao, Wen},
      booktitle={CVPR},
      year={2021}
     }

## Model architecture

### The overall network architecture of IPT is shown as below

![architecture](./image/ipt.png)

## Dataset

The benchmark datasets can be downloaded as follows:

For super-resolution:

 Set5,
[Set14](https://sites.google.com/site/romanzeyde/research-interests),
[B100](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/),
Urban100.

For denoising:

[CBSD68](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/).

For deraining:

[Rain100L](https://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html)

The result images are converted into YCbCr color space. The PSNR is evaluated on the Y channel only.

## Requirements

### Hardware (Ascend)

> Prepare hardware environment with Ascend.

### Framework

> [MindSpore](https://www.mindspore.cn/install/en)

### For more information, please check the resources below

[MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
[MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

## Script Description

> This is the inference script of IPT, you can following steps to finish the test of image processing tasks, like SR, denoise and derain, via the corresponding pretrained models.

### Scripts and Sample Code

```bash
IPT
├── eval.py # inference entry
├── train.py # pre-training entry
├── train_finetune.py # fine-tuning entry
├── export_onnx.py # export onnx format script
├── eval_onnx.py # exported onnx evaluation script
├── image
│   └── ipt.png # the illustration of IPT network
├── readme.md # Readme
├── scripts
│   ├── run_eval.sh # inference script for all tasks
│   ├── run_dr_eval_onnx_gpu.sh # exported onnx eval in gpu for deraining tasks
│   ├── run_dn_eval_onnx_gpu.sh # exported onnx eval in gpu for denoising tasks
│   ├── run_sr_eval_onnx_gpu.sh # exported onnx eval in gpu for super resolution tasks
│   ├── run_distributed.sh # pre-training script for all tasks
│   ├── run_dr_distributed.sh # pre-training script for deraining tasks
│   ├── run_dn_distributed.sh # pre-training script for denoising tasks
│   └── run_sr_distributed.sh # fine-tuning script for super resolution tasks
└── src
    ├── args.py # options/hyper-parameters of IPT
    ├── data
    │   ├── common.py # common dataset
    │   ├── bicubic.py # scripts for data pre-processing
    │   ├── div2k.py # DIV2K dataset
    │   ├── imagenet.py # Imagenet data for pre-training
    │   └── srdata.py # All dataset
    ├── metrics.py # PSNR calculator
    ├── utils.py # training scripts
    ├── loss.py # contrastive_loss
    ├── ipt_post_onnx.py # IPT post for onnx
    └── ipt_model.py # IPT network
```

### Script Parameter

```python
'task_id'
# 0 corresponds to the super-resolution ×2 task, 1 corresponds to the super-resolution ×3 task, 2 corresponds to the super-resolution ×4 task, 3 corresponds to the deraining task, 4 corresponds to the denoising task of 30, 5 corresponds to the denoising task of 50.
# TASK_ ID is set to 0 for all tasks during evaluation
```

> For more details about hyperparameters, see src/args.py.

## Training Process

### For pre-training

```bash
python train.py --distribute --imagenet 1 --batch_size 64 --lr 5e-5 --scale 2+3+4+1+1+1 --alltask --react --model vtip --num_queries 6 --chop_new --num_layers 4 --data_train imagenet --dir_data $DATA_PATH --derain --save $SAVE_PATH
```

> Or one can run following script for all tasks.

```bash
sh scripts/run_distributed.sh RANK_TABLE_FILE DATA_PATH
```

### For fine-tuning

> For SR tasks:

```bash
python train_finetune.py --distribute --imagenet 0 --batch_size 64 --lr 2e-5 --scale 2+3+4+1+1+1 --model vtip --num_queries 6 --chop_new --num_layers 4 --task_id $TASK_ID --dir_data $DATA_PATH --pth_path $MODEL --epochs 50
```

> For Denoising tasks:

```bash
python train_finetune.py --distribute --imagenet 0 --batch_size 64 --lr 2e-5 --scale 2+3+4+1+1+1 --model vtip --num_queries 6 --chop_new --num_layers 4 --task_id $TASK_ID --dir_data $DATA_PATH --pth_path $MODEL --denoise --sigma $Noise --epochs 50
```

> For deraining tasks:

```bash
python train_finetune.py --distribute --imagenet 0 --batch_size 64 --lr 2e-5 --scale 2+3+4+1+1+1 --model vtip --num_queries 6 --chop_new --num_layers 4 --task_id $TASK_ID --dir_data $DATA_PATH --pth_path $MODEL --derain --epochs 50
```

> Or one can run following script for all tasks.

```bash
sh scripts/run_sr_distributed.sh RANK_TABLE_FILE DATA_PATH MODEL TASK_ID
sh scripts/run_dn_distributed.sh RANK_TABLE_FILE DATA_PATH MODEL TASK_ID
sh scripts/run_dr_distributed.sh RANK_TABLE_FILE DATA_PATH MODEL TASK_ID
```

## Evaluation

### Evaluation Process

```text
This project refers to the ckpt file on the hub of the official website of MindSpore. Some parameters of the ckpt file cannot be imported, which leads to the low precision of the ckpt.
The reason is that the names of some parameters of the ckpt file cannot be mapped. Modify the parameter names to import.
The modified Ckpt acquisition method is as follows:

Baidu online disk self fetching link: https://pan.baidu.com/s/1AoaczO3bNtM12oxS0Z8Ogg
Extraction code: jtqo
```

> Inference example:
> For SR x4:

```bash
python eval.py --dir_data $DATA_PATH --data_test $DATA_TEST --test_only --ext img --pth_path $MODEL --task_id 0 --scale $SCALE --num_queries 1
```

> Or one can run following script for all tasks.

```bash
sh scripts/run_eval.sh DATA_PATH DATA_TEST MODEL TASK_ID
```

### Evaluation Result

The result are evaluated by the value of PSNR (Peak Signal-to-Noise Ratio), and the format is as following.

```bash
result: {"Mean psnr of Set5 x4 is 32.71"}
```

## ONNX Export And Evaluation

Note that run all onnx concerned scripts on GPU.

### ONNX Export

The command below will produce lots of IPT onnx files, named different input shapes due to different input shapes of evaluation data.

> For SR tasks:

```bash
python export_onnx.py --pth_path $CKPT_PATH --data_test $DATA_TEST --dir_data $DATA_PATH --file_name IPT --file_format ONNX --save $ONNX_SAVE_PATH --task_id 0 --scale $SCALE --test_only --ext img --num_queries 1
# example for SR x2: python export_onnx.py --pth_path ./checkpoint/ipt_x2_ascend_v180_set5_research_cv_acc38.3.ckpt --data_test set5 --dir_data ./cache/data/ --file_name IPT --file_format ONNX --save ./onnx/ --task_id 0 --scale 2 --test_only --ext img --num_queries 1
```

> For Denoising tasks:

```bash
python export_onnx.py --pth_path $CKPT_PATH --data_test $DATA_TEST --dir_data $DATA_PATH --file_name IPT --file_format ONNX --save $ONNX_SAVE_PATH --task_id 0 --scale $SCALE --test_only --ext img --denoise --sigma 30 --num_queries 1
# example for denoising: python export_onnx.py --pth_path ./checkpoint/ipt_denoise30_ascend_v180_cbsd68_research_cv_acc32.3.ckpt --data_test CBSD68 --dir_data ./cache/data/ --file_name IPT --file_format ONNX --save ./onnxdn30/ --task_id 0 --scale 1 --test_only --ext img --denoise --sigma 30 --num_queries 1
```

> For deraining tasks:

```bash
python export_onnx.py --pth_path $CKPT_PATH --data_test $DATA_TEST --dir_data $DATA_PATH --file_name IPT --file_format ONNX --save $ONNX_SAVE_PATH --task_id 0 --scale $SCALE --test_only --ext img --derain --num_queries 1
# example for deraining: python export_onnx.py --pth_path ./checkpoint/ipt_derain_ascend_v180_rain100l_research_cv_acc42.0.ckpt --data_test Rain100L --dir_data ./cache/data/ --file_name IPT --file_format ONNX --save ./onnxdr/ --task_id 0 --scale 1 --test_only --ext img --derain --num_queries 1
```

### ONNX Evaluation

Note that ONNX_PATH should be the absolute directory to the exported onnx files, such as: './onnx/'.

> For SR tasks:

```bash
python eval_onnx.py --dir_data [DATA_PATH]
                    --data_test [DATASET]
                    --pth_path [ONNX_FILE_PATH]
                    --test_only
                    --ext img
                    --scale [SCALE]
                    --task_id 0 > eval_onnx.log 2>&1 &
OR
bash scripts/run_sr_eval_onnx_gpu.sh [ONNX_FILE_PATH] [DATASET] [DATA_PATH] [SCALE]
```

> For Denoising tasks:

```bash
python eval_onnx.py --dir_data [DATA_PATH]
                    --data_test [DATASET]
                    --pth_path [ONNX_FILE_PATH]
                    --test_only
                    --ext img
                    --scale 1
                    --denoise
                    --sigma [SIGMA]
                    --task_id 0 > eval_onnx.log 2>&1 &
OR
bash scripts/run_dn_eval_onnx_gpu.sh [ONNX_FILE_PATH] [DATASET] [DATA_PATH] [SIGMA]
```

> For deraining tasks:

```bash
python eval_onnx.py --dir_data [DATA_PATH]
                    --data_test [DATASET]
                    --pth_path [ONNX_FILE_PATH]
                    --test_only
                    --ext img
                    --scale 1
                    --derain
                    --task_id 0 > eval_onnx.log 2>&1 &
OR
bash scripts/run_dr_eval_onnx_gpu.sh [ONNX_FILE_PATH] [DATASET] [DATA_PATH]
```

You can view the results through the file "eval_onnx". The accuracy of the test dataset will be as follows:

```text
current psnr:  25.76683705372035
current psnr:  24.81874580003199
current psnr:  23.47910049750682
current psnr:  25.253800698303067
current psnr:  24.73868258923604
Mean psnr of set5 x2 is 24.8114
```

## Performance

### Inference Performance

The Results on all tasks are listed as below.

Super-resolution results:

| Scale | Set5 | Set14 | B100 | Urban100 |
| ----- | ----- | ----- | ----- | ----- |
| ×2    | 38.33 | 34.49 | 32.46 | 33.74 |
| ×3    | 34.86 | 30.85 | 29.38 | 29.50 |
| ×4    | 32.71 | 29.03 | 27.84 | 27.24 |

Denoising results:

| noisy level | CBSD68 | Urban100 |
| ----- | ----- | ----- |
| 30    | 32.35 | 33.99 |
| 50    | 29.93 | 31.49 |

Derain results:

| Task | Rain100L |
| ----- | ----- |
| Derain   | 42.08 |

## ModeZoo Homepage

Please check the official [homepage](https://gitee.com/mindspore/models).
