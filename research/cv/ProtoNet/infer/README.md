# Description

This README file is to show how to inference Protonet by mxBase and mindX-SDK

# Environment Preparation

- (ALL Required) You should put `infer` folder into **Server Environment**  not must in subfolder of mxVision.
- (Convert Required) You must configure the environment variables correctly like [here](https://support.huaweicloud.com/atctool-cann503alpha1infer/atlasatc_16_0004.html), if you use docker you may skip this step.
- (mxBase mindX-SDK Required) You must config the environment parameter. for example:

    ```bash
    export MX_SDK_HOME="/home/data/xj_mindx/mxVision"
    export ASCEND_HOME=/usr/local/Ascend
    export ASCEND_VERSION=nnrt/latest
    export ARCH_PATTERN=.
    export MXSDK_OPENSOURCE_DIR=/home/data/xj_mindx/mxVision/opensource
    export LD_LIBRARY_PATH="${MX_SDK_HOME}/lib/plugins:${MX_SDK_HOME}/opensource/lib64:${MX_SDK_HOME}/lib:${MX_SDK_HOME}/lib/modelpostprocessors:${MX_SDK_HOME}/opensource/lib:/usr/local/Ascend/nnae/latest/fwkacllib/lib64:${LD_LIBRARY_PATH}"
    export ASCEND_OPP_PATH="/usr/local/Ascend/nnae/latest/opp"
    export ASCEND_AICPU_PATH="/usr/local/Ascend/nnae/latest"
    ```

# Model Convert

we offer a bash file `convert.sh`  that can help you to easy convert model from AIR to OM, it was placed in `convert` . for example:

```bash
bash convert.sh
```

If you want to see the help message of the bash file, you can use:

```bash
bash convert.sh --help
```

You will see the help and the default setting of the args.

# Input image

You must put the **Omniglot Dataset** into `infer/input/dataset` folder.and put the dataset after processed into `data/input`.

e.g. **Original**

```shell
└─dataset
    ├─raw
    ├─spilts
    │     vinyals
    │         test.txt
    │         train.txt
    │         val.txt
    │         trainval.txt
    └─data
           Alphabet_of_the_Magi
           Angelic
```

**Procession:** we offer a bash file (`convert/dataprocess.sh`) to process the dataset:

```bash
bash dataprocess.sh
```

e.g. **Processed**

```shell
└─data
    ├─dataset
    ├─data_preprocess_Result
    │     data_1.bin
    │     data_2.bin
    |         ···
    │     data_100.bin
    └─label_classes_preprocess_Result
          label_1.bin
          label_1.bin
              ···
          label_100.bin
          classes_1.bin
          classes_2.bin
              ···
          classes_100.bin

```

# Infer by mxBase

You should put OM file into `data/model`, then you need build the project by `build.sh`, for example:

```bash
cd mxbase
bash build.sh
```

if success, you should see a new file named `protonet`, then you can use command to infer:

```bash
./protonet
```

Inference result will store in folder `result`.

# Infer by mindX-SDK

if you want to infer by mindx-SDK, you should enter the folder `infer/sdk` and then use the shell command:

```bash
bash run.sh ../data/config/protonet.pipeline ../data/input/data_preprocess_Result/ ../data/input/label_classes_preprocess_Result
```

you will acquire the inference result in folder `result`.

# Calculate Inference Precision

We offer a python file to calculate the precision.

```bash
python postprocess.py --result_path=./infer/XXX/result
                      --label_classes_path=./infer/data/input/label_classes_preprocess_Result
```

**note:**

- XXX can be `mxbase` or `sdk`
- `--label_classes_path` is the label and class data after preprocessed.

# Self-Inspection Report

- We have obtained the following result through mindX-SDK and mxBase inference:
    || Accuracy|||  |   |
    |:----:| :----:|:----:|:----:| :----: | :----: |
    |mindX-SDK| 0.9943  |
    |mxBase| 0.9943  |

- The model precision in train:

    | | Accuracy  |   |
    | :----: | :----: | :----: |
    | Train | 0.9954  |

# ModelZoo Homepage

 Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
