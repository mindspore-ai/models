# Pix2Pix

## 1. Prepare

### 1.1 Dataset

- facades datasets

  The Facades dataset contains the building exterior wall images collected by the Machine Perception Center and corresponding annotations. The training dataset contains 400 images and the test contains 106 images. The dataset can be download from [here] (http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/).

- maps datasets

  The Maps dataset consists of satellite imagery collected by Google Maps and Maps, 1,096 images in the training set and 1,098 images in the test set. The dataset can be download from [here] (http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/).

The dataset should be downloaded and unzip as follows(take facades for example).

```bash
├─ PATH_TO_DATASET
  ├─ facades
    ├─ train
    ├─ val
    ├─ test

```

note: It's recommended that maps dataset be organized in the same way.

### 1.2 create folders

```bash
Pix2Pix_MindSpore_{version}_code
├──infer                    # MindX high performance pre-training model added
  └─convert                 # convert air model to om model
    └─convert_om.sh
  └─docker_start_infer
  └─export_for_infer.py     # export .air model
  └─data                    # Including .air model files,.om model files, inference test pictures, inference output results (note: this directory and its subdirectories need to be manually created before inference)
    └─test_img              # image for infer(It must be created manually before infer)
    └─sdk_result            # sdk results(It must be created manually before infer)
    └─mxbase_result         # mxbase results(It must be created manually before infer)
    └─.air model            # .air model
    └─.om model             # .om model
  └─sdk                     # sdk infer
    └─main.py
    └─Pix2Pix.pipeline
    └─run.sh
  └─mxbase                  # mxbase infer
    └─src
        └─mxbase
        └─mxbase
        └─mxbase
    └─build.sh
    └─CMakeLists
```

### 1.3 Export

```bash
cd /PATH/TO/Pix2Pix/infer/
python3 export_for_infer.py --ckpt [The path of the CKPT for exporting] --train_data_dir [The path of the training dataset]"
```

### 1.4 Convert

convert air to om

```bash
cd /PATH/TO/Pix2Pix/infer/convert
bash convert_om.sh ../data/Pix2Pix_for_facades.air ../data/Pix2Pix_for_facades
```

## 2. Infer

### 2.1 MxBase

- 1.Compile project (produce./Pix2Pix execution file for next model infer)

```bash
cd /PATH/TO/Pix2Pix/infer/mxbase/
bash build.sh
```

- 2.model infer

```bash
./Pix2Pix image_path

| args       | explanation                                                   |
| ---------- | ------------------------------------------------------------- |
| image_path | path to the images for MxBase infer,like：“../data/test_img”  |

```

- 3.result

Image results are saved as *.jpg in.. /data/mxbase_result folder.

### 2.2 SDK

- 1.model infer

```bash
cd infer/sdk/
bash run.sh
```

- 2.result

Image results are saved as *.jpg in.. /data/sdk_result folder.