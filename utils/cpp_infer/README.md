## Description

When the C++ inference backend is MindSpore, the MindIR model exported from MindSpore can be deployed on Ascend310 and Ascend310P.
When the C++ inference backend is MindSpore Lite, MindIR models can be deployed on Ascend310/310P, Nvidia GPU and CPU.

In order to ensure that the script can run normally, environment variables need to be set before inference is executed.

## Environment variables on Ascend

### Check the installation path

If the root user is used to install the run package, the default path is '/usr/local/Ascend', and the default installation path for non-root users is '/home/HwHiAiUser/Ascend'.

Take the path of the root user as an example, set the environment variables as follows:

```bash
export ASCEND_HOME=/usr/local/Ascend  # the root directory of run package
```

### Identify the run package version

The run package is divided into two versions according to whether the 'ascend-toolkit' directory exists in the installation path.

If 'ascend-toolkit' directory exists, set the environment variables as follows:

```bash
export ASCEND_HOME=/usr/local/Ascend
export PATH=${ASCEND_HOME}/ascend-toolkit/latest/compiler/bin:${ASCEND_HOME}/ascend-toolkit/latest/compiler/ccec_compiler/bin/:${PATH}
export LD_LIBRARY_PATH=${ASCEND_HOME}/driver/lib64:${ASCEND_HOME}/ascend-toolkit/latest/lib64:${LD_LIBRARY_PATH}
export ASCEND_OPP_PATH=${ASCEND_HOME}/ascend-toolkit/latest/opp
export ASCEND_AICPU_PATH=${ASCEND_HOME}/ascend-toolkit/latest/
export PYTHONPATH=${ASCEND_HOME}/ascend-toolkit/latest/compiler/python/site-packages:${PYTHONPATH}
export TOOLCHAIN_HOME=${ASCEND_HOME}/ascend-toolkit/latest/toolkit
```

Another version:

```bash
export ASCEND_HOME=/usr/local/Ascend
export PATH=${ASCEND_HOME}/latest/compiler/bin:${ASCEND_HOME}/latest/compiler/ccec_compiler/bin:${PATH}
export LD_LIBRARY_PATH=${ASCEND_HOME}/driver/lib64:${ASCEND_HOME}/latest/lib64:${LD_LIBRARY_PATH}
export ASCEND_OPP_PATH=${ASCEND_HOME}/latest/opp
export ASCEND_AICPU_PATH=${ASCEND_HOME}/latest
export PYTHONPATH=${ASCEND_HOME}/latest/compiler/python/site-packages:${PYTHONPATH}
export TOOLCHAIN_HOME=${ASCEND_HOME}/latest/toolkit
```

## Environment variables on Nvidia GPU

When the hardware backend is Nvidia GPU, cuda and TensorRT are depended. You need to install cuda and TensorRT first.

The following uses cuda11.1 and TensorRT8.2.5.1 as examples. You need to set the environment variables based on the actual installation paths.

```bash
export CUDA_HOME=/usr/local/cuda-11.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export TENSORRT_PATH=/usr/local/TensorRT-8.2.5.1
export PATH=$TENSORRT_PATH/bin:$PATH
export LD_LIBRARY_PATH=$TENSORRT_PATH/lib:$LD_LIBRARY_PATH
```

## Set the Host log level

```bash
export GLOG_v=2 # 0-DEBUG, 1-INFO, 2-WARNING, 3-ERROR, 4-CRITICAL, default level is WARNING.
```

## Inference Backend

If the build script identifies the environment variable `MS_LITE_HOME`, MindSpore Lite is used as the inference backend.
Otherwise, MindSpore is used as the inference backend.

### Use MindSpore Lite as the inference backend

When the MindSpore Lite is used as the inference backend, MindIR models can be deployed on Ascend310/310P, Nvidia GPU and CPU.

Download the MindSpore Lite Ascend, GPU and CPU three-in-one tar package from the [official website](https://mindspore.cn/versions).
Decompress the tar package and set the `MS_LITE_HOME` environment variable to the decompressed path. For example:

```bash
export MS_LITE_HOME=$some_path/mindpsore-lite-2.0.0-linux-x64
```

### Use MindSpore as the inference backend

When the MindSpore Lite is used as the inference backend, MindIR models can be deployed on Ascend310 and Ascend310P.

Download the MindSpore three-in-on or the corresponding hardware environment whl package from the [official website](https://mindspore.cn/versions).
The C++ build script identifies the installation path of MindSpore whl as the path of the dependent header file and dynamic link library.

### Build and Run inference

Generally, the `cpp_infer` directory exists in each model script directory, such as `official/cv/resnet`. The `build.sh` script
is used to compile C++ inference program. The C++ main.cc file depends on the header files in the `utils/cpp_infer/common_inc` relative to the top
directory. If you copy the model script directory to another directory for compilation, you need to copy the `utils/cpp_infer/common_inc` directory
to the `cpp_infer` directory.

Using resnet as an example:

```bash
cp -r utils/cpp_infer/common_inc official/cv/resnet/cpp_infer/
cd official/cv/resnet/cpp_infer/src/
bash build.sh
```

Generally, the `scrpits` directory exists in each model script directory, such as `official/cv/resnet`. The `run_infer_cpp.sh`
is used for inference evaluation. You can set the `DEVICE_TYPE` environment variable to set the running inference hardware backend.
Currently, the following options are supported: `Ascend`, `GPU` and `CPU`.
