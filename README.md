# ![MindSpore Logo](https://gitee.com/mindspore/mindspore/raw/master/docs/MindSpore-logo.png)

## Welcome to the Model Zoo for MindSpore

In order to facilitate developers to enjoy the benefits of MindSpore framework, we will continue to add typical networks and some of the related pre-trained models. If you have needs for the model zoo, you can file an issue on [gitee](https://gitee.com/mindspore/mindspore/issues) or [MindSpore](https://bbs.huaweicloud.com/forum/forum-1076-1.html), We will consider it in time.

- SOTA models using the latest MindSpore APIs

- The  best benefits from MindSpore

- Officially maintained and supported

## Table of Contents

### DynamicShape

- [Community](https://gitee.com/mindspore/models/tree/master/community)

## Announcements

### 2022.12 Set up dynamic_shape branch of repository `models`

`models` comes from the directory `model_zoo` of repository [mindspore](https://gitee.com/mindspore/mindspore). This new repository doesn't contain any history of commits about the directory `model_zoo` in `mindspore`, you could refer to the repository `mindspore` for the past commits.

## Related Website

Here is the ModelZoo for MindSpore which support different devices including Ascend, GPU, CPU and mobile.

If you are looking for exclusive models only for Ascend using different ML platform, you could refer to [Ascend ModelZoo](https://hiascend.com/software/modelzoo) and corresponding [gitee repository](https://gitee.com/ascend/modelzoo)

If you are looking for some pretrained checkpoint of mindspore, you could refer to [MindSpore Hub](https://www.mindspore.cn/resources/hub/en) or [Download Website](https://download.mindspore.cn/model_zoo/).

## Disclaimers

Mindspore only provides scripts that downloads and preprocesses public datasets. We do not own these datasets and are not responsible for their quality or maintenance. Please make sure you have permission to use the dataset under the dataset’s license. The models trained on these dataset are for non-commercial research and educational purpose only.

To dataset owners: we will remove or update all public content upon request if you don’t want your dataset included on Mindspore, or wish to update it in any way. Please contact us through a Github/Gitee issue. Your understanding and contribution to this community is greatly appreciated.

MindSpore is Apache 2.0 licensed. Please see the LICENSE file.

## License

[Apache License 2.0](https://gitee.com/mindspore/mindspore/blob/master/LICENSE)

## FAQ

For more information about `MindSpore` framework, please refer to [FAQ](https://www.mindspore.cn/docs/en/master/faq/installation.html)

- **Q: How to resolve the lack of memory while using the model directly under "models" with errors such as *Failed to alloc memory pool memory*?**

  **A**: The typical reason for insufficient memory when directly using models under "models" is due to differences in operating mode (`PYNATIVE_MODE`), operating environment configuration, and license control (AI-TOKEN).
    - `PYNATIVE_MODE` usually uses more memory than `GRAPH_MODE` , especially in the training graph that needs back propagation calculation, there are two ways to try to solve this problem.
        Method 1: You can try to use some smaller batch size;
        Method 2: Add context.set_context(mempool_block_size="XXGB"), where the current maximum effective value of "XX" can be set to "31".
        If method 1 and method 2 are used in combination, the effect will be better.
    - The operating environment will also cause similar problems due to the different configurations of NPU cores, memory, etc.;
    - Different gears of License control (AI-TOKEN ) will cause different memory overhead during execution. You can also try to use some smaller batch sizes.

- **Q: How to resolve the error about the interface are not supported in some network operations, such as `cann not import`?**

  **A**: Please check the version of MindSpore and the branch you fetch the modelzoo scripts. Some model scripits in latest branch will use new interface in the latest version of MindSpore.

- **Q: What is Some *RANK_TBAL_FILE* which mentioned in many models?**

  **A**: *RANK_TABLE_FILE* is the config file of cluster on Ascend while running distributed training. For more information, you could refer to the generator [hccl_tools](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools) and [Parallel Distributed Training Example](https://mindspore.cn/docs/programming_guide/en/r1.5/distributed_training_ascend.html#configuring-distributed-environment-variables)

- **Q: How to run the scripts on Windows system?**

  **A**: Most the start-up scripts are written in `bash`, but we usually can't run bash directly on Windows. You can try start python directly without bash scripts. If you really need the start-up bash scripts, we suggest you the following method to get a bash environment on Windows:
    1. Use a virtual system or docker container with linux system. Then run the scripts in the virtual system or container.
    2. Use WSL, you could turn on the `Windows Subsystem for Linux` on Windows to obtain an linux system which could run the bash scripts.
    3. Use some bash tools on Windows, such as [cygwin](http://www.cygwin.com) and [git bash](https://gitforwindows.org).

- **Q: How to resolve the compile error point to gflags when infer on ascend310 with errors such as *undefined reference to 'google::FlagRegisterer::FlagRegisterer'*?**

  **A**: Please check the version of GCC and gflags. You can refer to [GCC](https://www.mindspore.cn/install) and [gflags](https://github.com/gflags/gflags/archive/v2.2.2.tar.gz) to install GCC and gflags. You need to ensure that the components used are ABI compatible, for more information, please refer to [_GLIBCXX_USE_CXX11_ABI](https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html).

- **Q: How to solve the error when loading dataset in mindrecord format on Mac system, such as *Invalid file, failed to open files for reading mindrecord files.*?**

  **A**: Please check the system limit with *ulimit -a*, if the number of *file descriptors* is 256 (default), you need to use *ulimit -n 1024* to set it to 1024 (or larger). Then check whether the file is damaged or modified.

- **Q: What should I do if I can't reach the accuracy while training with several servers instead of a single server?**

  **A**: Most of the models has only been trained on single server with at most 8 pcs. Because the `batch_size` used in MindSpore only represent the batch size of single GPU/NPU, the `global_batch_size` will increase while training with multi-server. Different `gloabl_batch_size` requires different hyper parameter including learning_rate, etc. So you have to optimize these hyperparameters will training with multi-servers.
