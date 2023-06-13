# ![MindSpore Logo](https://gitee.com/mindspore/mindspore/raw/master/docs/MindSpore-logo.png)

## 欢迎来到MindSpore ModelZoo

MindSpore models仓中提供了不同任务领域，经典的SOTA模型实现和端到端解决方案。目的是方便MindSpore用户更加方便的利用MindSpore进行研究和产品开发。

为了让开发者更好地体验MindSpore框架优势，我们将陆续增加更多的典型网络和相关预训练模型。如果您对ModelZoo有任何需求，请通过[Gitee](https://gitee.com/mindspore/mindspore/issues)或[MindSpore](https://bbs.huaweicloud.com/forum/forum-1076-1.html)与我们联系，我们将及时处理。

| 目录                     | 描述                                                         |
|------------------------| ------------------------------------------------------------ |
| [official](official)   | • 业界SOTA算法模型实现<br/> • MindSpore团队官方维护|
| [research](research)   | • 业界前沿研究类算法模型实现 <br/> • 研究人员/机构维护 |
| [community](community) | • github/gitee 生态AI/ML repos powered by MindSpore |

## WHAT IS NEW

我们对经典SOTA模型进行了重构，模块化数据处理，模型定义，训练流程等常用组件，推出MindSpore CV/NLP/Audio/Yolo/OCR等系列，[参考](https://github.com/mindspore-lab)。

原models仓模型实现是基于MindSpore原生API，并且有一定训练推理加速优化。

## 免责声明

MindSpore仅提供下载和预处理公共数据集的脚本。我们不拥有这些数据集，也不对它们的质量负责或维护。请确保您具有在数据集许可下使用该数据集的权限。在这些数据集上训练的模型仅用于非商业研究和教学目的。

致数据集拥有者：如果您不希望将数据集包含在MindSpore中，或者希望以任何方式对其进行更新，我们将根据要求删除或更新所有公共内容。请通过GitHub或Gitee与我们联系。非常感谢您对这个社区的理解和贡献。

MindSpore已获得Apache 2.0许可，请参见LICENSE文件。

## 许可证

[Apache 2.0许可证](https://gitee.com/mindspore/mindspore/blob/master/LICENSE)

## FAQ

想要获取更多关于`MindSpore`框架使用本身的FAQ问题的，可以参考[官网FAQ](https://www.mindspore.cn/docs/zh-CN/master/faq/installation.html)

- **Q: 直接使用models下的模型出现内存不足错误，例如*Failed to alloc memory pool memory*, 该怎么处理?**

  **A**: 直接使用models下的模型出现内存不足的典型原因是由于运行模式（`PYNATIVE_MODE`)、运行环境配置、License控制（AI-TOKEN）的不同造成的：
    - `PYNATIVE_MODE`通常比`GRAPH_MODE`使用更多内存，尤其是在需要进行反向传播计算的训练图中，当前有2种方法可以尝试解决该问题。
        方法1：你可以尝试使用一些更小的batch size；
        方法2：添加context.set_context(mempool_block_size="XXGB")，其中，“XX”当前最大有效值可设置为“31”。
        如果将方法1与方法2结合使用，效果会更好。
    - 运行环境由于NPU的核数、内存等配置不同也会产生类似问题。
    - License控制（AI-TOKEN）的不同档位会造成执行过程中内存开销不同，也可以尝试使用一些更小的batch size。

- **Q: 一些网络运行中报错接口不存在，例如cannot import，该怎么处理?**

  **A**: 优先检查一下获取网络脚本的分支，与所使用的MindSpore版本是否一致，部分新分支中的模型脚本会使用一些新版本MindSpore才支持的接口，从而在使用老版本MindSpore时会发生报错.

- **Q: 一些模型描述中提到的*RANK_TABLE_FILE*文件，是什么？**

  **A**: *RANK_TABLE_FILE*是一个Ascend环境上用于指定分布式集群信息的文件，更多信息可以参考生成工具[hccl_toos](https://gitee.com/mindspore/models/tree/r2.0/utils/hccl_tools)和[分布式并行训练教程](https://mindspore.cn/docs/programming_guide/zh-CN/r1.5/distributed_training_ascend.html#id4)

- **Q: 在windows环境上要怎么运行网络脚本？**

  **A**: 多数模型都是使用bash作为启动脚本，在Windows环境上无法直接使用bash命令，你可以考虑直接运行python命令而不是bash启动脚本 ，如果你确实想需要使用bash脚本，你可以考虑使用以下几种方法来运行模型：
    1. 使用虚拟环境，可以构造一个linux的虚拟机或docker容器，然后在虚拟环境中运行脚本
    2. 使用WSL，可以开启Windows的linux子系统来在Windows系统中运行linux，然后再WSL中运行脚本。
    3. 使用Windows Bash，需要获取一个可以直接在Windows上运行bash的环境，常见的选择是[cygwin](http://www.cygwin.com)或[git bash](https://gitforwindows.org)
    4. 跳过bash脚本，直接调用python程序。

- **Q: 网络在310推理时出现编译失败，报错信息指向gflags，例如*undefined reference to 'google::FlagRegisterer::FlagRegisterer'*，该怎么处理?**

  **A**: 优先检查一下环境GCC版本和gflags版本是否匹配，可以参考[官方链接](https://www.mindspore.cn/install)安装对应的GCC版本，[gflags](https://github.com/gflags/gflags/archive/v2.2.2.tar.gz)安装gflags。你需要保证所使用的组件之间是ABI兼容的，更多信息可以参考[_GLIBCXX_USE_CXX11_ABI](https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html)

- **Q: 在Mac系统上加载mindrecord格式的数据集出错,例如*Invalid file, failed to open files for reading mindrecord files.*，该怎么处理?**

  **A**: 优先使用*ulimit -a*检查系统限制，如果*file descriptors*数量为256（默认值），需要使用*ulimit -n 1024*将其设置为1024（或者更大的值）。之后再检查文件是否损坏或者被修改。

- **Q: 我在多台服务器构成的大集群上进行训练，但是得到的精度比预期要低，该怎么办？**

  **A**: 当前模型库中的大部分模型只在单机内进行过验证，最大使用8卡进行训练。由于MindSpore训练时指定的`batch_size`是单卡的，所以当单机8卡升级到多机时，会导致全局的`global_batch_size`变大，这就导致需要针对当前多机场景的`global_batch_size`进行重新调参优化。
