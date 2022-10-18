# 简介

为了确保脚本能够正常地运行在昇腾310/310P，需要在执行推理前设置环境变量。

## 确认run包安装路径

若使用root用户完成run包安装，默认路径为'/usr/local/Ascend'，非root用户的默认安装路径为'/home/HwHiAiUser/Ascend'。

以root用户的路径为例，设置环境变量如下：

```bash
export ASCEND_HOME=/usr/local/Ascend  # the root directory of run package
```

## 区分run包版本

run包分为2个版本，用安装目录下是否存在'ascend-toolkit'文件夹进行区分。

如果存在'ascend-toolkit'文件夹，设置环境变量如下：

```bash
export ASCEND_HOME=/usr/local/Ascend
export PATH=${ASCEND_HOME}/ascend-toolkit/latest/compiler/bin:${ASCEND_HOME}/ascend-toolkit/latest/compiler/ccec_compiler/bin/:${PATH}
export LD_LIBRARY_PATH=${ASCEND_HOME}/driver/lib64:${ASCEND_HOME}/ascend-toolkit/latest/lib64:${LD_LIBRARY_PATH}
export ASCEND_OPP_PATH=${ASCEND_HOME}/ascend-toolkit/latest/opp
export ASCEND_AICPU_PATH=${ASCEND_HOME}/ascend-toolkit/latest/
export PYTHONPATH=${ASCEND_HOME}/ascend-toolkit/latest/compiler/python/site-packages:${PYTHONPATH}
export TOOLCHAIN_HOME=${ASCEND_HOME}/ascend-toolkit/latest/toolkit
```

若不存在，设置环境变量为：

```bash
export ASCEND_HOME=/usr/local/Ascend
export PATH=${ASCEND_HOME}/latest/compiler/bin:${ASCEND_HOME}/latest/compiler/ccec_compiler/bin:${PATH}
export LD_LIBRARY_PATH=${ASCEND_HOME}/driver/lib64:${ASCEND_HOME}/latest/lib64:${LD_LIBRARY_PATH}
export ASCEND_OPP_PATH=${ASCEND_HOME}/latest/opp
export ASCEND_AICPU_PATH=${ASCEND_HOME}/latest
export PYTHONPATH=${ASCEND_HOME}/latest/compiler/python/site-packages:${PYTHONPATH}
export TOOLCHAIN_HOME=${ASCEND_HOME}/latest/toolkit
```

## 设置Host侧日志级别

```bash
export GLOG_v=2 # 0-DEBUG, 1-INFO, 2-WARNING, 3-ERROR, 4-CRITICAL, default level is WARNING.
```
