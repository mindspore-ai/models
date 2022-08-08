# Dataset preparation

You can download aishell dataset by this [link](http://www.openslr.org/33/)

AISHELL数据集需要用Kaldi工具包进行预处理，你可以：

一、下载已经用kaldi处理好的aishell数据集,链接如下：
    https://blog.csdn.net/m0_45973994/article/details/124891389

二、自己下载kaldi工具包，并处理，步骤如下:

1. cd your/path/to/prepare_aishell_data/

2. 下载aishell数据集[link](http://www.openslr.org/33/)，并解压

3. git clone https://github.com/kaituoxu/Speech-Transformer

4. 下载kaldi，git clone https://github.com/kaldi-asr/kaldi.git

5. 把your/path/to/prepare_aishell_data/Speech-Transformer-master/egs/aishell/local文件夹下的parse_options.sh文件复制到your/path/to/prepare_aishell_data/kaldi/egs/aishell/s5/local

6. 用your/path/to/prepare_aishell_data/Speech-Transformer-master/egs/aishell/下的run.sh替换your/path/to/prepare_aishell_data/kaldi/egs/aishell/s5/下的run.sh

7. 进入kaldi下的tools进行make
    7.1 执行 tools/extras/check_dependencies.sh
    7.2 根据提示安装automake、autoconf、sox、gfortran、subversion等依赖。（apt-get install automake autoconf sox gfortran subversion）
    7.3 执行tools/extras/install_mkl.sh
    7.4 若7.3报错，则改用安装openblas替代，即执行tools/extras/install_openblas.sh
    7.5 安装第三方工具openfast，命令：sudo make openfst
    7.6 在tools下执行make -j 8

8. 进入kaldi下的src进行make
    8.1 bash configure
    8.2 make depend -j 8
    8.3 make -j 8

9. 上述编译完成，开始执行run.sh
    9.1 cd /prepare_aishell_data/kaldi/egs/aishell/s5/
    9.2 把run.sh中的数据路径改为你自己的路径（your/path/to/prepare_aishell_data/data）
    9.3 执行run.sh
    9.4 stage 2步骤会出现错误，未发现bc命令。应安装bc，执行命令apt-get install bc

注：步骤7和8是为了生成fbank特征提取所需工具compute-fbank-feats、copy-feats等等。如果不进行make，源码中只有.cc文件，执行fbank特征提取（run.sh的第二步）时会找不到所需工具，会报错找不到compute-fbank-feats命令和copy-feats命令
