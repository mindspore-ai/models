ARG FROM_IMAGE_NAME
FROM ${FROM_IMAGE_NAME}

# 配置镜像代理
ENV http_proxy="http://192.168.88.254:8080"
ENV https_proxy="http://192.168.88.254:8080"

# 添加用户以及用户组 username ID为当前用户
RUN useradd -d /home/hwMindX -u 9000 -m -s /bin/bash hwMindX && \
    useradd -d /home/HwHiAiUser -u 1000 -m -s /bin/bash HwHiAiUser && \
    useradd -d /home/sjtu_liu -u 1001 -m -s /bin/bash sjtu_liu -g HwHiAiUser && \
    usermod -a -G HwHiAiUser hwMindX
# 添加Python符号链接
RUN ln -s  /usr/local/python3.7.5/bin/python3.7 /usr/bin/python
# 安装相关依赖包，根据实际模型依赖修改
RUN apt-get update && \
    apt-get install libglib2.0-dev -y || \
    rm -rf /var/lib/dpkg/info && \
    mkdir /var/lib/dpkg/info && \
    apt-get install libglib2.0-dev dos2unix -y && \
    pip install pytest-runner==5.3.0
# 安装Python依赖包，根据实际模型依赖修改
COPY requirements.txt .
RUN pip3.7 install -r requirements.txt