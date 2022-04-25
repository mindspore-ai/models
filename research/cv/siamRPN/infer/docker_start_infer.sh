#!/bin/bash
docker_image=$1
share_dir=$2
echo "$1"
echo "$2"
if [ -z "${docker_image}" ]; then
    echo "please input docker_image"
    exit 1
fi

if [ ! -d "${share_dir}" ]; then
    echo "please input share directory that contains mxManufacture and models codes"
    exit 1
fi

docker run -it \
    --device=/dev/davinci0 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    --privileged \
    -v //usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v ${share_dir}:${share_dir} \
    ${docker_image} \
    /bin/bash