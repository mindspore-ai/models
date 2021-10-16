#!/bin/bash
docker_image=$1
model_dir=$2

if [ -z "${docker_image}" ]; then
    echo "please input docker_image"
    exit 1
fi

if [ ! -d "${model_dir}" ]; then
    echo "please input model_dir"
    exit 1
fi

docker run -it \
            --device=/dev/davinci1 \
            --device=/dev/davinci_manager \
            --device=/dev/devmm_svm \
            --device=/dev/hisi_hdc \
            -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
            -v ${model_dir}:${model_dir} \
            ${docker_image} \
            /bin/bash