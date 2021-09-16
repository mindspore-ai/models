#!/usr/bin/env bash

docker_image=$1
data_dir=$2

function show_help() {
    echo "Usage: docker_start.sh docker_image data_dir"
}

function param_check() {
    if [ -z "${docker_image}" ]; then
        echo "please input docker_image"
        show_help
        exit 1
    fi

    if [ -z "${data_dir}" ]; then
        echo "please input data_dir"
        show_help
        exit 1
    fi
}

param_check

docker run -it \
  --device=/dev/davinci1 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v ${data_dir}:${data_dir} \
  ${docker_image} \
  /bin/bash
