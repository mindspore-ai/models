#!/usr/bin/env bash

# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

docker_image=$1
model_dir=$2


function show_help() {
    echo "Usage: docker_start.sh docker_image model_dir data_dir"
}

function param_check() {
    if [ -z "${docker_image}" ]; then
        echo "please input docker_image"
        show_help
        exit 1
    fi

    if [ -z "${model_dir}" ]; then
        echo "please input model_dir"
        show_help
        exit 1
    fi
}

param_check

docker run -it -u root \
  --device=/dev/davinci0 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v ${model_dir}:${model_dir} \
  ${docker_image} \
  /bin/bash
