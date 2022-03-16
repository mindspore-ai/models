#! /bin/bash
# shellcheck disable=SC2044
# shellcheck disable=SC2045
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
# ============================================================================
mkdir ../train
cd ../ILSVRC2013_DET_train
for file in `ls *tar`
do
    tar -xvf $file
done
for img in `find . -type f -name "*.JPEG"`
do
    mv $img ../train
done

for dir in `find . -type d -maxdepth 1`
do
    cd $dir
    for img in `find . -type f -name "*.JPEG"`
    do
        mv $img ../train
    done
    cd ../ILSVRC2013_DET_train
done