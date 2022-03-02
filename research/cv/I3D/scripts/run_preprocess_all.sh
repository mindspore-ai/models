#!/bin/bash
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

#Batch execution pre-processing
if [ $# != 0 ]; then
  echo "Usage: This script does not require any parameters to be entered, please go to the script to modify the specific variables"
  exit 1
fi

function preprocess() {
  #  Parameters can be changed here
  bash ./scripts/run_preprocess.sh 8 hccl_8p.json hmdb51 rgb 40 ./rgb/hmdb51/jpg ./rgb/hmdb51/annotation/hmdb51_1.json ./src/pretrained/rgb_imagenet.ckpt
  sleep 4m
  bash ./scripts/run_preprocess.sh 8 hccl_8p.json ucf101 rgb 40 ./rgb/ucf101/jpg ./rgb/ucf101/annotation/ucf101_01.json ./src/pretrained/rgb_imagenet.ckpt
  sleep 4m
  bash ./scripts/run_preprocess.sh 8 hccl_8p.json hmdb51 flow 60 ./flow/hmdb51/jpg ./flow/hmdb51/annotation/hmdb51_1.json ./src/pretrained/flow_imagenet.ckpt
  sleep 4m
  bash ./scripts/run_preprocess.sh 8 hccl_8p.json ucf101 flow 60 ./flow/ucf101/jpg ./flow/ucf101/annotation/ucf101_01.json ./src/pretrained/flow_imagenet.ckpt
}

rm -rf $DIR/output_preprocess
mkdir $DIR/output_preprocess

preprocess
if [ $? -ne 0 ]; then
  echo "preprocess code failed"
  exit 1
fi
