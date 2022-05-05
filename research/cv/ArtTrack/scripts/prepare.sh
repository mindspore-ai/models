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
# ============================================================================

script_dir=$(cd "$(dirname "$0")" || exit;pwd)
base_dir=$(cd "$script_dir/.." || exit;pwd)

function prepare_env() {
  mkdir -p "$base_dir"/out
  cd "$base_dir"/out || exit
  echo 'clone tensorflow version code'
  if [ ! -d pose-tensorflow ]
  then
      git clone https://github.com/eldar/pose-tensorflow.git
  fi
  cd pose-tensorflow || exit
  git am "$base_dir"/patch/*
  cd lib/multicut_cython || exit
  python setup_pybind11.py install
  cd ../nms_cython || exit
  python setup_pybind11.py install

  cd "$base_dir" || exit
  echo 'convert pretrained resnet101 checkpoint for mindspore'
  python preprocess.py tf2ms --checkpoint out/resnet_v1_101.ckpt --map config/tf2ms.json --output out/pretrained_resnet101.ckpt
}

function prepare_ascend_env() {
  mkdir -p "$base_dir"/out
  cd "$base_dir"/out || exit
  echo 'clone tensorflow version code'
  if [ ! -d pose-tensorflow ]
  then
      git clone https://gitclone.com/github.com/eldar/pose-tensorflow.git
  fi
  cd pose-tensorflow || exit
  git am "$base_dir"/patch/*
  cd lib/multicut_cython || exit
  python setup_pybind11.py install
  cd ../nms_cython || exit
  python setup_pybind11.py install

  cd "$base_dir" || exit
  if [ -f out/resnet101_ascend_v130_imagenet2012_official_cv_bs32_top1acc78.55__top5acc94.34.ckpt ]
  then
      echo 'preproccess pretrained resnet101 checkpoint'
      python ./src/tool/preprocess/preprocess_ckpt.py
  fi
}

function prepare_mpii() {
  cd "$base_dir" || exit
  echo 'preprocess dataset'
  python preprocess.py mpii_single --dataset-dir mpii --dataset-name mpii_human_pose_v1_u12_1
  echo 'split dataset'
  python preprocess.py mat2json --index-mat out/pose-tensorflow/matlab/mpii/test_index.mat --name test_index \
  --dataset-json mpii/cropped/dataset.json \
  --output-dir out

  python preprocess.py mat2json --index-mat out/pose-tensorflow/matlab/mpii/train_index.mat --name train_index \
  --dataset-json mpii/cropped/dataset.json \
  --output-dir out
  echo 'prepare mpii successfully.'
  
  python ./src/tool/preprocess/image2bin.py
  echo 'convert images to .bin successfully, and it will be used in 310 infer.'
}

function prepare_coco() {
  if [ ! -f out/pairwise_stats.mat ]
  then
      python preprocess.py pairwise --config config/coco_pairwise.yaml
  fi

  if [ ! -d out/pairwise ]
  then
      cd "$base_dir"/out || exit
      $DOWNLOADER https://datasets.d2.mpi-inf.mpg.de/deepercut-models-tensorflow/pairwise_coco.tar.gz
      tar xvzf pairwise_coco.tar.gz
  fi

  echo 'prepare coco successfully.'

}


case $1 in
env)
  prepare_env
  ;;
mpii)
  prepare_mpii
  ;;
coco)
  prepare_coco
  ;;
env_ascend)
  prepare_ascend_env
  ;;
*)
  echo "Please run the script as: "
  echo "bash scripts/prepare.sh [TARGET]."
  echo "TARGET: env, mpii, coco, env_ascend"
  echo "For example: bash scripts/prepare.sh env"
  ;;
esac
