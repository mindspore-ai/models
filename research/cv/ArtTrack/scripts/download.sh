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
base_dir=$(cd "$script_dir"/.. || exit;pwd)
if command -v curl > /dev/null; then
    DOWNLOADER="curl -L -O"
else
    DOWNLOADER="wget"
fi

function download_dataset_mpii() {
  mkdir -p "$base_dir"/mpii
  cd "$base_dir"/mpii || exit
  echo 'downloading mpii dataset'
  if [  ! -f mpii_human_pose_v1_u12_1.mat ] && [ ! -f mpii_human_pose_v1_u12_2.zip  ]
  then
      $DOWNLOADER https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1_u12_2.zip
  fi
  if [  ! -d images ] && [ ! -f mpii_human_pose_v1.tar.gz  ]
  then
      $DOWNLOADER https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz
  fi
  echo 'extract dataset'
  if [  ! -f mpii_human_pose_v1_u12_1.mat  ]
  then
      unzip mpii_human_pose_v1_u12_2.zip
      ln -s mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat mpii_human_pose_v1_u12_1.mat
  fi
  if [  ! -d images  ]
  then
      tar xf mpii_human_pose_v1.tar.gz
  fi
}

function download_pretrained_resnet101() {
  mkdir -p $base_dir/out
  cd $base_dir/out || exit
  $DOWNLOADER http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz
  tar xf resnet_v1_101_2016_08_28.tar.gz
  $DOWNLOADER https://download.mindspore.cn/model_zoo/r1.3/resnet101_ascend_v130_imagenet2012_official_cv_bs32_top1acc78.55__top5acc94.34/resnet101_ascend_v130_imagenet2012_official_cv_bs32_top1acc78.55__top5acc94.34.ckpt
}

function download_dataset_coco() {
  mkdir -p "$base_dir"/coco
  cd "$base_dir"/coco || exit
  COCO_SITE="http://images.cocodataset.org"
  echo 'downloading coco dataset'
  if [  ! -d images/train2014 ] && [ ! -f train2014.zip  ]
  then
      $DOWNLOADER "$COCO_SITE/zips/train2014.zip"
  fi
  if [  ! -d images/val2014 ] && [ ! -f val2014.zip  ]
  then
      $DOWNLOADER "$COCO_SITE/zips/val2014.zip"
  fi
  if [  ! -d annotations/annotations_trainval2014 ] && [ ! -f annotations_trainval2014.zip  ]
  then
      $DOWNLOADER "$COCO_SITE/annotations/annotations_trainval2014.zip"
  fi

  echo 'extract dataset'
  if [  ! -d annotations  ]
  then
      unzip annotations_trainval2014.zip
  fi
  mkdir -p images
  if [  ! -d images/train2014  ]
  then
      cd "$base_dir"/coco/images || exit
      unzip ../train2014.zip
  fi
  if [  ! -d images/val2014  ]
  then
      cd "$base_dir"/coco/images || exit
      unzip ../val2014.zip
  fi

}

case $1 in
dataset_mpii)
  download_dataset_mpii
  ;;
dataset_coco)
  download_dataset_coco
  ;;
pretrained_resnet101)
  download_pretrained_resnet101
  ;;
*)
  echo "Please run the script as: "
  echo "bash scripts/download.sh TARGET."
  echo "TARGET: dataset_mpii, dataset_coco, pretrained_resnet101"
  echo "For example: bash scripts/download.sh dataset_mpii"
esac
