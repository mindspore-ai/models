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

export CUDA_VISIBLE_DEVICES="0,1,2,3"

echo 'Trans TensorFlow model to Numpy.'
if [ $# -lt 1 ] ; then
    echo "Usage: bash torch_get_param.sh [DATA_SET]"
exit 1
fi

if [ "$1" == "enwik8" ]; then
  # Data
  DATA_ROOT=./
  DATA_DIR=${DATA_ROOT}/pretrained_xl/tf_enwik8/data
  MODEL_DIR=${DATA_ROOT}/pretrained_xl/tf_enwik8/model

  # Model
  N_LAYER=24
  D_MODEL=1024
  D_EMBED=1024
  N_HEAD=8
  D_HEAD=128
  D_INNER=3072

  # Testing
  TEST_TGT_LEN=128
  TEST_MEM_LEN=3800
  TEST_CLAMP_LEN=1000

  TEST_CKPT_PATH=${MODEL_DIR}/model.ckpt-0
  TEST_BSZ=2
  TEST_NUM_CORE=2

  echo 'Preprocess test set...'
  python data_utils.py \
    --data_dir=${DATA_DIR}/ \
    --dataset=enwik8 \
    --tgt_len=${TEST_TGT_LEN} \
    --per_host_test_bsz=${TEST_BSZ} \
    --num_passes=1 \
    --use_tpu=False

  echo 'Run evaluation on test set...'
  python tf_get_param.py \
      --data_dir=${DATA_DIR}/tfrecords \
      --record_info_dir=${DATA_DIR}/tfrecords/ \
      --corpus_info_path=${DATA_DIR}/corpus-info.json \
      --eval_ckpt_path=${TEST_CKPT_PATH} \
      --model_dir=EXP-enwik8 \
      --n_layer=${N_LAYER} \
      --d_model=${D_MODEL} \
      --d_embed=${D_EMBED} \
      --n_head=${N_HEAD} \
      --d_head=${D_HEAD} \
      --d_inner=${D_INNER} \
      --dropout=0.0 \
      --dropatt=0.0 \
      --tgt_len=${TEST_TGT_LEN} \
      --mem_len=${TEST_MEM_LEN} \
      --clamp_len=${TEST_CLAMP_LEN} \
      --same_length=True \
      --eval_batch_size=${TEST_BSZ} \
      --num_core_per_host=${TEST_NUM_CORE} \
      --do_train=False \
      --do_eval=True \
      --eval_split=test

fi

if [ "$1" == "text8" ]; then
  # Data
  DATA_ROOT=./
  DATA_DIR=${DATA_ROOT}/pretrained_xl/tf_text8/data
  MODEL_DIR=${DATA_ROOT}/pretrained_xl/tf_text8/model

  # Model
  N_LAYER=24
  D_MODEL=1024
  D_EMBED=1024
  N_HEAD=8
  D_HEAD=128
  D_INNER=3072

  # Testing
  TEST_TGT_LEN=128
  TEST_MEM_LEN=3800
  TEST_CLAMP_LEN=1000

  TEST_CKPT_PATH=${MODEL_DIR}/model.ckpt-0
  TEST_BSZ=2
  TEST_NUM_CORE=2

  echo 'Preprocess test set...'
  python data_utils.py \
    --data_dir=${DATA_DIR}/ \
    --dataset=text8 \
    --tgt_len=${TEST_TGT_LEN} \
    --per_host_test_bsz=${TEST_BSZ} \
    --num_passes=1 \
    --use_tpu=False

  echo 'Run evaluation on test set...'
  python tf_get_param.py \
      --data_dir=${DATA_DIR}/tfrecords \
      --record_info_dir=${DATA_DIR}/tfrecords/ \
      --corpus_info_path=${DATA_DIR}/corpus-info.json \
      --eval_ckpt_path=${TEST_CKPT_PATH} \
      --model_dir=EXP-text8 \
      --n_layer=${N_LAYER} \
      --d_model=${D_MODEL} \
      --d_embed=${D_EMBED} \
      --n_head=${N_HEAD} \
      --d_head=${D_HEAD} \
      --d_inner=${D_INNER} \
      --dropout=0.0 \
      --dropatt=0.0 \
      --tgt_len=${TEST_TGT_LEN} \
      --mem_len=${TEST_MEM_LEN} \
      --clamp_len=${TEST_CLAMP_LEN} \
      --same_length=True \
      --eval_batch_size=${TEST_BSZ} \
      --num_core_per_host=${TEST_NUM_CORE} \
      --do_train=False \
      --do_eval=True \
      --eval_split=test
fi