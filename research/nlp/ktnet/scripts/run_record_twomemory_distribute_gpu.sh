#!/usr/bin/env bash
# Copyright 2021 Huawei Technologies Co., Ltd
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

if [ $# != 2 ]; then
  echo "Usage: sh run_record_twomemory_distribute_gpu.sh [DATA_PATH] [DEVICE_NUM]"
  exit 1
fi

PWD_DIR=$(pwd)
DATA=$1
DEVICE_NUM=$2
export DEVICE_NUM=$DEVICE_NUM

BERT_DIR=$DATA/cased_L-24_H-1024_A-16
WN_CPT_EMBEDDING_PATH=$DATA/KB_embeddings/wn_concept2vec.txt
NELL_CPT_EMBEDDING_PATH=$DATA/KB_embeddings/nell_concept2vec.txt

echo "start training for $DEVICE_NUM GPU devices"

rm -rf ./train_parallel
mkdir ./train_parallel
cp ../*.py ./train_parallel
cp -r ../src ./train_parallel
cp -r ../utils ./train_parallel
cd ./train_parallel || exit

mpirun -n $DEVICE_NUM --allow-run-as-root --output-filename log_output --merge-stderr-to-stdout \
python3 run_KTNET_record.py \
      --device_target "GPU" \
      --device_num $DEVICE_NUM \
      --batch_size 12 \
      --do_train True \
      --do_predict False \
      --do_lower_case False \
      --init_pretraining_params $BERT_DIR/params \
      --load_pretrain_checkpoint_path $BERT_DIR/roberta.ckpt \
      --train_file $DATA/ReCoRD/train.json \
      --predict_file $DATA/ReCoRD/dev.json \
      --train_mindrecord_file $DATA/ReCoRD/train.mindrecord \
      --predict_mindrecord_file $DATA/ReCoRD/dev.mindrecord \
      --vocab_path $BERT_DIR/vocab.txt \
      --bert_config_path $BERT_DIR/bert_config.json \
      --freeze False \
      --save_steps 4000 \
      --weight_decay 0.01 \
      --warmup_proportion 0.1 \
      --learning_rate 6e-5 \
      --epoch 4 \
      --max_seq_len 384 \
      --doc_stride 128 \
      --wn_concept_embedding_path $WN_CPT_EMBEDDING_PATH \
      --nell_concept_embedding_path $NELL_CPT_EMBEDDING_PATH \
      --use_wordnet True \
      --use_nell True \
      --random_seed 45 \
      --save_finetune_checkpoint_path $PWD_DIR/output/finetune_checkpoint/record/ \
      --is_distribute True \
      --is_modelarts False \
      --save_url /cache/ \
      --log_url /tmp/log/ \
      --checkpoints output/ &> train_record.log &
cd ..