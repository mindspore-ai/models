#!/bin/bash
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

"""
predict model

"""
if [ $# -ne 4 ]
then
    echo "=============================================================================================================="
    echo "Please run the script as: "
    echo "bash run_preduct.sh [TASK_NAME] [DATA_PATH] [CKPT_PATH] [PREDICT_PATH]"
    echo "TASK_TYPE including [match, match_kn, match_kn_gene]"
    echo "=============================================================================================================="
exit 1
fi
TASK_NAME=$1
DATA_PATH=$2
CKPT_PATH=$3
PREDICT_PATH=$4
cd ..
rm -rf ${PREDICT_PATH}
mkdir ${PREDICT_PATH}
candidate_file=data/resource/candidate.test.txt
cd ${CKPT_PATH}
for file in *.ckpt
do
    load_checkpoint_file=${CKPT_PATH}/${file}
    score_file=${PREDICT_PATH}/score.${file}.txt
    result_file=${PREDICT_PATH}/result.${file}.txt
    cd ../../
    echo ${load_checkpoint_file}
    echo ${score_file}
    echo ${DATA_PATH}
    python predict.py --task_name=${TASK_NAME} \
                    --max_seq_length=128 \
                    --batch_size=100 \
                    --eval_data_file_path=${DATA_PATH} \
                    --load_checkpoint_path=${load_checkpoint_file} \
                    --save_file_path=${score_file} > ${PREDICT_PATH}/predict_log.txt 2>&1

    python src/utils/extract.py ${candidate_file} ${score_file} ${result_file} > ${PREDICT_PATH}/extract_log.txt 2>&1
    python src/eval.py ${result_file} > ${PREDICT_PATH}/predict.${file}.log 2>&1
    cd src/utils
done
echo "predict finish"
