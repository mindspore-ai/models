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
build dataset
"""

if [ $# -ne 1 ]
then
    echo "=============================================================================================================="
    echo "Please run the script as: "
    echo "sh build_dataset.sh [TASK_NAME]"
    echo "for example: sh scripts/build_dataset.sh match_kn_gene"
    echo "TASK_TYPE including [match, match_kn, match_kn_gene]"
    echo "=============================================================================================================="
exit 1
fi
TASK_NAME=$1

case $TASK_NAME in
    "match")
        DICT_NAME="data/char.dict"
        USE_KNOWLEDGE=0
        TOPIC_GENERALIZATION=0
        ;;
    "match_kn")
        DICT_NAME="data/char.dict"
        USE_KNOWLEDGE=1
        TOPIC_GENERALIZATION=0
        ;;
    "match_kn_gene")
        DICT_NAME="data/gene.dict"
        USE_KNOWLEDGE=1
        TOPIC_GENERALIZATION=1
        ;;
    esac

FOR_PREDICT=0
CANDIDATE_NUM=9
INPUT_PATH="data"
DATA_TYPE=("train" "dev" "test")

# candidate set
candidate_set_file=${INPUT_PATH}/candidate_set.txt

# data preprocessing

for ((i=0; i<${#DATA_TYPE[*]}; i++))
do
    corpus_file=${INPUT_PATH}/${DATA_TYPE[$i]}.txt
    sample_file=${INPUT_PATH}/sample.${DATA_TYPE[$i]}.txt
    candidate_file=${INPUT_PATH}/candidate.${DATA_TYPE[$i]}.txt
    text_file=${INPUT_PATH}/build.${DATA_TYPE[$i]}.txt

    # step 1: build candidate set from session data for negative training cases and predicting candidates
    if [ "${DATA_TYPE[$i]}"x = "train"x ]; then
        python src/utils/build_candidate_set_from_corpus.py ${corpus_file} ${candidate_set_file}
    fi

    # step 2: firstly have to convert session data to sample data
    if [ "${DATA_TYPE[$i]}"x = "test"x ]; then
        sample_file=${corpus_file}
        FOR_PREDICT=1
        CANDIDATE_NUM=10
    else
        python src/utils/convert_session_to_sample.py ${corpus_file} ${sample_file}
    fi
    # step 3: construct candidate for sample data
    python src/utils/construct_candidate.py ${sample_file} ${candidate_set_file} ${candidate_file} ${CANDIDATE_NUM}

    # step 4: convert sample data with candidates to text data required by the model
    python src/utils/convert_conversation_corpus_to_model_text.py ${candidate_file} ${text_file} ${USE_KNOWLEDGE} ${TOPIC_GENERALIZATION} ${FOR_PREDICT}

    # step 5: build dict from the training data, here we build character dict for model
    if [ "${DATA_TYPE[$i]}"x = "train"x ]; then
        python src/utils/build_dict.py ${text_file} ${DICT_NAME}
    fi
done
