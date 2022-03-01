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

if [[ $# -lt 3 ]]; then
    echo "Usage: bash run_infer_310_om.sh [JSON_PATH] [CKPT_PATH] [VAL_DATA_ROOT] [BATCH_SIZE] [MODE]
    BATCH_SIZE is optional, it can be set by environment variable device_id, otherwise the value is zero"
exit 1
fi

source ${ASCEND_PATH}/ascend-toolkit/set_env.sh

# atc
export CPU_ARCH=x86_64
export THIRDPART_PATH=${ASCEND_PATH}/thirdpart/${CPU_ARCH}
export INSTALL_DIR=${ASCEND_PATH}/ascend-toolkit/latest
export PATH=${ASCEND_PATH}/ascend-toolkit/latest/atc/bin:${PATH}

# python
export PYTHON_INTERPRETER=$(which python3.7)
export PYTHON_BASE_PATH=$(readlink -f "${PYTHON_INTERPRETER}") # path to the symlink target
export PYTHON_BASE_PATH=$(dirname "${PYTHON_BASE_PATH}") # directory which ends with 'bin'
export PYTHON_BASE_PATH=$(dirname "${PYTHON_BASE_PATH}") # base path
export PATH=${PYTHON_BASE_PATH}/bin:$PATH
export LD_LIBRARY_PATH=${PYTHON_BASE_PATH}/lib:$LD_LIBRARY_PATH

# msprof
export PATH=${ASCEND_PATH}/ascend-toolkit/latest/tools/profiler/bin/:${PATH}
MSPROF_PY="${PYTHON_INTERPRETER} ${ASCEND_PATH}/ascend-toolkit/latest/toolkit/tools/profiler/profiler_tool/analysis/msprof/msprof.py"

#build
export DDK_PATH=${ASCEND_PATH}/ascend-toolkit/latest/x86_64-linux
export NPU_HOST_LIB=${ASCEND_PATH}/ascend-toolkit/latest/x86_64-linux/runtime/lib64/stub

INFER_EXEC=$(pwd)/../ascend_run_tool/out/main
ACL_CONFIG=$(realpath ./acl.json)
JSON_PATH=$(realpath $1)
CKPT_PATH=$(realpath $2)
VAL_DATA_ROOT=$(realpath $3)
BATCH_SIZE=1
if [ $# -ge 4 ]; then
    BATCH_SIZE=$4
fi

MODE="inference"
if [ $# -ge 5 ]; then
    MODE=$5
fi


NET_NAME=$(basename $CKPT_PATH .ckpt)

AUTOTUNE=0
EXP_BASE_PATH=$(realpath ./${NET_NAME}-bs${BATCH_SIZE})
if [ $AUTOTUNE -eq 1 ]; then
    EXP_BASE_PATH=${EXP_BASE_PATH}-autotune
fi

if [ -d ${EXP_BASE_PATH} ]; then
    echo "Experiment path $EXP_BASE_PATH exists!"
    exit 1
fi
mkdir -p "$EXP_BASE_PATH"

AIR_ROOT=$(realpath ${EXP_BASE_PATH}/air/)
OM_ROOT=$(realpath ${EXP_BASE_PATH}/om/)
INFER_RESULT_PATH=$(realpath ${EXP_BASE_PATH}/infer_result/)
mkdir -p "$AIR_ROOT"
mkdir -p "$OM_ROOT"
mkdir -p "$INFER_RESULT_PATH"

AIR_FILE=${AIR_ROOT}/${NET_NAME}.air
OM_FILE=${OM_ROOT}/${NET_NAME}.om

echo "ASCEND_PATH: ${ASCEND_PATH}"
echo "PYTHON_INTERPRETER: ${PYTHON_INTERPRETER}"
echo "PYTHON_BASE_PATH: ${PYTHON_BASE_PATH}"
echo "JSON_PATH: ${JSON_PATH}"
echo "CKPT_PATH: ${CKPT_PATH}"
echo "VAL_DATA_ROOT: ${VAL_DATA_ROOT}"
echo "DEVICE_ID: ${DEVICE_ID}"
echo "NET_NAME: ${NET_NAME}"
echo "EXP_BASE_PATH: ${EXP_BASE_PATH}"
echo "AIR_ROOT: ${AIR_ROOT}"
echo "OM_ROOT: ${OM_ROOT}"
echo "AIR_FILE: ${AIR_FILE}"
echo "OM_FILE: ${OM_FILE}"
echo "INFER_EXEC: ${INFER_EXEC}"
echo "ACL_CONFIG: ${ACL_CONFIG}"
echo "INFER_RESULT_PATH: ${INFER_RESULT_PATH}"
echo "BATCH_SIZE: ${BATCH_SIZE}"
echo "MODE: ${MODE}"

function ckpt2air()
{
    cd ..
    CMD="python3.7 ./utils/export.py --jsonFile ${JSON_PATH} --file_name ${AIR_FILE} --file_format AIR --checkpoint_file_path ${CKPT_PATH} --batch_size=${BATCH_SIZE}"
    echo ${CMD}
    ${CMD} 2>&1 | tee ${EXP_BASE_PATH}/ckpt2air.log
    cd -
}

function air2om()
{
    CMD="atc --model=${AIR_FILE} --framework=1 --output=${OM_ROOT}/${NET_NAME} --soc_version=Ascend310 --input_shape=data:$BATCH_SIZE,3,224,224 --input_format=NCHW --aicore_num=2"
    if [ $AUTOTUNE -eq 1 ]; then
        CMD="${CMD} --auto_tune_mode=RL,GA"
    fi
    echo ${CMD}
    ${CMD} 2>&1 | tee ${EXP_BASE_PATH}/air2om.log
}

function build_ascend_run_tool()
{
    rm -rf ../ascend_run_tool/build && mkdir -p ../ascend_run_tool/build/intermediates && cd ../ascend_run_tool/build/intermediates
    cmake -DCMAKE_SKIP_RPATH=TRUE -Dtarget=SOC -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=g++ -G "Unix Makefiles" ../../  2>&1 | tee ${EXP_BASE_PATH}/build.log
    cmake --build . --clean-first  2>&1 | tee -a ${EXP_BASE_PATH}/build.log
    cd -
}

function inference()
{
    rm -rf "${INFER_RESULT_PATH}"
    mkdir -p "${INFER_RESULT_PATH}"
    if [ ! -f ${ACL_CONFIG} ]; then
        echo "{}\n" > ${ACL_CONFIG}
    fi
    CMD="${INFER_EXEC} ${ACL_CONFIG} ${VAL_DATA_ROOT}/img_data/ ${OM_FILE} ${INFER_RESULT_PATH}"
    echo ${CMD}
    ${CMD} 2>&1 | tee ${EXP_BASE_PATH}/inference.log
}

function profile()
{
    rm -rf "${INFER_RESULT_PATH}"
    mkdir -p "${INFER_RESULT_PATH}"
    if [ ! -f ${ACL_CONFIG} ]; then
        echo "{}\n" > ${ACL_CONFIG}
    fi
    RUN_CMD="${INFER_EXEC} ${ACL_CONFIG} ${VAL_DATA_ROOT}/img_data/ ${OM_FILE} ${INFER_RESULT_PATH}"
    echo msprof --application="${RUN_CMD}" --output=${EXP_BASE_PATH}/profile --aic-mode=task-based --aic-metrics=PipeUtilization --ascendcl=on --task-time=on --ai-core=on
    msprof --application="${RUN_CMD}" --output=${EXP_BASE_PATH}/profile --aic-mode=task-based --aic-metrics=PipeUtilization --ascendcl=on --task-time=on --ai-core=on 2>&1 | tee ${EXP_BASE_PATH}/profile.log

    if [ $? -ne 0 ]; then
        echo "msprof failed"
        return 1
    fi
    TMPSTR=$(grep "Profiling data of device" ${EXP_BASE_PATH}/profile.log)

    ARR=("$TMPSTR")
    PROFILE_PATH=${ARR[-1]}

    CMD="${MSPROF_PY} export summary -dir ${PROFILE_PATH} --format=csv"
    echo ${CMD}
    ${CMD} 2>&1 | tee ${EXP_BASE_PATH}/profile_analyze.log
}

function calc_acc()
{
    CMD="python3.7 ../utils/count_acc.py --gt_path ${VAL_DATA_ROOT}/label/ --predict_path ${INFER_RESULT_PATH}"
    echo ${CMD}
    ${CMD} 2>&1 | tee ${EXP_BASE_PATH}/calc_acc.log
}

ckpt2air
if [ $? -ne 0 ]; then
    echo "Conversion CKPT->AIR failed"
    exit 1
fi

air2om
if [ $? -ne 0 ]; then
    echo "Conversion CKPT->AIR failed"
    exit 1
fi

build_ascend_run_tool
if [ $? -ne 0 ]; then
    echo "Run tool build failed"
    exit 1
fi

if [[ ${MODE} == "profile" ]]; then
    profile
    if [ $? -ne 0 ]; then
        echo "OM profile failed"
        exit 1
    fi
else
    inference
    if [ $? -ne 0 ]; then
        echo "OM profile failed"
        exit 1
    fi

    if [ ${BATCH_SIZE} -eq 1 ]; then
        calc_acc
        if [ $? -ne 0 ]; then
            echo "Computing accuracy failed"
            exit 1
        fi
    fi
fi
