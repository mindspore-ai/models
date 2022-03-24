#!/bin/bash
#copyright 2021 Huawei Technologies Co., Ltd
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
if [ $# != 4 ]
then
    echo "Usage: bash run_310_infer.sh [MINDIR_PATH] [EVAL_DATA_DIR] [DEVICE_ID] [BATCH_SIZE]."
    exit 1
fi

export ASCEND_HOME=/usr/local/Ascend
if [ -d ${ASCEND_HOME}/ascend-toolkit ]; then
    export PATH=$ASCEND_HOME/fwkacllib/bin:$ASCEND_HOME/fwkacllib/ccec_compiler/bin:$ASCEND_HOME/ascend-toolkit/latest/fwkacllib/ccec_compiler/bin:$ASCEND_HOME/ascend-toolkit/latest/atc/bin:$PATH
    export LD_LIBRARY_PATH=$ASCEND_HOME/fwkacllib/lib64:/usr/local/lib:$ASCEND_HOME/ascend-toolkit/latest/atc/lib64:$ASCEND_HOME/ascend-toolkit/latest/fwkacllib/lib64:$ASCEND_HOME/driver/lib64:$ASCEND_HOME/add-ons:$LD_LIBRARY_PATH
    export TBE_IMPL_PATH=$ASCEND_HOME/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe
    export PYTHONPATH=$ASCEND_HOME/fwkacllib/python/site-packages:${TBE_IMPL_PATH}:$ASCEND_HOME/ascend-toolkit/latest/fwkacllib/python/site-packages:$PYTHONPATH
    export ASCEND_OPP_PATH=$ASCEND_HOME/ascend-toolkit/latest/opp
else
    export ASCEND_HOME=/usr/local/Ascend/latest/
    export PATH=$ASCEND_HOME/fwkacllib/bin:$ASCEND_HOME/fwkacllib/ccec_compiler/bin:$ASCEND_HOME/atc/ccec_compiler/bin:$ASCEND_HOME/atc/bin:$PATH
    export LD_LIBRARY_PATH=$ASCEND_HOME/fwkacllib/lib64:/usr/local/lib:$ASCEND_HOME/atc/lib64:$ASCEND_HOME/acllib/lib64:$ASCEND_HOME/driver/lib64:$ASCEND_HOME/add-ons:$LD_LIBRARY_PATH
    export PYTHONPATH=$ASCEND_HOME/fwkacllib/python/site-packages:$ASCEND_HOME/atc/python/site-packages:$PYTHONPATH
    export ASCEND_OPP_PATH=$ASCEND_HOME/opp
fi

if [ -d $2 ];
then
    rm -rf $2
fi

mkdir $2

if [ -d 'label' ];
then
    rm -rf 'label'
fi

mkdir 'label'

echo "get dataset"
python get_310_eval_dataset.py ucf101 RGB ./data/ucf101_rgb_val_split_1.txt $2
echo "done"


echo "compile..."

cd ./ascend310_infer || exit
bash build.sh &> build.log

echo "compile done"


echo "infer..."

if [ -d result_Files ];
then
    rm -rf result_Files
fi

if [ -d time_Result ];
then
    rm -rf time_Result
fi

mkdir result_Files
mkdir time_Result

dir=$2
if [ ${dir:0:1} = "/" ] 
then 
    ./out/main $1 $2 $3
else 
    ./out/main .$1 .$2 $3
fi 

echo "cal acc..."
resultpath='./result_Files'
labelpath='../label/'
python ../postprocess.py --result_path ${resultpath} --label_path ${labelpath} --batch_size $4 > acc.log 2>&1 &
