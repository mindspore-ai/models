#!/bin/bash
if [ $# != 2 ]
then
  echo "Wrong parameter format."
  echo "Usage:"
  echo "        bash $0 [INPUT_AIR_PATH] [OUTPUT_OM_PATH_NAME]"
  echo "Example"
  echo "        bash convert_om.sh xxx.air xx"

  exit 1
fi

input_air_path=$1
output_om_path=$2

export install_path=/usr/local/Ascend

export ASCEND_ATC_PATH=${install_path}/atc
export LD_LIBRARY_PATH=${install_path}/atc/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg:$PYTHONPATH

export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_SLOG_PRINT_TO_STDOUT=1

echo "Input AIR file path: ${input_air_path}"
echo "Output OM file path: ${output_om_path}"


atc --framework=1 \
    --input_format=NCHW \
    --model="${input_air_path}" \
    --output="${output_om_path}" \
    --insert_op_conf=./aipp.config \
    --soc_version=Ascend310 \
    --output_type=FP32 \
    --op_select_implmode=high_precision
