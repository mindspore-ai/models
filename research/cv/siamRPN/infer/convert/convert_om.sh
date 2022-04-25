#!/bin/bash

if [ $# -ne 2 ]
then
  echo "Wrong parameter format."
  echo "Usage:"
  echo "         bash $0 [INPUT_AIR_PATH] [OUTPUT_OM_PATH_NAME]"
  echo "Example: "
  echo "         bash convert_om.sh  xxx.air xx"

  exit 1
fi

model_path=$1
output_model_name=$2


atc \
    --model=$model_path \
    --framework=1 \
    --output=$output_model_name \
    --input_format=NCHW --input_shape="actual_input_1:1,3,127,127;actual_input_2:1,3,255,255" \
    --enable_small_channel=0 \
    --log=error \
    --soc_version=Ascend310 \
    --output_type=FP32
