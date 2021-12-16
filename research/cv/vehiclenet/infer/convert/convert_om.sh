#!/bin/bash

if [ $# -ne 2 ]
then
  echo "Wrong parameter format."
  echo "Usage:"
  echo "         bash $0 [INPUT_AIR_PATH] [OUTPUT_OM_PATH_NAME]"
  echo "Example: "
  echo "         bash convert_om.sh  xxx.air ./aipp.cfg xx"

  exit 1
fi

input_air_path=$1
output_om_path=$2

echo "Input AIR file path: ${input_air_path}"
echo "Output OM file path: ${output_om_path}"

atc --input_format=NHWC \
    --framework=1 \
    --model="${input_air_path}" \
    --input_shape="x:1, 3, 384, 384" \
    --output="${output_om_path}" \
    --soc_version=Ascend310 
