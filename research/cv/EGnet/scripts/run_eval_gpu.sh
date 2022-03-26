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

# Get absolute path
get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

# Get current script path
BASE_PATH=$(cd "`dirname $0`" || exit; pwd)

cd $BASE_PATH/..

echo "evalating ECSSD"
python eval.py --device_target=GPU      \
      --test_fold='./result/ECSSD'   \
      --sal_mode=e >test_e.log

echo "evalating PASCAL-S"
python eval.py --device_target=GPU          \
      --test_fold='./result/PASCAL-S'  \
      --sal_mode=p >test_p.log

echo "evalating DUT-OMRON"
python eval.py --device_target=GPU            \
      --test_fold='./result/DUT-OMRON'  \
      --sal_mode=d >test_d.log

echo "evalating HKU-IS"
python eval.py --device_target=GPU      \
      --test_fold='./result/HKU-IS'  \
      --sal_mode=h >test_h.log

echo "evalating SOD"
python eval.py --device_target=GPU \
      --test_fold='./result/SOD'   \
      --sal_mode=s >test_s.log

echo "evalating DUTS-TE"
python eval.py --device_target=GPU        \
      --test_fold='./result/DUTS-TE'  \
      --sal_mode=t >test_t.log
