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
#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash eval_books.sh"
echo "=============================================================================================================="
set -e

python ../eval.py --mindrecord_path=../dataset_mindrecord --dataset_type=Books --dataset_file_path=../Books --device_id=0 --save_checkpoint_path=./scripts/Bookdevice0/ckpt/Books_DIEN2.ckpt > output.log 2>&1 &