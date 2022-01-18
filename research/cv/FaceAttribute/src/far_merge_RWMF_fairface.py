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
"""far merge fairface."""

import os

if __name__ == '__main__':
    src_txt1 = './fairface_label_train.txt'
    src_txt2 = './RWMF_label_train.txt'
    src_txt3 = './fairface_label_val.txt'
    target_txt = 'train.txt'
    if os.path.exists(target_txt):
        os.remove(target_txt)

    with open(target_txt, 'w') as txt:
        with open(src_txt1, 'r') as t1:
            lines = t1.readlines()
            for line in lines:
                txt.write(line)
        with open(src_txt2, 'r') as t2:
            lines = t2.readlines()
            for line in lines:
                txt.write(line)

    os.system('cp ' + src_txt3 + ' ' + 'eval.txt')
