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

accuracy = {}
acc_nums = {}
with open('modelarts-job-c0a748c1-47a8-45d0-9603-2097883482c0-worker-0.log') as f:
    lines = f.readlines()
    for line in lines:
        if 'Validation-Loss' in line:
            contents = line.strip().split(' ')
            ckpt_index = int(contents[1].strip(','))
            if str(ckpt_index) not in accuracy.keys():
                acc_nums[str(ckpt_index)] = 1
                accuracy[str(ckpt_index)] = float(contents[8].strip(','))
            else:
                acc_nums[str(ckpt_index)] += 1
                accuracy[str(ckpt_index)] += float(contents[8].strip(','))

print(accuracy)
print(acc_nums)

mean_acc = []
acc_go = acc_nums.keys()
acc_lo = accuracy.keys()
for key in acc_lo:
    if key not in acc_go:
        print('Wrong key!!!!!!!')
    else:
        mean_acc.append(accuracy[key]/acc_nums[key])

print(mean_acc)
print(max(mean_acc))
