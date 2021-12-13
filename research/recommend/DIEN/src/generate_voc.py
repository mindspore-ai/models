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

"""generate_voc"""

import _pickle as pk

with open("local_train_splitByUser", "r") as f_train:
    uid_dict = {}
    mid_dict = {}
    cat_dict = {}

    iddd = 0
    for line in f_train:
        arr = line.strip("\n").split("\t")
        clk = arr[0]
        uid = arr[1]
        mid = arr[2]
        cat = arr[3]
        mid_list = arr[4]
        cat_list = arr[5]
        if uid not in uid_dict:
            uid_dict[uid] = 0
        uid_dict[uid] += 1
        if mid not in mid_dict:
            mid_dict[mid] = 0
        mid_dict[mid] += 1
        if cat not in cat_dict:
            cat_dict[cat] = 0
        cat_dict[cat] += 1
        if not mid_list:
            continue
        for m in mid_list.split(""):
            if m not in mid_dict:
                mid_dict[m] = 0
            mid_dict[m] += 1
        iddd += 1
        for c in cat_list.split(""):
            if c not in cat_dict:
                cat_dict[c] = 0
            cat_dict[c] += 1

sorted_uid_dict = sorted(uid_dict.items(), key=lambda x: x[1], reverse=True)
sorted_mid_dict = sorted(mid_dict.items(), key=lambda x: x[1], reverse=True)
sorted_cat_dict = sorted(cat_dict.items(), key=lambda x: x[1], reverse=True)

uid_voc = {}
index = 0
for key, value in sorted_uid_dict:
    uid_voc[key] = index
    index += 1

mid_voc = {}
mid_voc["default_mid"] = 0
index = 1
for key, value in sorted_mid_dict:
    mid_voc[key] = index
    index += 1

cat_voc = {}
cat_voc["default_cat"] = 0
index = 1
for key, value in sorted_cat_dict:
    cat_voc[key] = index
    index += 1

pk.dump(uid_voc, open("uid_voc.pkl", "wb"))
pk.dump(mid_voc, open("mid_voc.pkl", "wb"))
pk.dump(cat_voc, open("cat_voc.pkl", "wb"))
