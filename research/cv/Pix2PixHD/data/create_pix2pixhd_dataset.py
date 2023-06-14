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
# ===========================================================================

import os
import shutil

# phase: train or test or val
phase = "train"
dst_root_dir = "./cityscapes/dataset"
phase_img_dir = os.path.join(dst_root_dir, phase + "_img")
phase_inst_dir = os.path.join(dst_root_dir, phase + "_inst")
phase_label_dir = os.path.join(dst_root_dir, phase + "_label")

src_root_dir = "/data/cityscapes/"
left_img_dir = os.path.join(src_root_dir, "leftImg8bit")
gt_fine_dir = os.path.join(src_root_dir, "gtFine")

dir_list = [dst_root_dir, phase_img_dir, phase_inst_dir, phase_label_dir]
for item in dir_list:
    if not os.path.exists(item):
        os.makedirs(item)

# copy leftImg8bit in phase to dst
left_img_phase_dir = os.path.join(left_img_dir, phase)
for dir_name in os.listdir(left_img_phase_dir):
    for entry in os.scandir(os.path.join((left_img_phase_dir, dir_name))):
        if entry.is_file():
            shutil.copyfile(
                os.path.join(left_img_phase_dir, dir_name, entry.name), os.path.join(phase_img_dir, entry.name)
            )

# copy gtFine in phase to dst
gt_fine_phase_dir = os.path.join(gt_fine_dir, phase)
for dir_name in os.listdir(gt_fine_phase_dir):
    for entry in os.scandir(os.path.join(gt_fine_phase_dir, dir_name)):
        if entry.is_file():
            # copy label
            if entry.name.split("_")[4] in "instanceIds.png":
                shutil.copyfile(
                    os.path.join(gt_fine_phase_dir, dir_name, entry.name), os.path.join(phase_label_dir, entry.name)
                )
            # copy inst
            elif entry.name.split("_")[4] in "labelIds.png":
                shutil.copyfile(
                    os.path.join(gt_fine_phase_dir, dir_name, entry.name), os.path.join(phase_inst_dir, entry.name)
                )
            else:
                continue

print("has done.")
