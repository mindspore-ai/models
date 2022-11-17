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
"""create checkpoint file list."""
import os

def find_file(path, qianzhui):
    r'''
    Find the file address according to the prefix.
    '''
    result_addr = None
    for i, _, k in os.walk(path):
        for file in k:
            if file.startswith(qianzhui):
                result_addr = os.path.join(i, file)
                break
    return result_addr


def create_ckpt_file_list(args, max_index=None, train_strategy=None, steps_per_epoch=4509):
    """user-defined ckpt file list"""
    ckpt_file_list = []
    # train_strategy
    if train_strategy is not None:
        true_path = find_file(args.ckpt_path_doc, train_strategy)
        if true_path is not None:
            ckpt_file_list.append(true_path)
        else:
            raise Exception("+++ ckpt not found!!! +++")
        return ckpt_file_list

    # order in rank_id
    for i in range(0, args.ckpt_partition):
        path_name = "cpm_rank_" + str(i) + "-"
        if max_index is not None:
            path_name = path_name + str(max_index) + "_"
        true_path = find_file(args.ckpt_path_doc, path_name)
        if true_path is not None:
            ckpt_file_list.append(true_path)
        else:
            path_name = "cpm_rank_" + str(i) + "-" + str(max_index * steps_per_epoch)
            true_path = find_file(args.ckpt_path_doc, path_name)
            if true_path is not None:
                ckpt_file_list.append(true_path)
            else:
                raise Exception("+++ ckpt not found!!! +++")
    return ckpt_file_list
