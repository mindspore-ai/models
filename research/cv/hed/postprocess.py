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
"""hed 310 infer."""
import os

from src.model_utils.config import config


def get_acc():
    '''get acc'''
    config.result_dir = config.post_result_path
    config.save_dir = os.path.join(config.res_output_path, 'result/hed_eval_result')
    config.gt_dir = os.path.join(config.data_path, 'BSDS500/data/groundTruth/test')
    alg = [config.alg]  # algorithms for plotting
    model_name_list = [config.model_name_list]  # model name
    result_dir = os.path.abspath(config.result_dir)  # forward result directory
    save_dir = os.path.abspath(config.save_dir)  # nms result directory
    gt_dir = os.path.abspath(config.gt_dir)  # ground truth directory
    workers = config.workers  # number workers

    nms_process(model_name_list, result_dir, save_dir, file_format=".bin")
    eval_edge(alg, model_name_list, save_dir, gt_dir, workers)

if __name__ == "__main__":
    from src.nms_process import nms_process
    from src.eval_edge import eval_edge
    get_acc()
