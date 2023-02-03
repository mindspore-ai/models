# Copyright 2023 Huawei Technologies Co., Ltd
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
import os

from longling import path_append, abs_current_dir
from mindspore import load_param_into_net, load_checkpoint

from KTScripts.options import get_exp_configure
from KTScripts.utils import load_model


def load_d_agent(model_name, dataset_name, skill_num, with_label=True):
    model_parameters = get_exp_configure(model_name)
    model_parameters.update({'feat_nums': skill_num, 'model': model_name, 'without_label': not with_label})
    if model_name == 'GRU4Rec':
        model_parameters.update({'output_size': skill_num})
    model = load_model(model_parameters)
    model_folder = path_append(abs_current_dir(__file__), os.path.join('meta_data'))
    model_path = os.path.join(model_folder, f'{model_name}_{dataset_name}')
    if not with_label:
        model_path += '_without'
    load_param_into_net(model, load_checkpoint(f'{model_path}.ckpt'))
    model.set_train(False)
    return model


def episode_reward(initial_score, final_score, full_score) -> (int, float):
    delta = final_score - initial_score
    normalize_factor = full_score - initial_score + 1e-9
    return delta / normalize_factor
