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
"""WaveGlow checkpoint converter."""
import pickle
from pathlib import Path

import numpy as np
from mindspore import Parameter
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import save_checkpoint

from src.cfg.config import config
from src.waveglow.model import WaveGlow


def main(ckpt_url):
    with Path(ckpt_url).open('rb') as file:
        waveglow_np_params = pickle.load(file)

    wn_config = {
        'n_layers': config.wg_n_layers,
        'n_channels': config.wg_n_channels,
        'kernel_size': config.wg_kernel_size
    }

    # Initialize model to get true names
    model = WaveGlow(
        n_mel_channels=config.wg_n_mel_channels,
        n_flows=config.wg_n_flows,
        n_group=config.wg_n_group,
        n_early_every=config.wg_n_early_every,
        n_early_size=config.wg_n_early_size,
        wn_config=wn_config
    )
    names_and_shapes = {key: param.shape for key, param in model.parameters_and_names()}

    # Put similar names into blocks
    wn_names = list(waveglow_np_params.keys())[2: 2 + 38 * 12]
    convinv_names = list(waveglow_np_params.keys())[-12:]
    ordered_names = list(waveglow_np_params.keys())[:2]

    # Mindspore order of weights into same block
    indexes_weighs = np.concatenate((np.arange(1, 34, 2), np.array([34, 37])))
    indexes_biases = np.concatenate((np.arange(0, 34, 2), np.array([35, 36])))

    for block_num in reversed(range(12)):
        block_layers = wn_names[block_num * 38: 38 * (block_num + 1)]
        for layer_index_weight, layer_index_bias in zip(indexes_weighs, indexes_biases):
            ordered_names.append(block_layers[layer_index_weight])
            ordered_names.append(block_layers[layer_index_bias])
        ordered_names.append(convinv_names[block_num])

    # Reshape weights and process inverted convolutions
    processed_weights = {}
    for torch_name, mindspore_name in zip(ordered_names, list(names_and_shapes.keys())):
        weights = waveglow_np_params[torch_name]
        if torch_name.startswith('convinv'):
            weights = np.linalg.inv((np.squeeze(weights)))
            weights = np.expand_dims(weights, -1)
        processed_weights[mindspore_name] = weights.reshape(names_and_shapes[mindspore_name])

    save_params = []
    for key, value in processed_weights.items():
        save_params.append({'name': key, 'data': Parameter(Tensor(value, mstype.float32), name=key)})

    save_name = Path(Path(ckpt_url).parent, 'WaveGlow.ckpt')
    save_checkpoint(save_params, str(save_name))

    print('Successfully converted checkpoint')
    print(f'New checkpoint path {save_name}')


if __name__ == "__main__":
    main(config.wg_ckpt_url)
