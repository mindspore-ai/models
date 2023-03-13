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
"""DeepSpeech2 checkpoint converter."""
from pathlib import Path

import numpy as np
from mindspore import Parameter
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import load_checkpoint
from mindspore import save_checkpoint

from src.cfg.config import config
from src.deepspeech2.model import DeepSpeechModel


def main(ckpt_url):
    spect_config = {
        'sampling_rate': config.ds_sampling_rate,
        'window_size': config.ds_window_size,
        'window_stride': config.ds_window_stride,
        'window': config.ds_window
    }
    # Initialize model to get new lstm params names
    model = DeepSpeechModel(
        batch_size=1,
        rnn_hidden_size=config.ds_hidden_size,
        nb_layers=config.ds_hidden_layers,
        labels=config.labels,
        rnn_type=config.ds_rnn_type,
        audio_conf=spect_config,
        bidirectional=True
    )

    filter_prefix = ['moment1', 'moment2', 'step', 'learning_rate', 'beta1_power', 'beta2_power']
    lstm_old_names = ['RNN.weight0', 'RNN.weight1', 'RNN.weight2', 'RNN.weight3', 'RNN.weight4']
    new_params = model.trainable_params()
    old_params = load_checkpoint(ckpt_url, choice_func=lambda x: not x.startswith(tuple(filter_prefix)))
    names_and_shapes = {param.name: param.shape for param in new_params}

    lstm_weights = {}
    # Reprocess flatten weights of LSTM from < 1.5 mindspore versions to new.
    for layer, old_layer in zip(range(0, 5), lstm_old_names):
        previous = 0
        for i in np.array(list(names_and_shapes.keys())[layer * 8 + 6: layer * 8 + 14])[[0, 2, 1, 3, 4, 6, 5, 7]]:
            weights = old_params[old_layer][int(previous): int(previous + np.prod(names_and_shapes[i]))].asnumpy()
            weights_shaped = weights.reshape(names_and_shapes[i])
            lstm_weights[i] = weights_shaped

            previous += np.prod(names_and_shapes[i])

        # Remove lstm layers to the load remaining layers
        old_params.pop(old_layer)

    # Put remaining weights into dictionary
    for remaining_key, remaining_param in old_params.items():
        lstm_weights[remaining_key] = remaining_param.asnumpy()

    # Process to checkpoint save format
    save_params = []
    for key, value in lstm_weights.items():
        save_params.append({'name': key, 'data': Parameter(Tensor(value, mstype.float32), name=key)})

    save_name = Path(Path(ckpt_url).parent, 'DeepSpeech2.ckpt')
    save_checkpoint(save_params, str(save_name))

    print('Successfully converted checkpoint')
    print(f'New checkpoint path {save_name}')


if __name__ == "__main__":
    main(config.ds_ckpt_url)
