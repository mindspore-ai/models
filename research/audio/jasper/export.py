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

"""
export checkpoint file to mindir model
"""
import json
import argparse
import numpy as np
from mindspore import context, Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net, export
from src.config import train_config, encoder_kw, decoder_kw
from src.model_test import Jasper, PredictWithSoftmax

parser = argparse.ArgumentParser(
    description='Export DeepSpeech model to Mindir')
parser.add_argument('--pre_trained_model_path', type=str,
                    default='', help=' existed checkpoint path')
parser.add_argument('--device_target', type=str, default="GPU", choices=("GPU", "CPU", "Ascend"),
                    help='Device target, support GPU and CPU, Default: GPU')
args = parser.parse_args()

if __name__ == '__main__':
    config = train_config
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=args.device_target, save_graphs=False)
    with open(config.DataConfig.labels_path) as label_file:
        labels = json.load(label_file)

    jasper_net = PredictWithSoftmax(
        Jasper(encoder_kw=encoder_kw, decoder_kw=decoder_kw))

    param_dict = load_checkpoint(args.pre_trained_model_path)
    load_param_into_net(jasper_net, param_dict)
    print('Successfully loading the pre-trained model')
    # 3500 is the max length in evaluation dataset(LibriSpeech). This is consistent with that in dataset.py
    # The length is fixed to this value because Mindspore does not support dynamic shape currently
    input_np = np.random.uniform(
        0.0, 1.0, size=[1, 64, 3500]).astype(np.float32)
    length = np.array([100], dtype=np.int32)
    export(jasper_net, Tensor(input_np), Tensor(length),
           file_name="jasper.mindir", file_format='MINDIR')
