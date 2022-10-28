# Copyright 2020 Huawei Technologies Co., Ltd
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
"""export checkpoint file into air models"""

import os
import numpy as np
from mindspore.common import dtype as mstype
from mindspore.train.serialization import export
from mindspore import context, Tensor, load_param_into_net, load_checkpoint, Parameter

from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_encoder_decoder import EncoderDecoderConfig
from src.roberta_model import RobertaGenerationConfig
from src.model_infer import EncoderDecoderInferModel

config.dtype = mstype.float32
config.compute_type = mstype.float16

context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
if config.device_target == "Ascend":
    context.set_context(device_id=config.device_id)


def modelarts_pre_process():
    '''modelarts pre process function.'''
    config.file_name = os.path.join(config.output_path, config.file_name)


def load_weights(model_path):
    """
    Load checkpoint as parameter dict, support both npz file and mindspore checkpoint file.
    """

    ms_ckpt = load_checkpoint(model_path)

    weights = {}
    for msname in ms_ckpt:
        infer_name = msname
        weights[infer_name] = ms_ckpt[msname].data.asnumpy()
        if 'decoder' in infer_name:
            new_infer_name = 'beam_decoder.' + infer_name
            weights[new_infer_name] = ms_ckpt[infer_name].data.asnumpy()
    parameter_dict = {}
    for name in weights:
        parameter_dict[name] = Parameter(Tensor(weights[name]), name=name)
    return parameter_dict


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_export():
    """run export."""
    cfg = config
    config.use_cache = True
    encoder_config = RobertaGenerationConfig(config)
    decoder_config = RobertaGenerationConfig(
        config, is_decoder=True, add_cross_attention=True)

    model_config = EncoderDecoderConfig(encoder_config, decoder_config,
                                        beam_width=cfg.beam_width, length_penalty_weight=cfg.length_penalty_weight,
                                        max_decode_length=cfg.max_decode_length, batch_size=cfg.batch_size)
    ende_model = EncoderDecoderInferModel(
        config=model_config, is_training=False, add_pooling_layer=False)

    parameter_dict = load_weights(cfg.model_file)
    # print(parameter_dict)
    load_param_into_net(ende_model, parameter_dict)

    print('Load weights successfully.')

    source_ids = Tensor(np.ones((1, config.seq_length)).astype(np.int32))
    source_mask = Tensor(np.ones((1, config.seq_length)).astype(np.int32))

    export(ende_model, source_ids, source_mask, file_name=config.file_name, file_format='MINDIR')


if __name__ == '__main__':
    run_export()
