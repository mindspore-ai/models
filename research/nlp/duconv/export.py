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

"""export checkpoint file into models"""
import numpy as np

import mindspore.common.dtype as mstype
from mindspore import Tensor, context, load_checkpoint, export
from src.model_utils.config import config
from src.model import RetrievalWithSoftmax
from src.bert import BertConfig

context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
if config.device_target == "Ascend":
    context.set_context(device_id=config.device_id)

if __name__ == "__main__":
    use_kn = bool("kn" in config.task_name)
    bertconfig = BertConfig(seq_length=config.max_seq_length, vocab_size=config.vocab_size)
    net = RetrievalWithSoftmax(bertconfig, use_kn)
    load_checkpoint(config.checkpoint_file, net=net)
    net.set_train(False)

    context_id = Tensor(np.zeros([config.batch_size, config.max_seq_length]), mstype.int32)
    context_segment_id = Tensor(np.zeros([config.batch_size, config.max_seq_length]), mstype.int32)
    context_pos_id = Tensor(np.zeros([config.batch_size, config.max_seq_length]), mstype.int32)
    kn_id = Tensor(np.zeros([config.batch_size, config.max_seq_length]), mstype.int32)
    kn_seq_length = Tensor(np.zeros([config.batch_size, 1]), mstype.int32)
    input_data = [context_id, context_segment_id, context_pos_id, kn_id, kn_seq_length]
    export(net.network, *input_data, file_name=config.file_name, file_format=config.file_format)
    print("{} success export".format(config.file_name))
