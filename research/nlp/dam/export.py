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
"""export checkpoint file into mindir models"""
import os
import numpy as np
from mindspore import Tensor, context, load_checkpoint, load_param_into_net, export

from src.net import DAMNet, PredictWithNet
from src.config import parse_args

context.set_context(mode=context.GRAPH_MODE, save_graphs=False)

if __name__ == '__main__':
    args = parse_args()
    if args.model_name == "DAM_ubuntu":
        args.vocab_size = 434512
        args.channel1_dim = 32
    elif args.model_name == "DAM_douban":
        args.vocab_size = 172130
        args.channel1_dim = 16
    else:
        raise RuntimeError('{} does not exist'.format(args.model_name))

    # net
    network = DAMNet(args)
    network = PredictWithNet(network)
    network.set_train(False)

    # load checkpoint
    ckpt_file = os.path.join(args.ckpt_path, args.ckpt_name)
    param_dict = load_checkpoint(ckpt_file)
    load_param_into_net(network, param_dict)

    turns = Tensor(np.zeros([args.batch_size, args.max_turn_num, args.max_turn_len]).astype(np.int32))
    every_turn_len = Tensor(np.zeros([args.batch_size, args.max_turn_num]).astype(np.int32))
    response = Tensor(np.zeros([args.batch_size, args.max_turn_len]).astype(np.int32))
    response_len = Tensor(np.zeros([args.batch_size, 1]).astype(np.int32))
    labels = Tensor(np.zeros([args.batch_size, 1]).astype(np.int32))

    input_data = [turns, every_turn_len, response, response_len, labels]
    export(network, *input_data, file_name=args.model_name, file_format=args.file_format)
