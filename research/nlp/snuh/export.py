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

import pickle
import numpy as np
import mindspore as ms
from mindspore import Tensor, context, load_checkpoint, load_param_into_net, export
from utils.args import get_export_argparser
from utils.data import LabeledDocuments
from models.snuh import SNUH

def main():
    argparser = get_export_argparser()
    args = argparser.parse_args()
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device)

    hparams = pickle.load(open(args.hparams_path, 'rb'))
    data = LabeledDocuments(hparams.data_path, hparams.num_neighbors)
    _, _, _, _ = data.get_loaders(hparams.num_trees, hparams.alpha, hparams.batch_size)

    snuh = SNUH(hparams, data.num_nodes, data.num_edges, data.vocab_size)
    param_dict = load_checkpoint(args.ckpt_path)
    load_param_into_net(snuh, param_dict)

    x = Tensor(np.ones((hparams.batch_size, data.vocab_size)), ms.float32)
    label = Tensor(np.ones((hparams.batch_size, 1)), ms.float32)
    edge1 = Tensor(np.ones((hparams.batch_size, data.vocab_size)), ms.float32)
    edge2 = Tensor(np.ones((hparams.batch_size, data.vocab_size)), ms.float32)
    weight = Tensor(np.ones((hparams.batch_size)), ms.float32)
    export(snuh, x, label, edge1, edge2, weight, file_name=args.file_name, file_format=args.file_format)
    print('=========================================')
    print('{}.mindir exported successfully!'.format(args.file_name))
    print('=========================================')

if __name__ == '__main__':
    main()
