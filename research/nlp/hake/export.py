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
"""export"""

import argparse
from mindspore import context
import mindspore.dataset as ds
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.serialization import export

from src.HAKE_for_train import HAKENetworkWithLoss_TAIL
from src.HAKE_model import HAKE_GRAPH
from src.config import config
from src.dataset import DataReader, TrainDataset, BatchType

parser = argparse.ArgumentParser(description="HAKE export")
parser.add_argument('--file_name', type=str, default='HAKE', help='output file name prefix.')
parser.add_argument('--file_format', type=str, choices=['AIR', 'ONNX', 'MINDIR'], default='MINDIR', \
                    help='file format')
parser.add_argument("--ckpt_path", type=str, default='./wn18rr/CKP-120_339.ckpt')
parser.add_argument('--platform', type=str, default='GPU', help='only support GPU')
parser.add_argument("--device_id", type=int, default=0, help="device id, default: 0.")

if __name__ == '__main__':
    args = parser.parse_args()
    context.set_context(mode=context.GRAPH_MODE, device_target=args.platform, device_id=args.device_id)
    data_reader = DataReader(config.data_path)
    num_entity = len(data_reader.entity_dict)
    num_relation = len(data_reader.relation_dict)

    hake = HAKE_GRAPH(num_entity, num_relation, config.hidden_dim, config.gamma, config.modulus_weight,
                      config.phase_weight)
    params = load_checkpoint(args.ckpt_path)
    load_param_into_net(hake, params)
    hake.set_train(False)

    netwithloss_tail = HAKENetworkWithLoss_TAIL(hake, config.adversarial_temperature)

    dataset_head = ds.GeneratorDataset(
        source=TrainDataset(data_reader, config.negative_sample_size, BatchType.HEAD_BATCH),
        column_names=["pos_triple", "neg_triples", "subsampling_weight"],
    ).batch(batch_size=config.batch_size)

    pos_triple = None
    neg_triples = None
    subsampling_weight = None

    for i in dataset_head.create_dict_iterator():
        pos_triple = i["pos_triple"]
        neg_triples = i["neg_triples"]
        subsampling_weight = i["subsampling_weight"]
        break

    G_file = f"{args.file_name}_model"
    export(netwithloss_tail, pos_triple, neg_triples, subsampling_weight, file_name=G_file,
           file_format=args.file_format)
