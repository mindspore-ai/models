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
"""MultiTaskNet eval"""
import ast
import argparse

from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.utils.evaluate import test
from src.model.DenseNet import DenseNet121
from src.dataset.dataset import eval_create_dataset

parser = argparse.ArgumentParser(description='eval MultiTaskNet')

parser.add_argument('--device_target', type=str, default="Ascend")
parser.add_argument('--ckpt_path', type=str, default='')
parser.add_argument('--device_id', type=int, default=0)

parser.add_argument('--root', type=str, default='./data', help="root path to data directory")
parser.add_argument('-d', '--dataset', type=str, default='veri', help="name of the dataset")
parser.add_argument('--height', type=int, default=256, help="height of an image (default: 256)")
parser.add_argument('--width', type=int, default=256, help="width of an image (default: 256)")
parser.add_argument('--test-batch', default=100, type=int, help="test batch size")
parser.add_argument('--heatmapaware', type=ast.literal_eval, default=False, help="embed heatmaps to images")
parser.add_argument('--segmentaware', type=ast.literal_eval, default=False, help="embed segments to images")

args = parser.parse_args()

if __name__ == '__main__':
    target = args.device_target

    device_id = args.device_id
    context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False, device_id=device_id)

    train_dataset_path = args.root

    query_dataloader, gallery_dataloader, num_train_vids, \
        num_train_vcolors, num_train_vtypes, _vcolor2label, \
            _vtype2label = eval_create_dataset(dataset_dir=args.dataset,
                                               root=train_dataset_path,
                                               width=args.width,
                                               height=args.height,
                                               keyptaware=True,
                                               heatmapaware=args.heatmapaware,
                                               segmentaware=args.segmentaware,
                                               train_batch=args.test_batch)

    _model = DenseNet121(pretrain_path='',
                         num_vids=num_train_vids,
                         num_vcolors=num_train_vcolors,
                         num_vtypes=num_train_vtypes,
                         keyptaware=True,
                         heatmapaware=args.heatmapaware,
                         segmentaware=args.segmentaware,
                         multitask=True,
                         is_pretrained=False)


    ckpt_path = args.ckpt_path

    print("ckpt_path is {}".format(ckpt_path))
    param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(_model, param_dict)

    _distmat = test(_model, True, True, query_dataloader, gallery_dataloader,
                    _vcolor2label, _vtype2label, return_distmat=True)
