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
"""export checkpoint file into air, mindir models"""
import ast
import argparse
import numpy as np
import mindspore
from mindspore import context, Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.dataset.data_manager import DatasetManager
from src.model.DenseNet import DenseNet121

parser = argparse.ArgumentParser(description='export MultiTask network')

parser.add_argument('--device_id', type=int, default=0)
parser.add_argument('--ckpt_path', type=str, default='./ckpt/')
parser.add_argument('--dataset', type=str, default='veri', help="name of the dataset")
parser.add_argument('--root', type=str, default='./data', help="root path to data directory")
parser.add_argument('--file_format', type=str, choices=["AIR", "ONNX", "MINDIR"], default="MINDIR")
parser.add_argument("--device_target", type=str, choices=["Ascend", "GPU", "CPU"], default="Ascend")
parser.add_argument('--segmentaware', type=ast.literal_eval, default=False, help="embed segments to images")
parser.add_argument('--heatmapaware', type=ast.literal_eval, default=False, help="embed heatmaps to images")

args = parser.parse_args()

if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target,
                        save_graphs=False, device_id=args.device_id)

    data = DatasetManager(dataset_dir=args.dataset, root=args.root)

    model = DenseNet121(pretrain_path='',
                        num_vids=data.num_train_vids,
                        num_vcolors=data.num_train_vcolors,
                        num_vtypes=data.num_train_vtypes,
                        keyptaware=True,
                        heatmapaware=args.heatmapaware,
                        segmentaware=args.segmentaware,
                        multitask=True,
                        is_pretrained=False)

    param_dict = load_checkpoint(args.ckpt_path)
    load_param_into_net(model, param_dict)

    if args.heatmapaware:
        input_img = Tensor(np.zeros((1, 39, 256, 256)), mindspore.float32)
        input_vkeypt = Tensor(np.zeros((1, 108)), mindspore.float32)
    if args.segmentaware:
        input_img = Tensor(np.zeros((1, 16, 256, 256)), mindspore.float32)
        input_vkeypt = Tensor(np.zeros((1, 108)), mindspore.float32)

    inputs = (input_img, input_vkeypt)
    if args.heatmapaware:
        mindspore.export(model, *inputs, file_name="MultiTask_export_heatmap", file_format=args.file_format)
    if args.segmentaware:
        mindspore.export(model, *inputs, file_name="MultiTask_export_segment", file_format=args.file_format)
