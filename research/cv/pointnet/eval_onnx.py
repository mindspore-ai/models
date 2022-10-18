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
"""Run evaluation for a model exported to ONNX"""
from __future__ import print_function
import argparse
import numpy as np
import onnxruntime as ort
from mindspore import Tensor
import mindspore.dataset as ds
import mindspore.ops as ops
from src.dataset import ShapeNetDataset
from tqdm import tqdm

parser = argparse.ArgumentParser(description='MindSpore Pointnet Segmentation')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--nepoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--device_id', type=int, default=0, help='device id')
parser.add_argument('--device_target', default='GPU', help='device id')
parser.add_argument('--data_path', type=str, default='./dataset/shapenetcore_partanno_segmentation_benchmark_v0',
                    help="dataset path")
parser.add_argument('--onnx_path', type=str, default='./mindir/pointnet.onnx', help="onnx path")
parser.add_argument('--class_choice', type=str, default='Chair', help="class_choice")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
parser.add_argument('--enable_modelarts', default=False, help="use feature transform")

args = parser.parse_args()

def run_eval():
    if args.device_target == 'GPU':
        providers = ['CUDAExecutionProvider']
    elif args.device_target in ('CPU', 'Ascend'):
        providers = ['CPUExecutionProvider']
    else:
        raise ValueError(f"Unsupported device_target '{device_target}'. Expected one of: 'CPU', 'GPU', 'Ascend'")
    session = ort.InferenceSession(args.onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name
    local_data_url = args.data_path
    dataset_generator = ShapeNetDataset(
        root=local_data_url,
        classification=False,
        class_choice=[args.class_choice])
    test_dataset_generator = ShapeNetDataset(
        root=local_data_url,
        classification=False,
        class_choice=[args.class_choice],
        split='test',
        data_augmentation=False)

    test_dataloader = ds.GeneratorDataset(test_dataset_generator, ["point", "label"], shuffle=True)
    test_dataset = test_dataloader.batch(args.batchSize)
    num_classes = dataset_generator.num_seg_classes

    print('batchSize', test_dataset.get_batch_size())
    print('shapes2', test_dataset.output_shapes())
    print('test_dataset_size', test_dataset.get_dataset_size())

    shape_ious = []
    for _, data in tqdm(enumerate(test_dataset.create_dict_iterator(), 0)):
        points, target = data['point'], data['label']
        points = Tensor(points).asnumpy()
        pred = session.run(None, {input_name: points})
        pred_choice = ops.ArgMaxWithValue(axis=3)(Tensor(pred))[0]
        pred_np = pred_choice.asnumpy()
        target_np = target.asnumpy() - 1

        for shape_idx in range(target_np.shape[0]):
            parts = range(num_classes)
            part_ious = []
            for part in parts:
                I = np.sum(np.logical_and((pred_np[shape_idx] == part), target_np[shape_idx] == part).reshape(
                    (target_np[shape_idx] == part).shape[0], 1))
                U = np.sum(np.logical_or((pred_np[shape_idx] == part), target_np[shape_idx] == part).reshape(
                    (target_np[shape_idx] == part).shape[0], 1))
                if U == 0:
                    iou = 1
                else:
                    iou = I / float(U)
                part_ious.append(iou)
            shape_ious.append(np.mean(part_ious))
            print(np.mean(part_ious))

    print("mIOU for class {}: {}".format(args.class_choice, np.mean(shape_ious)))

if __name__ == '__main__':
    run_eval()
