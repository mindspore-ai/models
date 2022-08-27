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
"""test tsn"""
import time
import datetime
import argparse
import numpy as np
import onnxruntime
from sklearn.metrics import confusion_matrix

from mindspore import context

from src.dataset import create_dataset
from src.transforms import Stack, ToTorchFormatTensor, GroupNormalize, GroupScale, GroupCenterCrop, GroupOverSample

parser = argparse.ArgumentParser(description="Standard video-level testing")
parser.add_argument('--onnx_path', type=str, default="")
parser.add_argument('--dataset', type=str, default="ucf101", choices=['ucf101', 'hmdb51', 'kinetics'])
parser.add_argument('--modality', type=str, default="RGB", choices=['RGB', 'Flow', 'RGBDiff'])
parser.add_argument('--test_list', type=str, default="")
parser.add_argument('--dataset_path', type=str, default="")
parser.add_argument('--arch', type=str, default="BNInception")
parser.add_argument('--save_scores', type=str, default="score_warmup")
parser.add_argument('--test_segments', type=int, default=25)
parser.add_argument('--test_crops', type=int, default=10)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--crop_fusion_type', type=str, default='avg',
                    choices=['avg', 'max', 'topk'])
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
parser.add_argument('--flow_prefix', type=str, default='flow_')
parser.add_argument('--device_id', default=0, type=int)
parser.add_argument('--platform', type=str, default='Ascend', choices=['Ascend', 'GPU', 'CPU'],
                    help='Running platform, only support Ascend now. Default is GPU.')


args = parser.parse_args()
context.set_context(mode=context.GRAPH_MODE, device_target=args.platform, device_id=args.device_id)

if args.platform == 'GPU':
    providers = ['CUDAExecutionProvider']
elif args.platform == 'CPU':
    providers = ['CPUExecutionProvider']
else:
    raise ValueError(
        f'Unsupported target device {args.platform}, '
        f'Expected one of: "CPU", "GPU"'
    )

test_start = datetime.datetime.now()

if args.dataset == 'ucf101':
    num_class = 101
elif args.dataset == 'hmdb51':
    num_class = 51
elif args.dataset == 'kinetics':
    num_class = 400
else:
    raise ValueError('Unknown dataset '+args.dataset)

session = onnxruntime.InferenceSession(args.onnx_path, providers=providers)

crop_size = 224
scale_size = 256
input_mean = [104, 117, 128]
input_std = [1]
if args.modality == 'Flow':
    input_mean = [128]

if args.modality == 'RGB':
    data_length = 1
    args.flow_prefix = ''
    args.dropout = 0.2
elif args.modality in ['Flow', 'RGBDiff']:
    data_length = 5

transform = []

if args.test_crops == 1:
    transform.append(GroupScale(scale_size))
    transform.append(GroupCenterCrop(crop_size))
elif args.test_crops == 10:
    transform.append(GroupOverSample(crop_size, scale_size))

transform.append(Stack(roll=args.arch == 'BNInception'))
transform.append(ToTorchFormatTensor(div=args.arch != 'BNInception'))
transform.append(GroupNormalize(input_mean, input_std))

image_tmpl = "img_{:05d}.jpg" if args.modality in ["RGB", "RGBDiff"] else args.flow_prefix+"{}_{:05d}.jpg"
data_loader = create_dataset(root_path=args.dataset_path, list_file=args.test_list,
                             batch_size=1, num_segments=args.test_segments, new_length=data_length,
                             modality=args.modality, image_tmpl=image_tmpl, transform=transform,
                             worker=args.workers, test_mode=2, run_distribute=False)

total_num = data_loader.get_dataset_size()

output = []

proc_start_time = time.time()
'''
reshape = ops.Reshape()
cast = ops.Cast()

def eval_video(video_data):
    """process input and compute"""
    j, images, label = video_data
    num_crop = args.test_crops

    if args.modality == 'RGB':
        length = 3
    elif args.modality == 'Flow':
        length = 10
    elif args.modality == 'RGBDiff':
        length = 18
    else:
        raise ValueError("Unknown modality "+args.modality)
    input_var = reshape(images, (-1, length, images.shape[2], images.shape[3]))

    if args.modality == 'RGBDiff':
        reverse = list(range(data_length, 0, -1))
        input_c = 3
        tmp = input_var.asnumpy()
        input_view = tmp.reshape((-1, args.test_segments, data_length + 1, input_c,) + tmp.shape[2:])

        new_data = input_view[:, :, 1:, :, :, :].copy()
        for x in reverse:
            new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
        input_var = Tensor(new_data, mstype.float32)

    rst = model(input_var).asnumpy().copy()
    rst = rst.reshape((num_crop, args.test_segments,\
         num_class)).mean(axis=0).reshape((args.test_segments, 1, num_class))

    return j, rst, label.asnumpy().tolist()
'''
for i, data in enumerate(data_loader.create_dict_iterator(output_numpy=True)):
    step_start = time.time()
    images, label = data['input'], data['label'].tolist()
    num_crop = args.test_crops

    if args.modality == 'RGB':
        length = 3
    elif args.modality == 'Flow':
        length = 10
    elif args.modality == 'RGBDiff':
        length = 18
    else:
        raise ValueError("Unknown modality " + args.modality)
    input_var = np.reshape(images, (-1, length, images.shape[2], images.shape[3]))
    if args.modality == 'RGBDiff':
        reverse = list(range(data_length, 0, -1))
        input_c = 3
        tmp = input_var
        input_view = tmp.reshape((-1, args.test_segments, data_length + 1, input_c,) + tmp.shape[2:])

        new_data = input_view[:, :, 1:, :, :, :].copy()
        for x in reverse:
            new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
        input_var = new_data

    #rst = model(input_var).asnumpy().copy()
    inputs = {session.get_inputs()[0].name: input_var}
    rst = session.run(None, inputs)[0]
    rst = rst.reshape((num_crop, args.test_segments, \
                       num_class)).mean(axis=0).reshape((args.test_segments, 1, num_class))
    res = (i, rst, label)
    #res = eval_video((i, data['input'], data['label']))
    output.append(res[1:])
    step_end = time.time()
    cnt_time = step_end - proc_start_time
    this_step = step_end - step_start
    print('step: {} , total: {}/{}, time used: {:.2f} , average {:.2f} sec/step'.format((i+1),\
         (i+1), total_num, this_step, float(cnt_time) / (i+1)))

video_pred = [np.argmax(np.mean(x[0], axis=0)) for x in output]
video_labels = [x[1] for x in output]

cf = confusion_matrix(video_labels, video_pred).astype(float)
cls_cnt = cf.sum(axis=1)
cls_hit = np.diag(cf)

cls_acc = cls_hit / cls_cnt

print('Accuracy {:.01f}%'.format(np.mean(cls_acc) * 100))

if args.save_scores is not None:

    # reorder before saving
    name_list = [x.strip().split()[0] for x in open(args.test_list)]

    order_dict = {e: i for i, e in enumerate(sorted(name_list))}

    reorder_output = [None] * len(output)
    reorder_label = [None] * len(output)

    for i in range(len(output)):
        idx = order_dict[name_list[i]]
        reorder_output[idx] = output[i]
        reorder_label[idx] = video_labels[i]

    np.savez(args.save_scores+args.modality, scores=reorder_output, labels=reorder_label)
