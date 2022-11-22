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
##############export checkpoint file into onnx models#################
python export.py
"""
import os
import numpy as np
import mindspore as ms
import mindspore.dataset as ds
from mindspore import Tensor

from src.args import args
import src.ipt_model as ipt
import src.ipt_post_onnx as ipt_onnx
from src.data.srdata import SRData

device_id = int(os.getenv('DEVICE_ID', '0'))
ms.set_context(mode=ms.GRAPH_MODE, device_target="GPU", device_id=device_id, save_graphs=False)
ms.set_context(max_call_depth=10000)

def sub_mean(x):
    red_channel_mean = 0.4488 * 255
    green_channel_mean = 0.4371 * 255
    blue_channel_mean = 0.4040 * 255
    x[:, 0, :, :] -= red_channel_mean
    x[:, 1, :, :] -= green_channel_mean
    x[:, 2, :, :] -= blue_channel_mean
    return x

def run_export():
    """run export."""
    for arg in vars(args):
        if vars(args)[arg] == 'True':
            vars(args)[arg] = True
        elif vars(args)[arg] == 'False':
            vars(args)[arg] = False

    net_m = ipt.IPT(args)
    if args.pth_path:
        param_dict = ms.load_checkpoint(args.pth_path)
        ms.load_param_into_net(net_m, param_dict)
    net_m.set_train(False)
    inference = ipt_onnx.IPT_post(net_m, args)
    print('load mindspore net successfully.')

    train_dataset = SRData(args, name=args.data_test, train=False, benchmark=False)
    train_de_dataset = ds.GeneratorDataset(train_dataset, ['LR', 'HR', "idx", "filename"], shuffle=False)
    train_de_dataset = train_de_dataset.batch(1, drop_remainder=True)
    train_loader = train_de_dataset.create_dict_iterator(output_numpy=True)

    if args.task_id == 0:
        idx = Tensor(np.ones(args.task_id+6), ms.int32)
    else:
        idx = Tensor(np.ones(args.task_id), ms.int32)
    for imgs in train_loader:
        lr = imgs['LR']
        lr = sub_mean(lr)
        lr = Tensor(lr, ms.float16)
        shape_onnx = inference.forward(lr, idx)
        shape = list(set(shape_onnx))
    for i in shape:
        args.file_name = str(i[0]) + '_' + str(i[1])
        file_name = os.path.join(args.save, args.file_name)
        input_x = Tensor(np.ones(list(i), np.float16))
        ms.export(net_m, input_x, idx, file_name=file_name, file_format=args.file_format)

if __name__ == '__main__':
    run_export()
