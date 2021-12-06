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
# ===========================================================================
"""Eval Auto-DeepLab"""
import os
import numpy as np

import mindspore
from mindspore import Tensor
from mindspore import load_checkpoint, load_param_into_net
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.communication import init

from src.config import obtain_autodeeplab_args
from src.core.model import AutoDeepLab
from src.utils.cityscapes import CityScapesDataset
from src.utils.utils import fast_hist, BuildEvalNetwork, rescale_batch

device_id = int(os.getenv('DEVICE_ID'))
device_num = int(os.getenv('RANK_SIZE'))
mindspore.set_seed(0)


def evaluate():
    """evaluate"""
    context.set_context(mode=context.GRAPH_MODE,
                        save_graphs=False,
                        device_target="Ascend",
                        device_id=int(os.getenv('DEVICE_ID')))

    args = obtain_autodeeplab_args()
    args.scale = (1.0,)
    args.scales = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75)

    ckpt_file = ""
    if args.modelArts:
        import moxing as mox
        init()
        context.set_auto_parallel_context(device_num=device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
        local_data_url = "/cache/data"
        local_eval_url = "/cache/eval"
        local_img_url = "/cache/eval/image"
        mox.file.make_dirs(local_img_url)
        device_data_url = os.path.join(local_data_url, "device{0}".format(device_id))
        device_train_url = os.path.join(local_eval_url, "device{0}".format(device_id))
        local_train_file = os.path.join(device_data_url, 'cityscapes_train.mindrecord')
        local_val_file = os.path.join(device_data_url, 'cityscapes_val.mindrecord')
        if args.ckpt_name is not None:
            ckpt_file = os.path.join(device_data_url, args.ckpt_name)
        mox.file.make_dirs(local_data_url)
        mox.file.make_dirs(local_eval_url)
        mox.file.make_dirs(device_data_url)
        mox.file.make_dirs(device_train_url)
        mox.file.copy_parallel(src_url=args.data_url, dst_url=device_data_url)
        os.system("ls -l /cache/data/")
    else:
        if args.parallel:
            init()
            context.set_auto_parallel_context(device_num=device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
        local_train_file = os.path.join(args.data_path, 'cityscapes_train.mindrecord')
        local_val_file = os.path.join(args.data_path, 'cityscapes_val.mindrecord')
        if args.ckpt_name is not None:
            ckpt_file = args.ckpt_name

    # define dataset
    batch_size = args.batch_size

    if args.split == 'train':
        eval_ds = CityScapesDataset(local_train_file, 'eval', args.ignore_label, None, None, None, shuffle=False)
        eval_ds = eval_ds.batch(batch_size=batch_size)
    elif args.split == 'val':
        eval_ds = CityScapesDataset(local_val_file, 'eval', args.ignore_label, None, None, None, shuffle=False)
        eval_ds = eval_ds.batch(batch_size=batch_size)
    else:
        raise ValueError("Unknown dataset split")

    # net
    args.total_iters = 0
    autodeeplab = AutoDeepLab(args=args)

    # load checkpoint
    param_dict = load_checkpoint(ckpt_file)
    load_param_into_net(autodeeplab, param_dict)

    net = BuildEvalNetwork(autodeeplab)
    net.set_train(False)

    print("start eval")
    num_classes = args.num_classes
    hist = np.zeros((num_classes, num_classes))
    for _, data in enumerate(eval_ds):
        inputs = data[0].asnumpy().copy()
        label = data[1].asnumpy().copy().astype(np.uint8)

        n, h, w = label.shape
        eval_scale = args.scales if args.ms_infer else args.scale
        total_pred = np.zeros((n, args.num_classes, h, w))

        for _, scale in enumerate(eval_scale):
            new_scale = [int(h * scale), int(w * scale)]

            scaled_batch = rescale_batch(inputs, new_scale)
            scaled_pred = net(Tensor(scaled_batch, mindspore.float32))
            scaled_pred = rescale_batch(scaled_pred.asnumpy().copy(), [h, w])

            total_pred += scaled_pred

            if args.eval_flip:
                flip_batch = scaled_batch
                flip_batch = Tensor(flip_batch[:, :, :, ::-1], mindspore.float32)
                flip_pred = net(flip_batch)
                flip_pred = rescale_batch(flip_pred.asnumpy().copy(), [h, w])

                total_pred += flip_pred[:, :, :, ::-1]

        pred = total_pred.argmax(1).astype(np.uint8)

        for j in range(n):
            hist = hist + fast_hist(pred[j].flatten(), label[j].flatten(), num_classes)
    miou = np.diag(hist) / (hist.sum(0) + hist.sum(1) - np.diag(hist) + 1e-10)
    miou = round(np.nanmean(miou) * 100, 2)
    print("eval mean IOU = {0:.2f}%".format(miou))

    if args.modelArts:
        mox.file.copy_parallel(src_url='/tmp', dst_url=args.train_url)
    return 0


if __name__ == "__main__":
    evaluate()
