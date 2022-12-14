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
import os
import time

import mindspore
from mindspore import nn, context
from mindspore.train.model import Model
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as C
from mindspore import dtype as mstype
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import set_seed
from src.utils.opts import parse_opts
from src.datasets.dataset import get_val_set, get_test_set
from src.models.non_local import I3DResNet50
from src.models.resnet import resnet56

if __name__ == '__main__':
    # init options
    opt = parse_opts()
    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        opt.result_path = os.path.join(opt.root_path, opt.result_path)

    dir_time = time.strftime('%Y%m%d', time.localtime(time.time()))
    print(opt)

    # init context
    set_seed(opt.manual_seed)
    context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target='Ascend',
                        device_id=int(opt.device_id))

    # define net
    assert opt.dataset in ['kinetics', 'cifar10']
    if opt.dataset == 'kinetics':
        net = I3DResNet50()
    else:
        if opt.nl:
            print("ResNet-56 with non-local block after second residual block..")
            net = resnet56(non_local=True)
        else:
            print("ResNet-56 without non-local block..")
            net = resnet56(non_local=False)

    # define loss
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)

    # create dataset
    if opt.dataset == 'kinetics':
        if opt.mode == 'multi':
            test_data = get_test_set(opt)
        else:
            test_data = get_val_set(opt)

    else:
        type_cast_op = mindspore.dataset.transforms.c_transforms.TypeCast(mstype.int32)
        transform_test = [
            C.Rescale(1.0 / 255.0, 0.0),
            C.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            C.HWC2CHW()
        ]
        test_data = ds.Cifar10Dataset(dataset_dir=opt.test_data_path, usage='test', shuffle=False,
                                      num_parallel_workers=opt.n_threads)
        test_data = test_data.map(transform_test, input_columns=["image"])
        test_data = test_data.map(type_cast_op, input_columns=["label"])
        test_data = test_data.batch(opt.batch_size, drop_remainder=True)

    # load checkpoint
    param_dict = load_checkpoint(opt.ckpt)
    load_param_into_net(net, param_dict)
    net.set_train(False)

    # define model
    model = Model(net, loss_fn=loss,
                  metrics={'top_1_accuracy': nn.Top1CategoricalAccuracy(),
                           'top_5_accuracy': nn.Top5CategoricalAccuracy()})

    # eval model
    res = model.eval(test_data, dataset_sink_mode=False)
    print("result:", res, "ckpt:", opt.ckpt)
