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
"""eval resnet."""
import os
import numpy as np
import mindspore as ms
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore_gs import PrunerKfCompressAlgo, PrunerFtCompressAlgo
from mindspore_gs.pruner.scop.scop_pruner import KfConv2d, MaskedConv2dbn
from src.CrossEntropySmooth import CrossEntropySmooth
from src.resnet import resnet50 as resnet
from src.model_utils.config import config

if config.dataset == "cifar10":
    from src.dataset import create_dataset1 as create_dataset
else:
    from src.dataset import create_dataset2 as create_dataset

ms.set_seed(1)


def eval_net():
    """eval net"""
    target = config.device_target

    # init context
    ms.set_context(mode=ms.GRAPH_MODE, device_target=target, save_graphs=False)
    if target == "Ascend":
        device_id = int(os.getenv('DEVICE_ID'))
        ms.set_context(device_id=device_id)

    # create dataset
    dataset = create_dataset(dataset_path=config.data_path, do_train=False, batch_size=config.batch_size,
                             eval_image_size=config.eval_image_size, target=target)

    # define net
    net = resnet(class_num=config.class_num)
    net = PrunerKfCompressAlgo({}).apply(net)
    out_index = []
    param_dict = ms.load_checkpoint(config.checkpoint_file_path)
    for key in param_dict.keys():
        if 'out_index' in key:
            out_index.append(param_dict[key])
    for _, (_, module) in enumerate(net.cells_and_names()):
        if isinstance(module, KfConv2d):
            module.out_index = out_index.pop(0)
    net = PrunerFtCompressAlgo({}).apply(net)

    # load checkpoint
    ms.load_param_into_net(net, param_dict)
    net.set_train(False)

    # define loss, model
    if config.dataset == "imagenet2012":
        if not config.use_label_smooth:
            config.label_smooth_factor = 0.0
        loss = CrossEntropySmooth(sparse=True, reduction='mean',
                                  smooth_factor=config.label_smooth_factor,
                                  num_classes=config.class_num)
    else:
        loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    # define model
    model = ms.Model(net, loss_fn=loss, metrics={'top_1_accuracy'})

    # eval model
    res = model.eval(dataset)
    masked_conv_list = []
    for imd, (nam, module) in enumerate(net.cells_and_names()):
        if isinstance(module, MaskedConv2dbn):
            masked_conv_list.append((nam, module))
    for imd in range(len(masked_conv_list)):
        if 'conv2' in masked_conv_list[imd][0] or 'conv3' in masked_conv_list[imd][0]:
            masked_conv_list[imd][1].in_index = masked_conv_list[imd - 1][1].out_index

    # Only use when calculate params, next version will provide the interface.
    net = PrunerFtCompressAlgo({})._pruning_conv(net)

    # calculate params
    total_params = 0
    for param in net.trainable_params():
        total_params += np.prod(param.shape)

    print("result:", res, "prune_rate=", config.prune_rate,
          "ckpt=", config.checkpoint_file_path, "params=", total_params)


if __name__ == '__main__':
    eval_net()
