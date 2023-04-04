# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
import mindspore as ms
from mindspore import Tensor
from mindspore.nn.optim import Momentum
from mindspore.common import set_seed
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.train.model import Model
from src.CrossEntropySmooth import CrossEntropySmooth
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper

set_seed(1)

if config.net_name in ("resnet18", "resnet34", "resnet50", "resnet152"):
    if config.net_name == "resnet18":
        from src.resnet import resnet18 as resnet
    elif config.net_name == "resnet34":
        from src.resnet import resnet34 as resnet
    elif config.net_name == "resnet50":
        from src.resnet import resnet50 as resnet
    else:
        from src.resnet import resnet152 as resnet
    if config.dataset == "cifar10":
        from src.dataset import create_dataset1 as create_dataset
    else:
        from src.dataset import create_dataset2 as create_dataset
elif config.net_name == "resnet101":
    from src.resnet import resnet101 as resnet
    from src.dataset import create_dataset3 as create_dataset
else:
    from src.resnet import se_resnet50 as resnet
    from src.dataset import create_dataset4 as create_dataset


def init_group_params(net):
    decayed_params = []
    no_decayed_params = []
    for param in net.trainable_params():
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)

    group_params = [{'params': decayed_params, 'weight_decay': config.weight_decay},
                    {'params': no_decayed_params},
                    {'order_params': net.trainable_params()}]
    return group_params


@moxing_wrapper()
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
                             eval_image_size=config.eval_image_size,
                             target=target)

    # define net
    net = resnet(class_num=config.class_num)

    # load checkpoint
    param_dict = ms.load_checkpoint(config.checkpoint_file_path)
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

    #Currently, boost inference only supports scenarios with optimizers
    #Optimizer waiting for decoupling in boost model
    group_params = init_group_params(net)
    opt = Momentum(group_params, Tensor(0.0), config.momentum, loss_scale=config.loss_scale)

    # define model, add boostmode for eval scenarios with train.py
    model = Model(net, loss_fn=loss, boost_level=config.boost_mode,
                  optimizer=opt, metrics={'top_1_accuracy', 'top_5_accuracy'})

    # eval model
    res = model.eval(dataset)
    print("result:", res, "ckpt=", config.checkpoint_file_path)

if __name__ == '__main__':
    eval_net()
