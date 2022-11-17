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
import json
import numpy as np
import mindspore as ms
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore_gs.pruner.uni_pruning import UniPruner

from src.CrossEntropySmooth import CrossEntropySmooth
from src.resnet import resnet18, resnet50
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
    else:
        device_id = config.device_id
    ms.set_context(device_id=device_id)

    # create dataset
    dataset = create_dataset(dataset_path=config.data_path, do_train=False, batch_size=config.batch_size,
                             eval_image_size=config.eval_image_size, target=target)

    # define net
    if config.net_name == 'resnet18':
        net = resnet18(class_num=config.class_num)
    elif config.net_name == 'resnet50':
        net = resnet50(class_num=config.class_num)
    net.set_train(False)
    input_size = [config.export_batch_size, 3, config.height, config.width]
    algo = UniPruner({"exp_name": config.exp_name,
                      "frequency": config.retrain_epochs,
                      "target_sparsity": 1 - config.prune_rate,
                      "pruning_step": config.pruning_step,
                      "filter_lower_threshold": config.filter_lower_threshold,
                      "input_size": input_size,
                      "output_path": config.output_path,
                      "prune_flag": config.prune_flag,
                      "rank": config.device_id,
                      "device_target": config.device_target})
    net = algo.apply(net)
    if config.mask_path is not None and os.path.exists(config.mask_path):
        with open(config.mask_path, 'r', encoding='utf8') as json_fp:
            mask = json.load(json_fp)
        tag = 'pruned'
    else:
        mask = None
        tag = 'original'
    ms.load_checkpoint(config.checkpoint_file_path, net)

    algo.prune_by_mask(net, mask, config, tag)

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
    # calculate params
    total_params = 0
    for param in net.trainable_params():
        total_params += np.prod(param.shape)

    print("result:", res, "prune_rate=", config.prune_rate,
          "ckpt=", config.checkpoint_file_path, "params=", total_params)


if __name__ == '__main__':
    eval_net()
