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
"""
################################eval glore_resnet series################################
python eval.py
"""

import os
import random
import numpy as np

from mindspore import context
from mindspore import dataset as de
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.glore_resnet import glore_resnet200, glore_resnet50, glore_resnet101
from src.dataset import create_eval_dataset
from src.dataset import create_dataset_ImageNet as ImageNet
from src.loss import CrossEntropySmooth, SoftmaxCrossEntropyExpand
from src.config import config

if config.isModelArts:
    import moxing as mox
if config.net == 'resnet200':
    if config.device_target == "GPU":
        config.cast_fp16 = False


random.seed(1)
np.random.seed(1)
de.config.set_seed(1)

if __name__ == '__main__':
    target = config.device_target
    # init context
    device_id = config.device_id
    context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False,
                        device_id=device_id)

    # dataset
    eval_dataset_path = os.path.abspath(config.eval_data_url)
    if config.isModelArts:
        mox.file.copy_parallel(src_url=config.eval_data_url, dst_url='/cache/dataset')
        eval_dataset_path = '/cache/dataset/'
    if config.net == 'resnet50':
        predict_data = create_eval_dataset(dataset_path=eval_dataset_path, repeat_num=1, batch_size=config.batch_size)
    else:
        predict_data = ImageNet(dataset_path=eval_dataset_path,
                                do_train=False,
                                repeat_num=1,
                                batch_size=config.batch_size,
                                target=target)
    step_size = predict_data.get_dataset_size()
    if step_size == 0:
        raise ValueError("Please check dataset size > 0 and batch_size <= dataset size")

    # define net
    if config.net == 'resnet50':
        net = glore_resnet50(class_num=config.class_num, use_glore=config.use_glore)
    elif config.net == 'resnet200':
        net = glore_resnet200(class_num=config.class_num, use_glore=config.use_glore)
    elif config.net == 'resnet101':
        net = glore_resnet101(class_num=config.class_num, use_glore=config.use_glore)

    # load checkpoint
    param_dict = load_checkpoint(config.ckpt_url)
    load_param_into_net(net, param_dict)

    # define loss, model
    if config.net == 'resnet50':
        if config.use_label_smooth:
            loss = CrossEntropySmooth(sparse=True, reduction="mean", smooth_factor=config.label_smooth_factor,
                                      num_classes=config.class_num)
    else:
        loss = SoftmaxCrossEntropyExpand(sparse=True)
    model = Model(net, loss_fn=loss, metrics={'top_1_accuracy', 'top_5_accuracy'})
    print("============== Starting Testing ==============")
    print("ckpt path : {}".format(config.ckpt_url))
    print("data path : {}".format(eval_dataset_path))
    acc = model.eval(predict_data)
    print("==============Acc: {} ==============".format(acc))
