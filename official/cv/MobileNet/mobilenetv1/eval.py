# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
"""eval mobilenet_v1."""
import os
import mindspore as ms
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from src.CrossEntropySmooth import CrossEntropySmooth
from src.mobilenet_v1 import mobilenet_v1 as mobilenet
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper, modelarts_process


ms.set_seed(1)

if config.dataset == 'cifar10':
    from src.dataset import create_dataset1 as create_dataset
else:
    from src.dataset import create_dataset2 as create_dataset


@moxing_wrapper(pre_process=modelarts_process)
def eval_mobilenetv1():
    """ eval_mobilenetv1 """
    if config.dataset == 'imagenet2012':
        config.dataset_path = os.path.join(config.dataset_path, 'validation_preprocess')
    print('\nconfig:\n', config)
    target = config.device_target

    # init context
    ms.set_context(mode=ms.GRAPH_MODE, device_target=target, save_graphs=False)
    if target == "Ascend":
        device_id = int(os.getenv('DEVICE_ID', '0'))
        ms.set_context(device_id=device_id)

    # create dataset
    dataset = create_dataset(dataset_path=config.dataset_path, do_train=False, batch_size=config.batch_size,
                             target=target)
    # step_size = dataset.get_dataset_size()

    # define net
    net = mobilenet(class_num=config.class_num)

    # load checkpoint
    param_dict = ms.load_checkpoint(config.checkpoint_path)
    ms.load_param_into_net(net, param_dict)
    net.set_train(False)

    # define loss, model
    if config.dataset == "imagenet2012":
        if not config.use_label_smooth:
            config.label_smooth_factor = 0.0
        loss = CrossEntropySmooth(sparse=True, reduction='mean',
                                  smooth_factor=config.label_smooth_factor, num_classes=config.class_num)
    else:
        loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    # define model
    model = ms.Model(net, loss_fn=loss, metrics={'top_1_accuracy', 'top_5_accuracy'})

    # eval model
    res = model.eval(dataset)
    print("result:", res, "ckpt=", config.checkpoint_path)


if __name__ == '__main__':
    eval_mobilenetv1()
