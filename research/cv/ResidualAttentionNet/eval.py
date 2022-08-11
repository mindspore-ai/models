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
import numpy as np
import mindspore as ms
from mindspore import Model, context, Tensor
import mindspore.nn as nn
from mindspore.common import set_seed
from mindspore import load_checkpoint, load_param_into_net
from src.lr_generator import get_lr
from src.model_utils.config import config
from src.dataset import create_dataset1, create_dataset2

set_seed(1)

if config.dataset == "cifar10":
    from model.residual_attention_network import ResidualAttentionModel_92_32input_update as ResidualAttentionModel
elif config.dataset == "imagenet":
    from model.residual_attention_network import ResidualAttentionModel_56 as ResidualAttentionModel
else:
    raise ValueError("Unsupported dataset.")

context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, device_id=config.device_id)

def lr_steps_cifar10(global_step, lr_max=None, total_epochs=None, steps_per_epoch=None):
    """Set learning rate."""
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    decay_epoch_index = [0.3 * total_steps, 0.6 * total_steps, 0.9 * total_steps]
    for i in range(total_steps):
        if i < decay_epoch_index[0]:
            lr_each_step.append(lr_max)
        elif i < decay_epoch_index[1]:
            lr_each_step.append(lr_max * 0.1)
        elif i < decay_epoch_index[2]:
            lr_each_step.append(lr_max * 0.01)
        else:
            lr_each_step.append(lr_max * 0.001)
    current_step = global_step
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    learning_rate = lr_each_step[current_step:]
    return learning_rate

def run_eval():
    if config.enable_modelarts:
        import moxing as mox
        obs_data_url = config.data_url
        config.data_url = '/home/work/user-job-dir/data/'
        DATA_DIR = config.data_url
        if not os.path.exists(config.data_url):
            os.mkdir(config.data_url)
        mox.file.copy_parallel(obs_data_url, config.data_url)
        print("Successfully Download {} to {}".format(obs_data_url, config.data_url))
    else:
        DATA_DIR = config.data_path
    if config.dataset == 'cifar10':
        test_dataset = create_dataset1(dataset_path=DATA_DIR, do_train=False, batch_size=config.batch_size,
                                       target=config.device_target)
    elif config.dataset == "imagenet":
        if config.enable_modelarts:
            DATA_DIR = os.path.join(DATA_DIR, 'imagenet')
        DATA_DIR = os.path.join(DATA_DIR, 'val')
        test_dataset = create_dataset2(dataset_path=DATA_DIR, do_train=False, batch_size=config.batch_size,
                                       target=config.device_target)
    else:
        raise ValueError("Unsupported dataset.")
    net = ResidualAttentionModel()
    net.set_train(False)
    batch_num = test_dataset.get_dataset_size()
    if config.dataset == 'cifar10':
        lr = lr_steps_cifar10(0, lr_max=config.lr, total_epochs=config.epoch_size, steps_per_epoch=batch_num)
    else:
        lr = ms.Tensor(get_lr(lr_init=config.lr_init, lr_end=config.lr_end, lr_max=config.lr_max,
                              warmup_epochs=config.warmup_epochs, total_epochs=config.epoch_size,
                              steps_per_epoch=batch_num, lr_decay_mode=config.lr_decay_mode))
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    optimizer = nn.Momentum(net.get_parameters(), learning_rate=Tensor(lr),
                            momentum=config.momentum, use_nesterov=True, weight_decay=config.weight_decay)
    model = Model(net, loss_fn=criterion, optimizer=optimizer, metrics={'top_1_accuracy', 'top_5_accuracy'})
    param_dict = load_checkpoint(config.checkpoint_file_path)
    load_param_into_net(net, param_dict)
    acc = model.eval(test_dataset)
    print("ckpt: ", config.checkpoint_file_path)
    print("accuracy:", acc)
if __name__ == '__main__':
    run_eval()
