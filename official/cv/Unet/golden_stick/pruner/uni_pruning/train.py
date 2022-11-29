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
import logging

import mindspore
import mindspore.nn as nn
from mindspore import Model, context
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore_gs.pruner.uni_pruning import UniPruner

from src.unet_medical import UNetMedical
from src.data_loader import create_dataset, create_multi_class_dataset
from src.loss import CrossEntropyWithLogits, MultiCrossEntropyWithLogits
from src.utils import StepLossTimeMonitor
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper

mindspore.set_seed(1)

@moxing_wrapper()
def train_net(cross_valid_ind=1,
              epochs=400,
              batch_size=16,
              lr=0.0001):
    rank = 0
    group_size = 1
    data_dir = config.data_path
    run_distribute = config.run_distribute
    if run_distribute:
        init()
        group_size = get_group_size()
        rank = get_rank()
        parallel_mode = ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(parallel_mode=parallel_mode,
                                          device_num=group_size,
                                          gradients_mean=False)
    net = UNetMedical(n_channels=config.num_channels, n_classes=config.num_classes)

    if config.resume:
        param_dict = load_checkpoint(config.resume_ckpt)
        load_param_into_net(net, param_dict)

    # pruner network
    input_size = [config.batch_size, config.num_channels, config.image_size[0], config.image_size[1]]
    algo = UniPruner({"exp_name": config.exp_name,
                      "frequency": config.frequency,
                      "target_sparsity": 1 - config.prune_rate,
                      "pruning_step": config.pruning_step,
                      "filter_lower_threshold": config.filter_lower_threshold,
                      "input_size": input_size,
                      "output_path": config.output_path,
                      "prune_flag": config.prune_flag,
                      "rank": rank,
                      "device_target": config.device_target})
    algo.apply(net)

    if hasattr(config, "use_ds") and config.use_ds:
        criterion = MultiCrossEntropyWithLogits()
    else:
        criterion = CrossEntropyWithLogits()
    if hasattr(config, "dataset") and config.dataset != "ISBI":
        dataset_sink_mode = True
        per_print_times = 0
        repeat = config.repeat if hasattr(config, "repeat") else 1
        split = config.split if hasattr(config, "split") else 0.8
        train_dataset = create_multi_class_dataset(data_dir, config.image_size, repeat, batch_size,
                                                   num_classes=config.num_classes, is_train=True,
                                                   augment=config.train_augment, split=split,
                                                   rank=rank, group_size=group_size, shuffle=True)
    else:
        repeat = config.repeat
        dataset_sink_mode = False
        if config.device_target == "GPU":
            dataset_sink_mode = True
        per_print_times = 1
        train_dataset, _ = create_dataset(data_dir, repeat, batch_size, True, cross_valid_ind, run_distribute,
                                          config.crop, config.image_size)
    train_data_size = train_dataset.get_dataset_size()
    print("dataset length is:", train_data_size)
    ckpt_save_dir = os.path.join(config.output_path, config.checkpoint_path, f'ckpt_{rank}')
    save_ck_steps = train_data_size * epochs
    ckpt_config = CheckpointConfig(save_checkpoint_steps=save_ck_steps,
                                   keep_checkpoint_max=config.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix='ckpt_{}_adam'.format(config.model_name),
                                 directory=ckpt_save_dir,
                                 config=ckpt_config)

    optimizer = nn.Adam(params=net.trainable_params(), learning_rate=lr, weight_decay=config.weight_decay,
                        loss_scale=config.loss_scale)

    loss_scale_manager = mindspore.train.loss_scale_manager.FixedLossScaleManager(config.FixedLossScaleManager, False)
    amp_level = "O0" if config.device_target == "GPU" else "O3"
    model = Model(net, loss_fn=criterion, loss_scale_manager=loss_scale_manager, optimizer=optimizer,
                  amp_level=amp_level)

    print("============== Starting Training ==============")
    callbacks = [StepLossTimeMonitor(batch_size=batch_size, per_print_times=per_print_times), ckpoint_cb]
    algo_cb = algo.callbacks()[0]
    callbacks.append(algo_cb)

    model.train(int(epochs / repeat), train_dataset, callbacks=callbacks, dataset_sink_mode=dataset_sink_mode)
    print("============== End Training ==============")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    # to keep GetNext from timeout, set op_timeout=600
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, save_graphs=False, op_timeout=600)
    assert config.device_target == "GPU"
    epoch_size = config.epoch_size if not config.run_distribute else config.distribute_epochs
    batchsize = config.batch_size
    if config.run_distribute:
        batchsize = config.distribute_batchsize
    train_net(cross_valid_ind=config.cross_valid_ind,
              epochs=epoch_size,
              batch_size=batchsize,
              lr=config.lr)
