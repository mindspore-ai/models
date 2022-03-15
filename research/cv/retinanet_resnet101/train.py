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

"""Train retinanet and get checkpoint files."""

import os
import mindspore
import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, LossMonitor, TimeMonitor, Callback
from mindspore.train import Model
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import set_seed
from src.retinahead import retinanetWithLossCell, TrainingWrapper, retinahead
from src.backbone import resnet101
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_num, get_device_id
from src.dataset import create_retinanet_dataset, create_mindrecord
from src.lr_schedule import get_lr
from src.init_params import init_net_param, filter_checkpoint_parameter


set_seed(1)


class Monitor(Callback):
    """
    Monitor loss and time.

    Args:
        lr_init (numpy array): train lr

    Returns:
        None

    Examples:
        >>> Monitor(100,lr_init=Tensor([0.05]*100).asnumpy())
    """

    def __init__(self, lr_init=None):
        super(Monitor, self).__init__()
        self.lr_init = lr_init
        self.lr_init_len = len(lr_init)

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        print("lr:[{:8.6f}]".format(self.lr_init[cb_params.cur_step_num-1]), flush=True)


def modelarts_process():
    config.save_checkpoint_path = os.path.join(config.output_path, str(get_device_id()), config.save_checkpoint_path)
    if config.need_modelarts_dataset_unzip:
        config.coco_root = os.path.join(config.coco_root, config.modelarts_dataset_unzip_name)
        print(os.listdir(os.path.join(config.data_path, config.modelarts_dataset_unzip_name)))


@moxing_wrapper(pre_process=modelarts_process)
def train_retinanet_resnet101():
    """ train_retinanet_resnet101 """

    context.set_context(mode=context.GRAPH_MODE, device_target=config.run_platform)
    if config.run_platform == "Ascend":
        if config.distribute:
            if os.getenv("DEVICE_ID", "not_set").isdigit():
                context.set_context(device_id=int(os.getenv("DEVICE_ID")))
            init()
            device_num = get_device_num()
            rank = get_rank()
            context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                              device_num=device_num)
        else:
            rank = 0
            device_num = 1
            context.set_context(device_id=get_device_id())

    elif config.run_platform == "GPU":
        rank = config.device_id
        device_num = config.device_num
        if config.distribute:
            init()
            rank = get_rank()
            device_num = get_group_size()
            context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                              device_num=device_num)

    else:
        raise ValueError("Unsupported platform, GPU or Ascend is supported only.")

    mindrecord_file = create_mindrecord(config.dataset, "retina2.mindrecord", True)

    if not config.only_create_dataset:
        loss_scale = float(config.loss_scale)

        # When create MindDataset, using the fitst mindrecord file, such as retinanet.mindrecord0.
        dataset = create_retinanet_dataset(mindrecord_file, repeat_num=1,
                                           batch_size=config.batch_size, device_num=device_num, rank=rank)

        dataset_size = dataset.get_dataset_size()
        print("Create dataset done!")

        backbone = resnet101(config.num_classes)
        retinanet = retinahead(backbone, config)
        net = retinanetWithLossCell(retinanet, config)
        if config.run_platform == "Ascend":
            net.to_float(mindspore.float16)
        else:
            net.to_float(mindspore.float32)
        init_net_param(net)

        if config.pre_trained:
            if config.pre_trained_epoch_size <= 0:
                raise KeyError("pre_trained_epoch_size must be greater than 0.")
            param_dict = load_checkpoint(config.pre_trained)
            if config.filter_weight:
                filter_checkpoint_parameter(param_dict)
            load_param_into_net(net, param_dict)

        lr = Tensor(get_lr(global_step=config.global_step,
                           lr_init=config.lr_init, lr_end=config.lr_end_rate * config.lr, lr_max=config.lr,
                           warmup_epochs1=config.warmup_epochs1, warmup_epochs2=config.warmup_epochs2,
                           warmup_epochs3=config.warmup_epochs3, warmup_epochs4=config.warmup_epochs4,
                           warmup_epochs5=config.warmup_epochs5, total_epochs=config.epoch_size,
                           steps_per_epoch=dataset_size))
        opt = nn.Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), lr,
                          config.momentum, config.weight_decay, loss_scale)
        net = TrainingWrapper(net, opt, loss_scale)
        model = Model(net)
        print("Start train retinanet, the first epoch will be slower because of the graph compilation.")
        cb = [TimeMonitor(), LossMonitor()]
        cb += [Monitor(lr_init=lr.asnumpy())]
        config_ck = CheckpointConfig(save_checkpoint_steps=dataset_size * config.save_checkpoint_epochs,
                                     keep_checkpoint_max=config.keep_checkpoint_max)
        ckpt_cb = ModelCheckpoint(prefix="retinanet", directory=config.save_checkpoint_path, config=config_ck)
        if config.distribute:
            if rank == 0:
                cb += [ckpt_cb]
            model.train(config.epoch_size, dataset, callbacks=cb, dataset_sink_mode=True)
        else:
            cb += [ckpt_cb]
            model.train(config.epoch_size, dataset, callbacks=cb, dataset_sink_mode=True)


if __name__ == '__main__':
    train_retinanet_resnet101()
