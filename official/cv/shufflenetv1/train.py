# Copyright 2020 Huawei Technologies Co., Ltd
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
"""train ShuffleNetV1"""
import os
import time
from mindspore import context, nn
from mindspore import Tensor
from mindspore.common import set_seed
from mindspore.nn.optim.momentum import Momentum
from mindspore.context import ParallelMode
from mindspore.train.model import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor, LossMonitor, Callback
from mindspore.train.serialization import load_checkpoint, load_param_into_net, save_checkpoint
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from src.lr_generator import get_lr
from src.shufflenetv1 import ShuffleNetV1
from src.dataset import create_dataset, create_flower_dataset
from src.crossentropysmooth import CrossEntropySmooth
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id

set_seed(1)

class ValAccMonitor(Callback):
    """
    Train loss and validation accuracy monitor, after each epoch save the
    best checkpoint file with highest validation accuracy.
    """

    def __init__(self,
                 model,
                 dataset_val,
                 num_epochs,
                 interval=1,
                 eval_start_epoch=1,
                 save_best_ckpt=True,
                 ckpt_directory="./",
                 best_ckpt_name="best.ckpt",
                 metric_name="Accuracy",
                 dataset_sink_mode=True):
        super(ValAccMonitor, self).__init__()
        self.model = model
        self.dataset_val = dataset_val
        self.num_epochs = num_epochs
        self.eval_start_epoch = eval_start_epoch
        self.save_best_ckpt = save_best_ckpt
        self.metric_name = metric_name
        self.interval = interval
        self.best_res = 0
        self.dataset_sink_mode = dataset_sink_mode

        if not os.path.isdir(ckpt_directory):
            os.makedirs(ckpt_directory)
        self.best_ckpt_path = os.path.join(ckpt_directory, best_ckpt_name)

    def apply_eval(self):
        """Model evaluation, return validation accuracy."""
        return self.model.eval(self.dataset_val, dataset_sink_mode=self.dataset_sink_mode)[self.metric_name]

    def epoch_end(self, run_context):
        """
        After epoch, print train loss and val accuracy,
        save the best ckpt file with highest validation accuracy.
        """
        def remove_ckpt_file(file_name):
            os.chmod(file_name, stat.S_IWRITE)
            os.remove(file_name)

        callback_params = run_context.original_args()
        cur_epoch = callback_params.cur_epoch_num

        if cur_epoch >= self.eval_start_epoch and (cur_epoch - self.eval_start_epoch) % self.interval == 0:
            # Validation result
            res = self.apply_eval()
            print("-" * 20)
            print(f"Epoch: [{cur_epoch: 3d} / {self.num_epochs: 3d}], "
                  f"Train Loss: [{callback_params.net_outputs.asnumpy() :5.3f}], "
                  f"{self.metric_name}: {res: 5.3f}")
            # Save the best ckpt file
            if res >= self.best_res:
                self.best_res = res
                if self.save_best_ckpt:
                    if os.path.exists(self.best_ckpt_path):
                        remove_ckpt_file(self.best_ckpt_path)
                    save_checkpoint(callback_params.train_network, self.best_ckpt_path)

    # pylint: disable=unused-argument
    def end(self, run_context):
        print("=" * 80)
        print(f"End of validation the best {self.metric_name} is: {self.best_res: 5.3f}, "
              f"save the best ckpt file in {self.best_ckpt_path}", flush=True)

def modelarts_pre_process():
    pass


@moxing_wrapper(pre_process=modelarts_pre_process)
def train():
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, save_graphs=False)

    # init distributed
    if config.is_distributed:
        if os.getenv('DEVICE_ID', "not_set").isdigit():
            context.set_context(device_id=get_device_id())
        init()
        rank = get_rank()
        group_size = get_group_size()
        parallel_mode = ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(parallel_mode=parallel_mode, device_num=group_size, gradients_mean=True)
    else:
        rank = 0
        group_size = 1
        context.set_context(device_id=config.device_id)

    if config.device_target == "GPU":
        context.set_context(enable_graph_kernel=True)
    # define network
    net = ShuffleNetV1(model_size=config.model_size, n_class=config.num_classes)

    # define loss
    loss = CrossEntropySmooth(sparse=True, reduction="mean", smooth_factor=config.label_smooth_factor,
                              num_classes=config.num_classes)

    # define dataset
    if config.is_transfer:
        train_dataset, eval_dataset = create_flower_dataset(device_num=group_size, rank=rank)
        batches_per_epoch = train_dataset.get_dataset_size()
    else:
        dataset = create_dataset(config.train_dataset_path, do_train=True, device_num=group_size, rank=rank)
        batches_per_epoch = dataset.get_dataset_size()

    # resume
    if config.resume:
        ckpt = load_checkpoint(config.resume)
        if config.is_transfer:
            ckpt.pop('classifier.weight')
            ckpt.pop('classifier.bias')
            ckpt.pop('moments.classifier.weight')
            ckpt.pop('moments.classifier.bias')
        load_param_into_net(net, ckpt)

    # get learning rate
    lr = get_lr(lr_init=config.lr_init, lr_end=config.lr_end, lr_max=config.lr_max, warmup_epochs=config.warmup_epochs,
                total_epochs=config.epoch_size, steps_per_epoch=batches_per_epoch, lr_decay_mode=config.decay_method)
    lr = Tensor(lr)
    # define optimization
    optimizer = Momentum(params=net.trainable_params(), learning_rate=lr, momentum=config.momentum,
                         weight_decay=config.weight_decay, loss_scale=config.loss_scale)

    # model
    loss_scale_manager = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)

    if config.is_transfer:
        model = Model(net, loss_fn=loss, optimizer=optimizer, amp_level=config.amp_level,
                      loss_scale_manager=loss_scale_manager, metrics={'Top_1_Acc': nn.Top1CategoricalAccuracy()})
    else:
        model = Model(net, loss_fn=loss, optimizer=optimizer, amp_level=config.amp_level,
                      loss_scale_manager=loss_scale_manager)

    # define callbacks
    cb = [TimeMonitor(), LossMonitor()]
    if config.save_checkpoint:
        save_ckpt_path = config.save_ckpt_path
        if config.is_transfer:
            ckpt_cb = ValAccMonitor(model=model, dataset_val=eval_dataset, num_epochs=config.epoch_size,
                                    ckpt_directory=save_ckpt_path, best_ckpt_name="shufflenetv1_transfer_best.ckpt",
                                    metric_name="Top_1_Acc", dataset_sink_mode=False)
        else:
            config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * batches_per_epoch,
                                         keep_checkpoint_max=config.keep_checkpoint_max)
            ckpt_cb = ModelCheckpoint("shufflenetv1", directory=save_ckpt_path, config=config_ck)
        if config.is_distributed:
            if rank == 0:
                cb += [ckpt_cb]
        else:
            cb += [ckpt_cb]

    print("============== Starting Training ==============")
    start_time = time.time()
    # begin train
    if config.is_transfer:
        model.train(config.epoch_size, train_dataset, callbacks=cb, dataset_sink_mode=False)
    else:
        model.train(config.epoch_size, dataset, callbacks=cb, dataset_sink_mode=True)
    use_time = time.time() - start_time
    hour = str(int(use_time // 60 // 60))
    minute = str(int(use_time // 60 % 60))
    second = str(int(use_time % 60))
    print("total time:" + hour + "h " + minute + "m " + second + "s")
    print("============== Train Success ==============")


if __name__ == '__main__':
    train()
