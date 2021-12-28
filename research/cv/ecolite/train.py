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
Use this file for standalone training and distributed training
"""
import sys
import os
import time
import mindspore as ms
from mindspore import context, DynamicLossScaleManager
from mindspore import Tensor, Model, set_seed
from mindspore import dtype as mstype
import mindspore.dataset as ds
from mindspore import nn
from mindspore import save_checkpoint
from mindspore.train.callback import LossMonitor, Callback, TimeMonitor
from mindspore.communication.management import init
from mindspore.context import ParallelMode
from model_utils.config import config
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id, get_device_num, get_rank_id
from src.dataset import create_dataset_train, create_dataset_val
from src.econet import ECONet
from src.utils import load_pretrain_checkpint

set_seed(520)


def modelarts_pre_process():
    '''modelarts pre process function.'''

    def unzip(zip_file, save_dir):
        import zipfile
        s_time = time.time()
        if not os.path.exists(os.path.join(save_dir, config.modelarts_dataset_unzip_name)):
            zip_isexist = zipfile.is_zipfile(zip_file)
            if zip_isexist:
                fz = zipfile.ZipFile(zip_file, 'r')
                data_num = len(fz.namelist())
                print("Extract Start...")
                print("unzip file num: {}".format(data_num))
                data_print = int(data_num / 100) if data_num > 100 else 1
                i = 0
                for file in fz.namelist():
                    if i % data_print == 0:
                        print("unzip percent: {}%".format(int(i * 100 / data_num)), flush=True)
                    i += 1
                    fz.extract(file, save_dir)
                print("cost time: {}min:{}s.".format(int((time.time() - s_time) / 60),
                                                     int(int(time.time() - s_time) % 60)))
                print("Extract Done.")
            else:
                print("This is not zip.")
        else:
            print("Zip has been extracted.")

    if config.need_modelarts_dataset_unzip:
        zip_file_1 = os.path.join(config.data_path, config.modelarts_dataset_unzip_name + ".zip")
        save_dir_1 = os.path.join(config.data_path)
        sync_lock = "/tmp/unzip_sync.lock"

        # Each server contains 8 devices as most.
        if get_device_id() % min(get_device_num(), 8) == 0 and not os.path.exists(sync_lock):
            print("Zip file path: ", zip_file_1)
            print("Unzip file save dir: ", save_dir_1)
            unzip(zip_file_1, save_dir_1)
            print("===Finish extract data synchronization===")
            try:
                os.mknod(sync_lock)
            except IOError:
                pass

        while True:
            if os.path.exists(sync_lock):
                break
            time.sleep(1)

        print("Device: {}, Finish sync unzip data from {} to {}.".format(get_device_id(), zip_file_1, save_dir_1))

    dirname, _ = os.path.split(os.path.abspath(sys.argv[0]))
    config.resume = os.path.join(dirname, config.resume)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_train():
    """train"""
    print("Set Context...")
    rank_size = get_device_num()
    rank_id = get_rank_id()
    device_id = get_device_id()
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, device_id=device_id)
    if config.run_distribute:
        print("Init distribute train...")
        init()
        context.set_auto_parallel_context(device_num=rank_size,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
    assert config.device_target == "Ascend"
    if config.dataset == 'ucf101':
        num_class = 101
        rgb_read_format = "{:06d}.jpg"
    elif config.dataset == 'something':
        num_class = 174
        rgb_read_format = "{:05d}.jpg"
        config.rgb_prefix = ""
    net = ECONet(num_class, config.num_segments, config.modality,
                 base_model=config.arch,
                 consensus_type=config.consensus_type, dropout=config.dropout, partial_bn=not config.no_partialbn)
    policies = net.get_optim_policies()
    train_dataset = create_dataset_train(config, rgb_read_format)
    if config.run_distribute:
        train_dataset = ds.GeneratorDataset(train_dataset, ["image", "label"], shuffle=True,
                                            num_shards=rank_size, shard_id=rank_id)
    else:
        train_dataset = ds.GeneratorDataset(train_dataset, ["image", "label"], shuffle=True)
    train_dataset = train_dataset.batch(config.batch_size, drop_remainder=True)
    val_dataset = create_dataset_val(config, rgb_read_format)
    model_dict = net.parameters_dict()
    load_pretrain_checkpint(model_dict, net)
    if config.loss_type == 'nll':
        criterion = ms.nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    else:
        raise ValueError("Unknown loss type")
    for group in policies:
        print(('group: has {} params'.format(
            len(group['params']))))
    net = net.set_train(True)
    scale_factor = 65536
    scale_window = 300
    loss_scale_manager = DynamicLossScaleManager(scale_factor, scale_window)
    optimizer = nn.SGD(policies,
                       config.lr,
                       momentum=config.momentum,
                       weight_decay=config.weight_decay, nesterov=config.nesterov)
    metrics = {'accuracy', 'loss', 'top_1_accuracy', 'top_5_accuracy'}
    model = Model(net, loss_fn=criterion, optimizer=optimizer, metrics=metrics, amp_level="O0",
                  loss_scale_manager=loss_scale_manager)
    epoch_per_eval = {"epoch": [], "acc": []}
    eval_per_epoch = config.eval_freq
    eval_cb = EvalCallBack(model, val_dataset, eval_per_epoch, epoch_per_eval)
    model.train(epoch=config.epochs, train_dataset=train_dataset, callbacks=[LossMonitor(50), eval_cb, TimeMonitor()],
                dataset_sink_mode=False)
    print('done')


class EvalCallBack(Callback):
    """Precision verification using callback function."""

    # define the operator required
    def __init__(self, models, eval_dataset, eval_per_epochs, epochs_per_eval):
        super(EvalCallBack, self).__init__()
        self.models = models
        self.eval_dataset = eval_dataset
        self.eval_per_epochs = eval_per_epochs
        self.epochs_per_eval = epochs_per_eval
        self.acc = 0
        self.saturate_cnt = 0
        self.exp_num = 2

    def adjust_learning_rate(self, opt, cb_param):
        """adjust_learning_rate"""
        decay = 0.1 ** (self.exp_num)
        lr = config.lr * decay
        opt = cb_param.optimizer
        pa = []
        for param in opt.parameters:
            pa.append(param)
        first_3d_conv_weight = pa[0:1]
        first_3d_conv_bias = pa[1: 2]
        normal_weight = pa[2:34]
        normal_bias = pa[34:66]
        bn = pa[66:126]
        lr_list = opt.get_lr_parameter(first_3d_conv_weight)
        for item in lr_list:
            item.assign_value(Tensor(lr * 1, mstype.float32))

        lr_list = opt.get_lr_parameter(first_3d_conv_bias)
        for item in lr_list:
            item.assign_value(Tensor(lr * 2, mstype.float32))

        lr_list = opt.get_lr_parameter(normal_weight)
        for item in lr_list:
            item.assign_value(Tensor(lr, mstype.float32))

        lr_list = opt.get_lr_parameter(normal_bias)
        for item in lr_list:
            item.assign_value(Tensor(lr * 2, mstype.float32))

        lr_list = opt.get_lr_parameter(bn)
        for item in lr_list:
            item.assign_value(Tensor(lr, mstype.float32))

    def epoch_begin(self, run_context):
        """epoch_begin"""
        cb_param = run_context.original_args()
        opt = cb_param.optimizer
        if self.saturate_cnt == config.num_saturate:
            self.saturate_cnt = 0
            self.exp_num = self.exp_num + 1
            print("- Learning rate decreases by a factor of '{}'".format(10 ** (self.exp_num)))
        self.adjust_learning_rate(opt, cb_param)
        cur_epoch = cb_param.cur_epoch_num
        if cur_epoch % self.eval_per_epochs == 0:
            rank_id = get_rank_id()
            if rank_id == 0:
                acc = self.models.eval(self.eval_dataset)
                if acc['accuracy'] >= self.acc:
                    self.saturate_cnt = 0
                    self.acc = acc['accuracy']
                    file_name = config.dataset + "bestacc" + ".ckpt"
                    save_checkpoint(save_obj=cb_param.train_network, ckpt_file_name=file_name)
                    print("Save the maximum accuracy checkpoint,the accuracy is", self.acc)
                else:
                    self.saturate_cnt = self.saturate_cnt + 1
                self.epochs_per_eval["epoch"].append(cur_epoch)
                self.epochs_per_eval["acc"].append(acc["accuracy"])
                print(acc)


if __name__ == '__main__':
    run_train()
