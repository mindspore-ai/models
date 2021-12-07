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
"""ntsnet train."""
import argparse
import ast
import math
import os

from mindspore import Model
from mindspore import Tensor
from mindspore import context
from mindspore import dtype as mstype
from mindspore import nn
from mindspore import ops as P
from mindspore.common import set_seed
from mindspore.communication import init, get_rank
from mindspore.communication.management import get_group_size
from mindspore.context import ParallelMode
from mindspore.nn import LossBase
from mindspore.train.callback import CheckpointConfig, TimeMonitor, LossMonitor

from src.callback import EvaluateCallBack
from src.dataset_gpu import create_dataset_train, create_dataset_test
from src.lr_generator_gpu import step_lr, warmup_cosine_annealing_lr
from src.network import NTS_NET, WithLossCell, NtsnetModelCheckpoint

parser = argparse.ArgumentParser(description='ntsnet train running')
parser.add_argument("--run_modelart", type=ast.literal_eval, default=False, help="Run on modelArt, default is false.")
parser.add_argument("--run_distribute", type=ast.literal_eval, default=False, help="Run distribute, default is false.")
parser.add_argument('--data_url', default='./data',
                    help='Directory contains resnet50.ckpt and CUB_200_2011 dataset.')
parser.add_argument('--train_url', default="./", help='Directory of training output.')
parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")
parser.add_argument("--device_target", type=str, default="Ascend", help="Device Target, default Ascend",
                    choices=["Ascend", "GPU"])
parser.add_argument("--device_num", type=int, default=1, help="Use device nums, default is 1.")
args = parser.parse_args()
run_modelart = args.run_modelart

if run_modelart:
    device_id = int(os.getenv('DEVICE_ID'))
    device_num = int(os.getenv('RANK_SIZE'))
    local_input_url = '/cache/data' + str(device_id)
    local_output_url = '/cache/ckpt' + str(device_id)
    if args.device_target == "GPU":
        from src.config_gpu import config_gpu as config
    elif args.device_target == "Ascend":
        from src.config_gpu import config_ascend as config
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target,
                        save_graphs=False)
    context.set_context(device_id=device_id)
    if device_num > 1:
        init()
        context.set_auto_parallel_context(device_num=device_num,
                                          global_rank=device_id,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
        rank = get_rank()
    else:
        rank = 0
    import moxing as mox

    mox.file.copy_parallel(src_url=args.data_url, dst_url=local_input_url)
elif args.run_distribute:
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, save_graphs=False)
    init()
    device_num = get_group_size()
    context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                      gradients_mean=True)
    local_input_url = args.data_url
    local_output_url = args.train_url
    rank = get_rank()
else:
    device_id = args.device_id
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, save_graphs=False)
    context.set_context(device_id=device_id)
    rank = 0
    device_num = 1
    local_input_url = args.data_url
    local_output_url = args.train_url

learning_rate = config.learning_rate
momentum = config.momentum
weight_decay = config.weight_decay
batch_size = config.batch_size
num_train_images = config.num_train_images
num_epochs = config.num_epochs
steps_per_epoch = math.ceil(num_train_images / batch_size / device_num)
print(f"steps_per_epoch: {steps_per_epoch}")
if config.lr_scheduler == "step":
    lr = Tensor(step_lr(global_step=0,
                        lr_init=0.,
                        lr_max=learning_rate,
                        warmup_epochs=0,
                        total_epochs=num_epochs,
                        steps_per_epoch=steps_per_epoch,
                        lr_step=config.lr_step))
elif config.lr_scheduler == "cosine":
    lr = Tensor(warmup_cosine_annealing_lr(
        global_step=0,
        base_lr=learning_rate,
        steps_per_epoch=steps_per_epoch,
        warmup_epochs=0.,
        max_epoch=num_epochs
    ))
else:
    raise ValueError(f"{config.lr_scheduler} is not supported")


class CrossEntropySmooth(LossBase):
    """CrossEntropy"""

    def __init__(self, sparse=True, reduction='mean', smooth_factor=0., num_classes=1000):
        super(CrossEntropySmooth, self).__init__()
        self.onehot = P.OneHot()
        self.sparse = sparse
        self.on_value = Tensor(1.0 - smooth_factor, mstype.float32)
        self.off_value = Tensor(1.0 * smooth_factor / (num_classes - 1), mstype.float32)
        self.ce = nn.SoftmaxCrossEntropyWithLogits(reduction=reduction)
        self.cast = P.Cast()

    def construct(self, logit, label):
        if self.sparse:
            label = self.onehot(label, logit.shape[1], self.on_value, self.off_value)
        loss2 = self.ce(logit, label)
        return loss2


if __name__ == '__main__':
    set_seed(0)
    resnet50Path = os.path.join(local_input_url, "resnet50.ckpt")
    ntsnet = NTS_NET(topK=config.topK, resnet50Path=resnet50Path)
    decayed_params = []
    no_decayed_params = []
    for param in ntsnet.trainable_params():
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)

    group_params = [{'params': decayed_params, 'weight_decay': config.weight_decay},
                    {'params': no_decayed_params},
                    {'order_params': ntsnet.trainable_params()}]

    loss_fn = CrossEntropySmooth(reduction="mean", num_classes=config.num_classes, smooth_factor=0.0)
    if config.optimizer == "momentum":
        optimizer = nn.Momentum(group_params, learning_rate=lr, momentum=momentum, weight_decay=weight_decay)
    elif config.optimizer == "sgd":
        optimizer = nn.SGD(group_params, learning_rate=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"{config.optimizer} is not supported")
    loss_net = WithLossCell(ntsnet, loss_fn)
    oneStepNTSNet = nn.TrainOneStepCell(loss_net, optimizer)

    train_data_set = create_dataset_train(train_path=os.path.join(local_input_url, "CUB_200_2011/train"),
                                          batch_size=batch_size)
    dataset_size = train_data_set.get_dataset_size()
    time_cb = TimeMonitor(data_size=dataset_size)
    loss_cb = LossMonitor(per_print_times=dataset_size)

    cb = [time_cb, loss_cb]

    if config.save_checkpoint:
        save_checkpoint_path = os.path.join(local_output_url, "ckpt_" + str(rank) + "/")
        if rank == 0:
            ckptconfig = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * dataset_size,
                                          keep_checkpoint_max=config.keep_checkpoint_max)
            ckpoint_cb = NtsnetModelCheckpoint(prefix=config.prefix, directory=save_checkpoint_path,
                                               ckconfig=ckptconfig,
                                               device_num=device_num, device_id=args.device_id, args=args,
                                               run_modelart=run_modelart)
            cb += [ckpoint_cb]
        test_data_set = create_dataset_test(test_path=os.path.join(local_input_url, "CUB_200_2011/test"),
                                            batch_size=batch_size)
        eval_cb = EvaluateCallBack(model=ntsnet, eval_dataset=test_data_set, save_path=save_checkpoint_path)
        cb += [eval_cb,]

    model = Model(oneStepNTSNet, amp_level="O0", keep_batchnorm_fp32=True)
    model.train(config.num_epochs, train_data_set, callbacks=cb, dataset_sink_mode=True)
