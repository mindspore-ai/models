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
"""train"""
import os

from mindspore import Model
from mindspore import context
from mindspore import nn
from mindspore.common import set_seed
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor

from src.args import args
from src.tools.callback import EvaluateCallBack
from src.tools.cell import cast_amp
from src.tools.criterion import get_criterion, NetWithLoss
from src.tools.get_misc import get_dataset, set_device, get_model, pretrained, get_train_one_step
from src.tools.optimizer import get_optimizer


def main():
    assert args.crop, f"{args.arch} is only for evaluation"
    set_seed(args.seed)
    mode = {
        0: context.GRAPH_MODE,
        1: context.PYNATIVE_MODE
    }
    context.set_context(mode=mode[args.graph_mode], device_target=args.device_target)
    if args.device_target == "GPU":
        context.set_context(enable_graph_kernel=True)
    if args.device_target == "Ascend":
        context.set_context(enable_auto_mixed_precision=True)
    rank = set_device(args)

    # get model and cast amp_level
    net = get_model(args)
    cast_amp(net)
    criterion = get_criterion(args)
    net_with_loss = NetWithLoss(net, criterion)
    if args.pretrained:
        pretrained(args, net)

    data = get_dataset(args)
    batch_num = data.train_dataset.get_dataset_size()
    optimizer = get_optimizer(args, net, batch_num)
    # save a yaml file to read to record parameters

    net_with_loss = get_train_one_step(args, net_with_loss, optimizer)

    eval_network = nn.WithEvalCell(net, criterion, args.amp_level in ["O2", "O3", "auto"])
    eval_indexes = [0, 1, 2]
    model = Model(net_with_loss, metrics={"acc", "loss"},
                  eval_network=eval_network,
                  eval_indexes=eval_indexes)

    config_ck = CheckpointConfig(save_checkpoint_steps=data.train_dataset.get_dataset_size(),
                                 keep_checkpoint_max=args.save_every)
    time_cb = TimeMonitor(data_size=data.train_dataset.get_dataset_size())

    ckpt_save_dir = "./ckpt_" + str(rank)
    if args.run_modelarts:
        ckpt_save_dir = "/cache/ckpt_" + str(rank)

    ckpoint_cb = ModelCheckpoint(prefix=args.arch + str(rank), directory=ckpt_save_dir,
                                 config=config_ck)
    loss_cb = LossMonitor()
    eval_cb = EvaluateCallBack(model, eval_dataset=data.val_dataset, src_url=ckpt_save_dir,
                               train_url=os.path.join(args.train_url, "ckpt_" + str(rank)),
                               save_freq=args.save_every)

    print("begin train")
    model.train(int(args.epochs - args.start_epoch), data.train_dataset,
                callbacks=[time_cb, ckpoint_cb, loss_cb, eval_cb],
                dataset_sink_mode=True)
    print("train success")

    if args.run_modelarts:
        import moxing as mox
        mox.file.copy_parallel(src_url=ckpt_save_dir, dst_url=os.path.join(args.train_url, "ckpt_" + str(rank)))


if __name__ == '__main__':
    main()
