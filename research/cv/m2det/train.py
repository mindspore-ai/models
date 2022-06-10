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
""" M2Det training """
import argparse
import ast
import os

from mindspore import Model
from mindspore import Tensor
from mindspore import context
from mindspore import load_checkpoint
from mindspore import nn
from mindspore import ops
from mindspore import save_checkpoint
from mindspore.common import set_seed
from mindspore.communication import get_group_size
from mindspore.communication import get_rank
from mindspore.communication import init
from mindspore.context import ParallelMode
from mindspore.train.callback import CheckpointConfig
from mindspore.train.callback import ModelCheckpoint

from src import config as cfg
from src.callback import TimeLossMonitor
from src.dataset import get_dataset
from src.loss import MultiBoxLoss
from src.lr_scheduler import get_lr
from src.model import M2DetWithLoss
from src.model import get_model
from src.priors import PriorBox
from src.utils import get_param_groups


class CustomTrainOneStepCell(nn.Cell):
    """Custom TrainOneStepCell with global gradients clipping"""

    def __init__(self, network, optimizer, max_grad_norm):
        super().__init__()
        self.network = network
        self.network.set_grad()
        self.optimizer = optimizer
        self.weights = self.optimizer.parameters
        self.grad = ops.GradOperation(get_by_list=True)
        self.reducer_flag = False
        self.grad_reducer = None
        self.max_grad_norm = max_grad_norm
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
            mean = context.get_auto_parallel_context("gradients_mean")
            degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)

    def construct(self, *inputs):
        """construct"""
        pred = self.network(*inputs)
        grads = self.grad(self.network, self.weights)(*inputs)
        if self.max_grad_norm:
            grads = ops.clip_by_global_norm(grads, clip_norm=self.max_grad_norm)
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        pred = ops.depend(pred, self.optimizer(grads))
        return pred


def parse_args():
    """Get arguments from command-line."""
    parser = argparse.ArgumentParser(description="Mindspore HRNet Training Configurations.")
    parser.add_argument("--train_url", type=str, default='./checkpoints/', help="Storage path of training results.")
    parser.add_argument("--run_distribute", type=ast.literal_eval, default=False,
                        help="Use one card or multiple cards training.")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--pretrained_backbone", type=str, default=None,
                        help="Path to pretrained backbone checkpoint")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Path to dataset root folder")

    return parser.parse_args()


def print_config():
    """Print the configuration file"""
    print('CONFIG_FILE')
    for cfg_item in dir(cfg):
        if cfg_item[0] != '_':
            item = getattr(cfg, cfg_item)
            print(f'{cfg_item} = {item}')
    print('CONFIG_FILE_END')


def set_device(device):
    """Set device"""
    if device == 'GPU':
        context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    elif device == 'Ascend':
        raise ValueError('Ascend training not implemented')
    else:
        raise ValueError(f'Unknown device type: {device}. Only "GPU" device type implemented')


def get_optimizer(net, lr, config):
    """Get optimizer"""
    opt = nn.SGD(get_param_groups(net),
                 learning_rate=Tensor(lr),
                 momentum=config.optimizer['momentum'],
                 weight_decay=config.optimizer['weight_decay'],
                 dampening=config.optimizer['dampening'])
    return opt


def main():
    """Training process."""
    set_seed(1)
    args = parse_args()

    local_train_url = args.train_url
    model_name = cfg.model['m2det_config']['backbone'] + '_' + str(cfg.model['input_size'])

    if args.pretrained_backbone:
        cfg.model['m2det_config']['checkpoint_path'] = args.pretrained_backbone

    if args.dataset_path:
        cfg.COCOroot = args.dataset_path

    device_id = args.device_id
    device = cfg.device
    set_device(device)

    # Print configuration to log
    print_config()

    # Create dataset
    if args.run_distribute:
        init('nccl')
        parallel_mode = ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True)
        priorbox = PriorBox(cfg)
        priors = priorbox.forward()
        loader, _ = get_dataset(cfg, 'COCO', priors.asnumpy(), 'train_sets',
                                random_seed=cfg.random_seed, distributed=True)
    else:
        context.set_context(device_id=device_id)
        priorbox = PriorBox(cfg)
        priors = priorbox.forward()
        loader, _ = get_dataset(cfg, 'COCO', priors.asnumpy(), 'train_sets',
                                random_seed=cfg.random_seed, distributed=False)

    loss = MultiBoxLoss(cfg.model['m2det_config']['num_classes'], cfg.loss['neg_pos'])

    # Create network
    backbone = get_model(cfg.model['m2det_config'], cfg.model['input_size'])
    if not cfg.start_epoch:
        cfg.start_epoch = 0
    if cfg.checkpoint_path is not None:
        print(f'Loading checkpoint for epoch {cfg.start_epoch}')
        print(f'Checkpoint filename: {cfg.checkpoint_path}')
        cfg.model['m2det_config']['checkpoint_path'] = None
        net = M2DetWithLoss(backbone, loss)
        load_checkpoint(cfg.checkpoint_path, net=net)
    else:
        net = M2DetWithLoss(backbone, loss)

    net.set_train()

    steps_per_epoch = loader.get_dataset_size()

    # Learning rate adjustment with linear scaling rule
    if args.run_distribute:
        n_gpus = get_group_size()
        lr_default = cfg.train_cfg['lr']
        print(f'Adjusting learning rate (default = {lr_default}) to {n_gpus} GPUs')
        cfg.train_cfg['lr'] = cfg.train_cfg['lr'] * n_gpus

    # Learning rate schedule construction
    lr = get_lr(cfg.train_cfg, steps_per_epoch)
    if cfg.start_epoch > 0:
        lr = lr[int(cfg.start_epoch * steps_per_epoch):]

    # Optimizer
    opt = get_optimizer(net, lr, cfg)

    # Create model
    net_loss_opt = CustomTrainOneStepCell(net, opt, cfg.optimizer['clip_grad_norm'])
    model = Model(net_loss_opt)

    # Callbacks
    ckpt_config = CheckpointConfig(save_checkpoint_steps=steps_per_epoch * cfg.model['checkpoint_interval'],
                                   keep_checkpoint_max=10)
    ckpt_cb = ModelCheckpoint(prefix=model_name,
                              directory=os.path.join(local_train_url, cfg.experiment_tag),
                              config=ckpt_config)

    if args.run_distribute:
        if get_rank() == 0:
            callbacks = [TimeLossMonitor(lr_init=lr), ckpt_cb]
        else:
            callbacks = [TimeLossMonitor(lr_init=lr)]
    else:
        callbacks = [TimeLossMonitor(lr_init=lr), ckpt_cb]

    # Run the training process
    model.train(cfg.train_cfg['total_epochs'] - cfg.start_epoch, loader, callbacks=callbacks, dataset_sink_mode=False)

    last_checkpoint = os.path.join(local_train_url, cfg.experiment_tag, f"{model_name}-final.ckpt")
    if args.run_distribute & (get_rank() == 0):
        save_checkpoint(net, last_checkpoint)
    elif not args.run_distribute:
        save_checkpoint(net, last_checkpoint)


if __name__ == "__main__":
    main()
