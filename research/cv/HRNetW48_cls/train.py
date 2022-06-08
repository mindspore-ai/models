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
"""train hrnet for classification."""
import argparse
import ast
import os

from mindspore import context
from mindspore import nn
from mindspore import save_checkpoint
from mindspore.common import dtype as mstype
from mindspore.common import set_seed
from mindspore.communication.management import get_rank
from mindspore.communication.management import init
from mindspore.train.callback import CheckpointConfig
from mindspore.train.callback import ModelCheckpoint
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.model import Model
from mindspore.train.model import ParallelMode
from mindspore.train.serialization import load_checkpoint
from mindspore.train.serialization import load_param_into_net

from src.callback import EvalCallBack
from src.callback import TimeLossMonitor
from src.cls_hrnet import get_cls_model
from src.config import config_hrnetw48_cls as config
from src.config import organize_configuration
from src.dataset import create_imagenet
from src.loss import CrossEntropySmooth
from src.model_utils.moxing_adapter import moxing_wrapper
from src.utils import get_linear_lr
from src.utils import params_initializer

set_seed(1)


def parse_args():
    """Get arguments from terminal."""
    parser = argparse.ArgumentParser(description='Image classification training.')
    # Path parameter
    parser.add_argument('--data_url', type=str, default=None, help='Dataset path.')
    parser.add_argument('--data_path', type=str, default=None, help='Dataset path.')
    parser.add_argument('--train_url', type=str, default=None, help='Train output path.')
    parser.add_argument('--train_path', type=str, default=None, help='Train output path.')
    parser.add_argument('--checkpoint_url', type=str, default=None, help='Resume training with existed checkpoint.')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Resume training with existed checkpoint.')
    parser.add_argument('--eval_data_url', type=str, default=None, help='Eval dataset path.')
    parser.add_argument('--eval_data_path', type=str, default=None, help='Eval dataset path.')
    parser.add_argument('--eval_interval', type=int, default=10, help='Evaluate frequency.')
    # Platform parameter
    parser.add_argument('--modelarts', type=ast.literal_eval, default=False, help='Run mode.')
    parser.add_argument('--run_distribute', type=ast.literal_eval, default=False, help='Run distribute.')
    parser.add_argument('--device_target', type=str, default='Ascend', choices=('Ascend', 'GPU'), help='Run platform.')
    # Train parameter
    parser.add_argument('--begin_epoch', type=int, default=0, help='Begin epoch.')
    parser.add_argument('--end_epoch', type=int, default=120, help='End epoch.')
    parser.add_argument('--total_epoch', type=int, default=120, help='Total epochs.')
    parser.add_argument('--batchsize', type=int, default=16, help='Batch size.')
    parser.add_argument('--optimizer', type=str, default='rmsprop', help='Chosen optimizer.')
    parser.add_argument('--lr', type=float, default=0.01, help='Base lr.')
    parser.add_argument('--lr_init', type=float, default=0.0001, help='Init lr.')
    parser.add_argument('--lr_end', type=float, default=0.00001, help='End lr.')
    parser.add_argument('--lr_scheme', type=str, default='linear', help='Lr scheduler strategy.')
    parser.add_argument('--sink_mode', action='store_false', help='Disable data sink mode.')
    args_opt = parser.parse_args()

    return args_opt


def set_context(cfg):
    """
    Set process context.

    Args:
        cfg: Config parameters.

    Returns:
        dev_id (int): Current process device id.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target=cfg.device_target)

    if cfg.run_distribute:
        if cfg.device_target == "Ascend":
            backend_name = None
        elif cfg.device_target == "GPU":
            backend_name = 'nccl'
        else:
            raise ValueError("Unsupported platform.")

        init(backend_name)
        dev_num = int(os.getenv("RANK_SIZE"))
        context.set_auto_parallel_context(
            device_num=dev_num,
            parallel_mode=ParallelMode.DATA_PARALLEL,
            gradients_mean=True,
        )
        dev_id = int(os.environ.get("DEVICE_ID", get_rank()))
    else:
        dev_id = 0

    return dev_id


def init_callbacks(cfg, dev_id, batch_number, learning_rate, network):
    """
    Initialize training callbacks.

    Args:
        cfg: Config parameters.
        dev_id: Current process device id.
        batch_number: Number of batches into one epoch on one device.
        learning_rate: Learning rate schedule.
        network: Network to be save into checkpoint.

    Returns:
        cbs: Inited callbacks.
    """
    cb = [TimeLossMonitor(lr_init=learning_rate)]

    # Save-checkpoint callback from one device
    if dev_id == 0 and cfg.save_ckpt:
        ckpt_config = CheckpointConfig(
            save_checkpoint_steps=batch_number*cfg.save_checkpoint_epochs,
            keep_checkpoint_max=cfg.keep_checkpoint_max,
        )
        ckpt_cb = ModelCheckpoint(
            prefix=f"hrnetw48-{cfg.dataset}",
            directory=cfg.train_path,
            config=ckpt_config,
        )
        cb.append(ckpt_cb)

    # Define evaluation callbacks during training
    if cfg.eval_data_path:
        eval_data = create_imagenet(
            dataset_path=cfg.eval_data_path,
            do_train=False,
            repeat_num=1,
            input_size=cfg.input_size,
            batch_size=cfg.batchsize,
            target=cfg.device_target,
            distribute=False,
        )

        eval_cb = EvalCallBack(network, eval_data, cfg.eval_interval)
        cb.append(eval_cb)

    return cb


@moxing_wrapper(config)
def main():
    """Main function."""
    device_id = set_context(config)

    print("Generating {}...".format(config.model.name))
    dataset = create_imagenet(
        dataset_path=config.data_path,
        do_train=True,
        repeat_num=1,
        input_size=config.input_size,
        batch_size=config.batchsize,
        target=config.device_target,
        distribute=config.run_distribute,
    )

    step_size = dataset.get_dataset_size()

    net = get_cls_model(config)

    if config.checkpoint_path:
        params = load_checkpoint(config.checkpoint_path)
        load_param_into_net(net, params)
    else:
        params_initializer(config, net)
    net.to_float(mstype.float16)
    net.set_train(True)

    # Define loss
    if not config.use_label_smooth:
        config.label_smooth_factor = 0.0
    loss = CrossEntropySmooth(smooth_factor=config.label_smooth_factor, num_classes=config.num_classes)

    # Get learning rate
    loss_scale = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
    lr = get_linear_lr(
        args.lr,
        config.total_epoch,
        step_size,
        args.lr_init,
        args.lr_end,
        config.warmup_epochs,
    )
    lr = lr[config.begin_epoch * step_size: config.end_epoch * step_size]

    # Init optimizer
    optimizer = nn.RMSProp(
        net.trainable_params(),
        learning_rate=lr,
        decay=0.9,
        weight_decay=config.wd,
        momentum=config.opt_momentum,
        epsilon=config.eps,
        loss_scale=config.loss_scale,
    )

    # Define model
    metrics = {
        'Top1-Acc': nn.Top1CategoricalAccuracy(),
        'Top5-Acc': nn.Top5CategoricalAccuracy()
    }
    model = Model(
        network=net,
        loss_fn=loss,
        optimizer=optimizer,
        loss_scale_manager=loss_scale,
        metrics=metrics,
        amp_level='O3',
    )

    cbs = init_callbacks(config, device_id, step_size, lr, model)

    # Begin train
    epoch_size = config.end_epoch - config.begin_epoch
    model.train(epoch_size, dataset, callbacks=cbs, dataset_sink_mode=config.sink_mode)
    last_checkpoint = os.path.join(config.train_path, f"hrnetw48cls-{config.dataset}-final.ckpt")
    save_checkpoint(net, last_checkpoint)


if __name__ == '__main__':
    args = parse_args()
    organize_configuration(config, args=args)
    main()
