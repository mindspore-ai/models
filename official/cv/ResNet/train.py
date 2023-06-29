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
"""train resnet."""
import os

import mindspore as ms
import mindspore.nn as nn
from mindspore.train.train_thor import ConvertModelUtils
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.communication.management import init, get_rank
from mindspore.parallel import set_algo_parameters

from src.logger import get_logger
from src.lr_generator import get_lr, warmup_cosine_annealing_lr
from src.CrossEntropySmooth import CrossEntropySmooth
from src.callback import LossCallBack, ResumeCallback
from src.util import eval_callback, init_weight, init_group_params, set_output_dir
from src.metric import DistAccuracy, ClassifyCorrectCell
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_num

ms.set_seed(1)

if config.net_name in ("resnet18", "resnet34", "resnet50", "resnet152"):
    if config.net_name == "resnet18":
        from src.resnet import resnet18 as resnet
    elif config.net_name == "resnet34":
        from src.resnet import resnet34 as resnet
    elif config.net_name == "resnet50":
        from src.resnet import resnet50 as resnet
    else:
        from src.resnet import resnet152 as resnet
    if config.dataset == "cifar10":
        from src.dataset import create_dataset1 as create_dataset
    else:
        from src.dataset import create_dataset2 as create_dataset
elif config.net_name == "resnet101":
    from src.resnet import resnet101 as resnet
    from src.dataset import create_dataset3 as create_dataset
else:
    from src.resnet import se_resnet50 as resnet
    from src.dataset import create_dataset4 as create_dataset


def set_graph_kernel_context(run_platform, net_name):
    if run_platform == "GPU" and net_name == "resnet101":
        ms.set_context(enable_graph_kernel=True)
        ms.set_context(graph_kernel_flags="--enable_parallel_fusion --enable_expand_ops=Conv2D")


def set_parameter():
    """set_parameter"""
    target = config.device_target
    if target == "CPU":
        config.run_distribute = False

    # init context
    if config.mode_name == 'GRAPH':
        if target == "Ascend":
            rank_save_graphs_path = os.path.join(config.save_graphs_path, "soma", str(os.getenv('DEVICE_ID', '0')))
            ms.set_context(mode=ms.GRAPH_MODE, device_target=target, save_graphs=config.save_graphs,
                           save_graphs_path=rank_save_graphs_path)
        else:
            ms.set_context(mode=ms.GRAPH_MODE, device_target=target, save_graphs=config.save_graphs)
        set_graph_kernel_context(target, config.net_name)
    else:
        ms.set_context(mode=ms.PYNATIVE_MODE, device_target=target, save_graphs=False)

    if config.parameter_server:
        ms.set_ps_context(enable_ps=True)
    if config.run_distribute:
        if target == "Ascend":
            device_id = int(os.getenv('DEVICE_ID', '0'))
            ms.set_context(device_id=device_id)
            ms.set_auto_parallel_context(device_num=config.device_num, parallel_mode=ms.ParallelMode.DATA_PARALLEL,
                                         gradients_mean=True)
            set_algo_parameters(elementwise_op_strategy_follow=True)
            if config.net_name == "resnet50" or config.net_name == "se-resnet50":
                if config.boost_mode not in ["O1", "O2"]:
                    ms.set_auto_parallel_context(all_reduce_fusion_config=config.all_reduce_fusion_config)
            elif config.net_name in ["resnet101", "resnet152"]:
                ms.set_auto_parallel_context(all_reduce_fusion_config=config.all_reduce_fusion_config)
            init()
        # GPU target
        else:
            init()
            ms.set_auto_parallel_context(device_num=get_device_num(),
                                         parallel_mode=ms.ParallelMode.DATA_PARALLEL,
                                         gradients_mean=True)
            if config.net_name == "resnet50":
                ms.set_auto_parallel_context(all_reduce_fusion_config=config.all_reduce_fusion_config)
    config.rank_id = get_rank() if config.run_distribute else 0


def init_lr(step_size):
    """init lr"""
    if config.optimizer == "Thor":
        from src.lr_generator import get_thor_lr
        lr = get_thor_lr(config.start_epoch * step_size, config.lr_init, config.lr_decay, config.lr_end_epoch,
                         step_size, decay_epochs=39)
    else:
        if config.net_name in ("resnet18", "resnet34", "resnet50", "resnet152", "se-resnet50"):
            config.lr_max = config.lr_max / 8 * config.device_num
            lr = get_lr(lr_init=config.lr_init, lr_end=config.lr_end, lr_max=config.lr_max,
                        warmup_epochs=config.warmup_epochs, total_epochs=config.epoch_size,
                        start_epoch=config.start_epoch, steps_per_epoch=step_size, lr_decay_mode=config.lr_decay_mode)
        else:
            lr = warmup_cosine_annealing_lr(config.lr, step_size, config.warmup_epochs, config.epoch_size,
                                            config.start_epoch * step_size)
    return lr


def init_loss_scale():
    if config.dataset == "imagenet2012":
        if not config.use_label_smooth:
            config.label_smooth_factor = 0.0
        loss = CrossEntropySmooth(sparse=True, reduction="mean",
                                  smooth_factor=config.label_smooth_factor, num_classes=config.class_num)
    else:
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    return loss


@moxing_wrapper()
def train_net():
    """train net"""
    target = config.device_target
    set_parameter()
    set_output_dir(config)
    config.logger = get_logger(config.log_dir, config.rank_id, config.parameter_server)
    dataset = create_dataset(dataset_path=config.data_path, do_train=True,
                             batch_size=config.batch_size, train_image_size=config.train_image_size,
                             eval_image_size=config.eval_image_size, target=target,
                             distribute=config.run_distribute)
    step_size = dataset.get_dataset_size()
    net = resnet(class_num=config.class_num)
    if config.parameter_server:
        net.set_param_ps()
    init_weight(net, config)

    if config.resume_ckpt:
        resume_param = ms.load_checkpoint(config.resume_ckpt,
                                          choice_func=lambda x: not x.startswith(('learning_rate', 'global_step')))
        config.start_epoch = int(resume_param.get('epoch_num', ms.Tensor(0, ms.int32)).asnumpy().item())

    lr = ms.Tensor(init_lr(step_size=step_size))
    # define opt
    group_params = init_group_params(net, config)
    opt = nn.Momentum(group_params, lr, config.momentum, loss_scale=config.loss_scale)
    if config.optimizer == "LARS":
        opt = nn.LARS(opt, epsilon=config.lars_epsilon, coefficient=config.lars_coefficient,
                      lars_filter=lambda x: 'beta' not in x.name and 'gamma' not in x.name and 'bias' not in x.name)
    loss = init_loss_scale()
    loss_scale = ms.FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
    dist_eval_network = ClassifyCorrectCell(net) if config.run_distribute else None
    metrics = {"acc"}
    if config.run_distribute:
        metrics = {'acc': DistAccuracy(batch_size=config.batch_size, device_num=config.device_num)}
    if (config.net_name not in ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "se-resnet50")) or \
            config.parameter_server or target == "CPU":
        # fp32 training
        model = ms.Model(net, loss_fn=loss, optimizer=opt, metrics=metrics, eval_network=dist_eval_network)
    else:
        model = ms.Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, metrics=metrics,
                         amp_level="O3", boost_level=config.boost_mode,
                         eval_network=dist_eval_network,
                         boost_config_dict={"grad_freeze": {"total_steps": config.epoch_size * step_size}})

    if config.optimizer == "Thor" and config.dataset == "imagenet2012":
        from src.lr_generator import get_thor_damping
        damping = get_thor_damping(step_size * config.start_epoch, config.damping_init, config.damping_decay, 70,
                                   step_size)
        split_indices = [26, 53]
        opt = nn.thor(net, lr, ms.Tensor(damping), config.momentum, config.weight_decay, config.loss_scale,
                      config.batch_size, split_indices=split_indices, frequency=config.frequency)
        model = ConvertModelUtils().convert_to_thor_model(model=model, network=net, loss_fn=loss, optimizer=opt,
                                                          loss_scale_manager=loss_scale, metrics={'acc'},
                                                          amp_level="O3")
        config.run_eval = False
        config.logger.warning("Thor optimizer not support evaluation while training.")

    # load resume param
    if config.resume_ckpt:
        ms.load_param_into_net(net, resume_param)
        ms.load_param_into_net(opt, resume_param)
        config.logger.info('resume train from epoch: %s', config.start_epoch)

    # define callbacks
    loss_cb = LossCallBack(config.epoch_size, config.logger, lr, per_print_time=10)
    resume_cb = ResumeCallback(config.start_epoch)
    cb = [loss_cb, resume_cb]
    if config.save_checkpoint and config.rank_id == 0:
        ckpt_append_info = [{"epoch_num": 0, "step_num": 0}]
        config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * step_size,
                                     keep_checkpoint_max=config.keep_checkpoint_max,
                                     append_info=ckpt_append_info)
        ckpt_cb = ModelCheckpoint(prefix=config.net_name, directory=config.save_ckpt_dir, config=config_ck)
        cb += [ckpt_cb]

    if config.run_eval:
        eval_dataset = create_dataset(dataset_path=config.eval_dataset_path, do_train=False,
                                      batch_size=config.batch_size, train_image_size=config.train_image_size,
                                      eval_image_size=config.eval_image_size,
                                      target=target, enable_cache=config.enable_cache,
                                      cache_session_id=config.cache_session_id)
        eval_cb = eval_callback(model, config, eval_dataset)
        cb.append(eval_cb)

    # train model
    if config.net_name == "se-resnet50":
        config.epoch_size = config.train_epoch_size
    dataset_sink_mode = (not config.parameter_server) and target != "CPU"
    config.logger.save_args(config)
    model.train(config.epoch_size - config.start_epoch, dataset, callbacks=cb,
                sink_size=dataset.get_dataset_size(), dataset_sink_mode=dataset_sink_mode)

    config.logger.info("If run eval and enable_cache Remember to shut down the cache server via \"cache_admin --stop\"")


if __name__ == '__main__':
    train_net()
