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
"""Train mobilenetV2 on ImageNet."""

import os
import time
import random
import numpy as np

import mindspore as ms
import mindspore.communication as comm
import mindspore.nn as nn

from src.dataset import create_dataset, extract_features
from src.lr_generator import get_lr
from src.utils import config_ckpoint
from src.models import CrossEntropyWithLabelSmooth, define_net, load_ckpt, build_params_groups
from src.metric import DistAccuracy, ClassifyCorrectCell
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper, modelarts_process
from src.model_utils.device_adapter import get_device_id


ms.set_seed(1)


@moxing_wrapper(pre_process=modelarts_process)
def train_mobilenetv2():
    """ train_mobilenetv2 """
    if config.platform == "CPU":
        config.run_distribute = False
    ms.set_context(mode=ms.GRAPH_MODE, device_target=config.platform, save_graphs=False)
    if config.run_distribute:
        comm.init()
        config.rank_id = comm.get_rank()
        config.rank_size = comm.get_group_size()
        ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL,
                                     gradients_mean=True)
    config.train_dataset_path = os.path.join(config.dataset_path, 'train')
    config.eval_dataset_path = os.path.join(config.dataset_path, 'validation_preprocess')
    if not config.device_id:
        config.device_id = get_device_id()
    start = time.time()
    # set context and device init
    print('\nconfig: {} \n'.format(config))
    # define network
    backbone_net, head_net, net = define_net(config, config.is_training)
    dataset = create_dataset(dataset_path=config.train_dataset_path, do_train=True, config=config,
                             enable_cache=config.enable_cache, cache_session_id=config.cache_session_id)
    step_size = dataset.get_dataset_size()
    if config.platform == "GPU":
        ms.set_context(enable_graph_kernel=True)
    if config.pretrain_ckpt:
        if config.freeze_layer == "backbone":
            load_ckpt(backbone_net, config.pretrain_ckpt, trainable=False)
            step_size = extract_features(backbone_net, config.train_dataset_path, config)
        elif config.filter_head:
            load_ckpt(backbone_net, config.pretrain_ckpt)
        else:
            load_ckpt(net, config.pretrain_ckpt)
    if step_size == 0:
        raise ValueError("The step_size of dataset is zero. Check if the images' count of train dataset is more \
            than batch_size in config.py")

    # define loss
    if config.label_smooth > 0:
        loss = CrossEntropyWithLabelSmooth(
            smooth_factor=config.label_smooth, num_classes=config.num_classes)
    else:
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    epoch_size = config.epoch_size

    # get learning rate
    lr = ms.Tensor(get_lr(global_step=0,
                          lr_init=config.lr_init,
                          lr_end=config.lr_end,
                          lr_max=config.lr_max,
                          warmup_epochs=config.warmup_epochs,
                          total_epochs=epoch_size,
                          steps_per_epoch=step_size))
    metrics = {"acc"}
    dist_eval_network = None
    eval_dataset = None
    if config.run_eval:
        metrics = {'acc': DistAccuracy(batch_size=config.batch_size, device_num=config.rank_size)}
        dist_eval_network = ClassifyCorrectCell(net, config.run_distribute)
        eval_dataset = create_dataset(dataset_path=config.eval_dataset_path, do_train=False, config=config)
    if config.pretrain_ckpt == "" or config.freeze_layer != "backbone":
        if config.platform == "Ascend":
            loss_scale = ms.FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
            group_params = build_params_groups(net)
            opt = nn.Momentum(group_params, lr, config.momentum, loss_scale=config.loss_scale)
            model = ms.Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale,
                             metrics=metrics, eval_network=dist_eval_network,
                             amp_level="O2", keep_batchnorm_fp32=False,
                             boost_level=config.boost_mode,
                             boost_config_dict={"boost": {"mode": "manual", "grad_freeze": False}})

        else:
            opt = nn.Momentum(net.trainable_params(), lr, config.momentum, config.weight_decay)
            model = ms.Model(net, loss_fn=loss, optimizer=opt, metrics=metrics, eval_network=dist_eval_network,
                             boost_level=config.boost_mode)
        cb = config_ckpoint(config, lr, step_size, model, eval_dataset)
        print("============== Starting Training ==============")
        model.train(epoch_size, dataset, callbacks=cb, dataset_sink_mode=True)
        print("============== End Training ==============")

    else:
        opt = nn.Momentum(filter(lambda x: x.requires_grad, head_net.get_parameters()),
                          lr, config.momentum, config.weight_decay)

        network = nn.WithLossCell(head_net, loss)
        network = nn.TrainOneStepCell(network, opt)
        network.set_train()

        features_path = config.train_dataset_path + '_features'
        idx_list = list(range(step_size))
        rank = config.rank_id
        save_ckpt_path = os.path.join(config.save_checkpoint_path, 'ckpt_' + str(rank) + '/')
        if not os.path.isdir(save_ckpt_path):
            os.mkdir(save_ckpt_path)

        for epoch in range(epoch_size):
            random.shuffle(idx_list)
            epoch_start = time.time()
            losses = []
            for j in idx_list:
                feature = ms.Tensor(np.load(os.path.join(features_path, "feature_{}.npy".format(j))))
                label = ms.Tensor(np.load(os.path.join(features_path, "label_{}.npy".format(j))))
                losses.append(network(feature, label).asnumpy())
            epoch_mseconds = (time.time()-epoch_start) * 1000
            per_step_mseconds = epoch_mseconds / step_size
            print("epoch[{}/{}], iter[{}] cost: {:5.3f}, per step time: {:5.3f}, avg loss: {:5.3f}"\
            .format(epoch + 1, epoch_size, step_size, epoch_mseconds, per_step_mseconds, np.mean(np.array(losses))))
            if (epoch + 1) % config.save_checkpoint_epochs == 0:
                ms.save_checkpoint(net, os.path.join(save_ckpt_path, "mobilenetv2_{}.ckpt".format(epoch + 1)))
        print("total cost {:5.4f} s".format(time.time() - start))

    if config.enable_cache:
        print("Remember to shut down the cache server via \"cache_admin --stop\"")


if __name__ == '__main__':
    train_mobilenetv2()
