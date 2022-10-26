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
import time
import mindspore
import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore import DynamicLossScaleManager, Model
import mindspore.context as context
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_group_size, get_rank
from mindspore.train.callback import TimeMonitor

from src.dinknet import DinkNet34, DinkNet50
from src.data import ImageFolderGenerator
from src.loss import dice_bce_loss
from src.model_utils.config import config
from src.callback import MyCallback


def create_dataset(_dataset_generator, _device_num, _rank_id, _batch_size):
    """
    when doing distributed training, dataset.GeneratorDataset need to set num_shards and shard_id
    see:
    https://mindspore.cn/docs/zh-CN/master/api_python/dataset/mindspore.dataset.GeneratorDataset.html#mindspore.dataset.GeneratorDataset
    https://blog.csdn.net/weixin_43431343/article/details/121895510
    """
    if device_num == 1:
        _dataset = ds.GeneratorDataset(_dataset_generator,
                                       ["img", "mask"],
                                       shuffle=True,
                                       num_parallel_workers=4,
                                       )
    else:
        _dataset = ds.GeneratorDataset(_dataset_generator,
                                       ["img", "mask"],
                                       shuffle=True,
                                       num_parallel_workers=_device_num,
                                       num_shards=_device_num,
                                       shard_id=_rank_id
                                       )

    _dataset = _dataset.batch(_batch_size)  # set batch size
    return _dataset


if __name__ == "__main__":
    print(config)

    batch_size = config.batch_size
    learning_rate = config.learning_rate
    time_start = time.time()
    if config.device_target not in ['Ascend', 'GPU']:
        raise Exception("Only support on Ascend or GPU currently.")

    # set context
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=config.device_target,
                        )

    epoch_num = config.epoch_num
    if config.run_distribute == "True":
        init()
        device_num = get_group_size()
        epoch_num = config.distribute_epoch_num
        print("group_size(device_num) is: ", device_num)
        rank_id = get_rank()
        print("rank_id is: ", rank_id)
        # set auto parallel context
        context.set_auto_parallel_context(device_num=device_num,
                                          gradients_mean=True,
                                          parallel_mode=ParallelMode.DATA_PARALLEL
                                          )
        if config.device_target == 'Ascend':
            context.set_context(device_id=int(os.environ["DEVICE_ID"]))
    else:
        device_num = 1
        rank_id = 0
        context.set_context(device_id=int(os.environ["DEVICE_ID"]))
    mindspore.common.set_seed(2022)
    # mox copy parallel
    if config.enable_modelarts:
        import moxing as mox
        local_data_url = "/cache/dataset/train"
        mox.file.copy_parallel(config.data_url, local_data_url)
        pretrained_ckpt_path = "/cache/origin_weights/pretrained_model.ckpt"
        mox.file.copy_parallel(config.pretrained_ckpt, pretrained_ckpt_path)
        mox.file.make_dirs('../../../train_out/weights')
        print('path[/cache/train_out/weights] exist:', mox.file.exists('../../../train_out/weights'))
        mox.file.make_dirs('../../../train_out/logs')
        print('path[/cache/train_out/logs] exist:', mox.file.exists('../../../train_out/logs'))

    else:
        local_data_url = config.data_path
        pretrained_ckpt_path = config.pretrained_ckpt

    # prepare weight file and log file
    log_name = config.log_name
    rank_label = '[' + str(rank_id) + ']'
    if config.enable_modelarts:
        file_name = os.path.join("../../../train_out/weights", str(log_name) + "_rank" + str(rank_id) + ".ckpt")
        log = open(os.path.join("../../../train_out/logs", str(log_name) + "_rank" + str(rank_id) + ".log"), 'w')
    else:
        file_name = os.path.join(config.output_path, str(log_name) + "_rank" + str(rank_id) + ".ckpt")
        log = open(os.path.join(config.output_path, str(log_name) + "_rank" + str(rank_id) + ".log"), 'w')

    # prepare for dataset
    image_list = filter(lambda x: x.find('sat') != -1, os.listdir(local_data_url))
    train_list = list(map(lambda x: x[:-8], image_list))
    dataset_generator = ImageFolderGenerator(train_list, local_data_url)

    dataset = create_dataset(dataset_generator, device_num, rank_id, batch_size)

    # define network
    if config.model_name == 'dinknet34':
        network = DinkNet34(use_backbone=True)
    else:
        network = DinkNet50(use_backbone=True)

    # define optimizer
    optimizer = nn.Adam(params=network.trainable_params(), learning_rate=learning_rate)

    # define loss
    loss = dice_bce_loss()

    dataset_sink_mode = True
    if config.device_target == "Ascend":
        dataset_sink_mode = False

    # define loss scale
    init_loss_scale = config.init_loss_scale
    scale_factor = config.scale_factor
    scale_window = config.scale_window
    loss_scale_manager = DynamicLossScaleManager(
        init_loss_scale=init_loss_scale,
        scale_factor=scale_factor,
        scale_window=scale_window
    )
    # callback
    myCallback = MyCallback(log, file_name, rank_label, device_num, show_step=dataset_sink_mode,
                            learning_rate=learning_rate, model_name=config.model_name)

    # define model
    model = Model(network, loss, optimizer, loss_scale_manager=loss_scale_manager)
    # train
    model.train(epoch_num, dataset, callbacks=[TimeMonitor(), myCallback], dataset_sink_mode=dataset_sink_mode)

    if config.enable_modelarts:
        mox.file.copy_parallel('/cache/train_out', config.train_url)

    time_end = time.time()
    print('train_time: %f' % (time_end - time_start))
