# Copyright 2022 Huawei Technologies Co., Ltd

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

from mindspore import nn, Model, context
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor

from data_provider.mnist_to_mindrecord import create_mnist_dataset
from nets.predrnn_pp import PreRNN, NetWithLossCell
from config import config

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_mindrecord', type=str, default='')
    parser.add_argument('--device_id', type=int, default=0)
    args_opt = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', device_id=args_opt.device_id)
    device_num = config.device_num
    rank = 0

    num_hidden = [int(x) for x in config.num_hidden.split(',')]
    num_layers = len(num_hidden)

    shape = [config.batch_size,
             config.seq_length,
             config.patch_size*config.patch_size*config.img_channel,
             int(config.img_width/config.patch_size),
             int(config.img_width/config.patch_size)]

    shape = list(map(int, shape))

    network = PreRNN(input_shape=shape,
                     num_layers=num_layers,
                     num_hidden=num_hidden,
                     filter_size=config.filter_size,
                     stride=config.stride,
                     seq_length=config.seq_length,
                     input_length=config.input_length,
                     tln=config.layer_norm)

    netwithloss = NetWithLossCell(network, config.batch_size, config.seq_length, \
        config.input_length, shape[-3], shape[-1], config.reverse_input, True)
    exponential_decay_lr = nn.ExponentialDecayLR(config.lr, 0.95, 10000, is_stair=True)
    opt = nn.Adam(params=netwithloss.trainable_params(), learning_rate=config.lr)

    train_step = nn.TrainOneStepCell(netwithloss, opt).set_train()

    model = Model(train_step)

    ds = create_mnist_dataset(dataset_files=args_opt.train_mindrecord, rank_size=device_num, \
        rank_id=rank, do_shuffle=True, batch_size=config.batch_size)

    time_cb = TimeMonitor(data_size=ds.get_dataset_size())
    config_ck = CheckpointConfig(save_checkpoint_steps=config.snapshot_interval, keep_checkpoint_max=5)
    ckpt_cb = ModelCheckpoint(prefix=config.model_name, directory=config.save_dir, config=config_ck)

    model.train(epoch=int(config.max_iterations/config.sink_size), train_dataset=ds, \
        sink_size=config.sink_size, dataset_sink_mode=True, callbacks=[time_cb, ckpt_cb, LossMonitor()])
