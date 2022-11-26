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

from argparse import ArgumentParser
from pathlib import Path

import mindspore
import mindspore.dataset as ds
import mindspore.numpy as np
from mindspore import Model, context, nn, ops
from mindspore.train.callback import (CheckpointConfig, LossMonitor,
                                      ModelCheckpoint, TimeMonitor)

from src import lpcnet
from src.dataloader import make_dataloader
from src.lossfuncs import SparseCategoricalCrossentropy


class MyTrainStep(nn.TrainOneStepCell):
    def __init__(self, network, optimizer, sparcify_start, sparcify_end, sparcify_interval,
                 grua_density):
        super(MyTrainStep, self).__init__(network=network, optimizer=optimizer)
        self.batch = mindspore.Parameter(1, requires_grad=False)
        self.mask_inited = mindspore.Parameter(0, requires_grad=False)
        self.mask = mindspore.Parameter(np.ones((384 * 3, 384)), requires_grad=False)
        self.t_start = sparcify_start
        self.t_end = sparcify_end
        self.interval = sparcify_interval
        self.grua_density = grua_density
        self.cast = ops.Cast()

    def construct(self, in_data, features, periods, out_data, noise=None):
        loss = super(MyTrainStep, self).construct(in_data, features, periods, out_data, noise)

        self.batch += 1
        new_w = self.sparsify_a(self.network.backbone.decoder.rnn.w_hh_list[0])
        ops.Assign()(self.network.backbone.decoder.rnn.w_hh_list[0], new_w)

        return loss

    def sparsify_a(self, p):
        if ((self.batch > self.t_start and (self.batch - self.t_start) % self.interval == 0) and
                self.batch < self.t_end):
            nb = p.shape[0] // p.shape[1]
            N = p.shape[1]
            total_mask = []
            for k in range(nb):
                density = self.cast(self.grua_density[k], mindspore.float32)
                if self.batch < self.t_end:
                    r = 1 - (self.batch - self.t_start).astype(mindspore.float32) / (self.t_end - self.t_start)
                    r = self.cast(r, mindspore.float32)
                    density = 1 - (1 - density) * (1 - r * r * r)
                A = p[k * N:(k + 1) * N, :]
                A = A - np.diag(np.diag(A))
                L = np.reshape(A, (N // 4, 4, N // 8, 8))
                S = np.sum(L * L, axis=-1)
                S = np.sum(S, axis=1)
                SS, _ = ops.Sort()(np.reshape(S, (-1,)))
                idx = N * N // 32 * (1. - density) * np.ones((1,))
                idx = ops.Round()(idx)
                idx = idx.astype(mindspore.int32)
                thresh = SS[idx]
                mask = (S >= thresh).astype(mindspore.float32)
                mask = np.repeat(mask, 4, axis=0)
                mask = np.repeat(mask, 8, axis=1)
                mask = np.minimum(1, mask + np.diag(np.ones((N,))))

                total_mask.append(mask)
            self.mask = np.concatenate(total_mask, axis=0)
        p = p * self.mask
        return p

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('features', help='binary features file (float32)', type=Path)
    parser.add_argument('data', help='binary audio data file (uint8)', type=Path)
    parser.add_argument('output', help='path to directory where .ckpt will be stored', type=Path)
    parser.add_argument('--retrain', default=None, help='continue training model')
    parser.add_argument('--density-split', nargs=3, type=float, default=[0.05, 0.05, 0.2],
                        help='density of each recurrent gate (default 0.05, 0.05, 0.2)')
    parser.add_argument('--grua-size', default=384, type=int,
                        help='number of units in GRU A (default 384)')
    parser.add_argument('--grub-size', default=16, type=int,
                        help='number of units in GRU B (default 16)')
    parser.add_argument('--epochs', default=4, type=int,
                        help='number of epochs to train for (default 4)')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='batch size to use (default 64)')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--decay', default=2.5e-5, type=float, help='learning rate decay')
    parser.add_argument('--gamma', default=2.0, type=float,
                        help='adjust u-law compensation (default 2.0, should not be less than 1.0)')
    parser.add_argument('--device_target', type=str, default='Ascend',
                        choices=['Ascend'])

    args = parser.parse_args()

    _density = args.density_split
    gamma = args.gamma
    nb_epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    decay = args.decay
    retrain = args.retrain is not None
    if retrain:
        input_model = args.retrain

    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target,
                        max_call_depth=5000)  # NOTE: fails without max_call_depth due to RNN
    ds.config.set_prefetch_size(16)

    loader, total_batches = make_dataloader(args.data, args.features, batch_size=args.batch_size)

    net_loss = SparseCategoricalCrossentropy(reduction='mean')
    net = lpcnet.WithLossLPCNet(rnn_units1=args.grua_size, rnn_units2=args.grub_size, training=True,
                                loss_fn=net_loss)

    net.backbone.to_float(mindspore.float16)

    net_opt = nn.Adam(list(net.trainable_params()), learning_rate=lr, weight_decay=decay,
                      beta2=0.99)

    if retrain:
        t_start, t_end, intervlal = 0, 0, 1
    else:
        t_start, t_end, intervlal = 2000, 40000, 400

    config = CheckpointConfig(saved_network=net)
    ckpt_callback = ModelCheckpoint('lpcnet', directory=str(args.output))
    train_net = MyTrainStep(net, net_opt, t_start, t_end, intervlal, grua_density=_density)
    train_net = Model(network=train_net)
    train_net.train(nb_epochs, loader, callbacks=[LossMonitor(1), TimeMonitor(1), ckpt_callback],
                    dataset_sink_mode=True)
