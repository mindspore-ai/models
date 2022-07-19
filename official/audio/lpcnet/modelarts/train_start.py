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
from argparse import ArgumentParser
from pathlib import Path

import mindspore
import mindspore.dataset as ds
import mindspore.numpy as np
from mindspore import context, export, load_checkpoint
from mindspore import Model, nn, ops
from mindspore.communication import get_group_size, init
from mindspore.context import ParallelMode
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
                density = self.grua_density[k]
                if self.batch < self.t_end:
                    r = 1 - (self.batch - self.t_start).astype(mindspore.float32) / (self.t_end - self.t_start)
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

def _export(args_):
    NB_USED_FEATURES = 20

    # NOTE: fails without max_call_depth due to RNN
    context.set_context(mode=context.GRAPH_MODE, device_target=args_.device_target, max_call_depth=30000)
    model = lpcnet.WithLossLPCNet()
    model.backbone.to_float(mindspore.float16)
    ckpt_name = args_.ckpt_path + "lpcnet-" + str(args_.epochs) + "_37721.ckpt"
    load_checkpoint(ckpt_name, net=model)
    model.set_train(False)
    enc = model.backbone.encoder
    dec = model.backbone.decoder
    enc_input_feat = mindspore.Tensor(np.zeros([1, args_.max_len, NB_USED_FEATURES]), mindspore.float32)
    enc_input_pitch = mindspore.Tensor(np.zeros([1, args_.max_len, 1]), mindspore.int32)
    export(enc, enc_input_feat, enc_input_pitch, file_name=args_.ckpt_path + '_enc', file_format=args_.file_format)
    print("Encoder exported successfully")

    dec_input_pcm = mindspore.Tensor(np.zeros([1, 1, 3]), mindspore.int32)
    dec_input_cfeat = mindspore.Tensor(np.zeros([1, 1, 128]), mindspore.float32)
    dec_input_state1 = mindspore.Tensor(np.zeros([1, 1, model.rnn_units1]), mindspore.float32)
    dec_input_state2 = mindspore.Tensor(np.zeros([1, 1, model.rnn_units2]), mindspore.float32)
    dec_inputs = [dec_input_pcm, dec_input_cfeat, dec_input_state1, dec_input_state2]
    export(dec, *dec_inputs, file_name=args_.ckpt_path + '_dec', file_format='AIR')
    print("Decoder exported successfully")
    print("Total Parameters:", sum(p.asnumpy().size for p in model.trainable_params()))

if __name__ == "__main__":
    print("start here")
    parser = ArgumentParser()
    parser.add_argument('--feature_path', help='binary features file (float32)', type=Path)
    parser.add_argument('--data_path', help='binary audio data file (uint8)', type=Path)
    parser.add_argument('--ckpt_path', type=str, default='../ckpt/lpcnet-4_37721.ckpt',
                        help='path of checkpoint')
    parser.add_argument('--air_path', type=str, default='../ckpt/lpcnet-4_37721.ckpt',
                        help='path of air')
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
    parser.add_argument('--max_len', type=int, default=500, help='number of 10ms frames')
    # parser.add_argument('--out_file', '-n', type=str, default='lpcnet', help='name of model')
    parser.add_argument('--file_format', type=str, default='AIR', help='format of model')
    parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend'],
                        help='device where the code will be implemented')

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

    device_id = int(os.getenv('DEVICE_ID'))
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target,
                        max_call_depth=5000)  # NOTE: fails without max_call_depth due to RNN
    context.set_context(device_id=device_id)
    init()

    context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                      gradients_mean=False, parameter_broadcast=True)
    ds.config.set_prefetch_size(16)


    loader, total_batches = make_dataloader(args.data_path, args.feature_path, batch_size=args.batch_size,
                                            parallel=True)

    net_loss = SparseCategoricalCrossentropy(reduction='mean')
    net = lpcnet.WithLossLPCNet(rnn_units1=args.grua_size, rnn_units2=args.grub_size, training=True,
                                loss_fn=net_loss)

    net.backbone.to_float(mindspore.float16)

    net_opt = nn.Adam(list(net.trainable_params()), learning_rate=lr, weight_decay=decay,
                      beta2=0.99)

    rank_size = get_group_size()
    if retrain:
        t_start, t_end, intervlal = 0, 0, 1
    else:
        t_start, t_end, intervlal = 2000 // rank_size, 40000 // rank_size, 400 // rank_size

    config = CheckpointConfig(saved_network=net)
    ckpt_callback = ModelCheckpoint('lpcnet', directory=str(args.ckpt_path))
    train_net = MyTrainStep(net, net_opt, t_start, t_end, intervlal, grua_density=_density)
    train_net = Model(network=train_net)
    train_net.train(nb_epochs, loader, callbacks=[LossMonitor(1), TimeMonitor(1), ckpt_callback],
                    dataset_sink_mode=True)
    _export(args)
