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

import numpy as np
import mindspore
import mindspore.numpy as mnp
from mindspore import nn

from src.dataloader import FEATURE_CHUNK_SIZE
from src.lossfuncs import SparseCategoricalCrossentropy
from src.mdense import MDense
from src.rnns import GRU


def pcm_init(shape, gain=.1, seed=None):
    """ Embedding table initializer """
    num_rows = 1
    for dim in shape[:-1]:
        num_rows *= dim
    num_cols = shape[-1]
    flat_shape = (num_rows, num_cols)
    if seed is not None:
        np.random.seed(seed)
    a = np.random.uniform(-1.7321, 1.7321, flat_shape)

    a = a + np.reshape(np.sqrt(12) * np.arange(-.5 * num_rows + .5, .5 * num_rows - .4) / num_rows, (num_rows, 1))
    a = gain * a.astype("float32")
    return mindspore.Tensor(a)


def interleave(p, samples):
    """ Interleaving of probability """
    p2 = mnp.expand_dims(p, 3)
    nb_repeats = 256 // (2 * p.shape[2])
    p3 = mnp.reshape(mnp.repeat(mnp.concatenate([1 - p2, p2], 3), nb_repeats), (-1, samples, 256))
    return p3


def tree_to_pdf(p, samples):
    """ Transforms tree probability representation to pdf """
    return interleave(p[:, :, 1:2], samples) * \
           interleave(p[:, :, 2:4], samples) * \
           interleave(p[:, :, 4:8], samples) * \
           interleave(p[:, :, 8:16], samples) * \
           interleave(p[:, :, 16:32], samples) * \
           interleave(p[:, :, 32:64], samples) * \
           interleave(p[:, :, 64:128], samples) * \
           interleave(p[:, :, 128:256], samples)


def tree_to_pdf_train(p):
    return tree_to_pdf(p, 160 * FEATURE_CHUNK_SIZE)


def tree_to_pdf_infer(p):
    return tree_to_pdf(p, 1)


class Encoder(nn.Cell):
    """ Implementation of frame rate network """
    def __init__(self, nb_used_features=20, training=False):
        super().__init__()
        self.embedding = nn.Embedding(256, 64, embedding_table='uniform')

        # NOTE: training mode leads to output shorter than input
        pad_mode = 'valid' if training else 'same'
        self.conv1 = nn.Conv1d(64 + nb_used_features, 128, 3, pad_mode=pad_mode, weight_init='xavier_uniform',
                               has_bias=True)
        self.conv2 = nn.Conv1d(128, 128, 3, pad_mode=pad_mode, weight_init='xavier_uniform', has_bias=True)

        self.dense1 = nn.Dense(128, 128, weight_init='xavier_uniform')
        self.dense2 = nn.Dense(128, 128, weight_init='xavier_uniform')

        self.tanh = nn.Tanh()

    def construct(self, feat, pitch):
        batch_size = pitch.shape[0]
        pitch = self.embedding(pitch)
        pitch = pitch.reshape((batch_size, -1, 64))

        cfeat = mnp.concatenate([feat, pitch], axis=-1)
        cfeat = mnp.transpose(cfeat, axes=(0, 2, 1))

        cfeat = self.tanh(self.conv1(cfeat))
        cfeat = self.tanh(self.conv2(cfeat))

        cfeat = mnp.transpose(cfeat, axes=(0, 2, 1))
        cfeat = self.tanh(self.dense1(cfeat))
        cfeat = self.tanh(self.dense2(cfeat))

        return cfeat


class Decoder(nn.Cell):
    """ Implementation of sample rate network """
    def __init__(self, rnn_units1=384, rnn_units2=16,
                 adaptation=False, quantize=False, flag_e2e=False):
        super().__init__()
        if adaptation or quantize or flag_e2e:
            raise NotImplementedError
        self.embed_size = 128
        self.rnn_units1 = rnn_units1
        self.rnn_units2 = rnn_units2

        self.embedding = nn.Embedding(256, self.embed_size, embedding_table=pcm_init(
            (256, self.embed_size)))
        self.rnn = GRU(input_size=3 * self.embed_size + 128, hidden_size=rnn_units1, batch_first=True)
        self.rnn2 = GRU(input_size=rnn_units1 + 128, hidden_size=rnn_units2, batch_first=True)
        self.md = MDense(rnn_units2, 2 ** 8, activation=nn.Sigmoid)

        self.sigmoid = nn.Sigmoid()

    def construct(self, pcm, cfeat, rnn1_state=None, rnn2_state=None, noise=None):
        batch_size = pcm.shape[0]

        cpcm = self.embedding(pcm)

        cpcm = cpcm.reshape(batch_size, -1, 3 * self.embed_size)
        if self.training:
            cfeat = cfeat.repeat(160, 1)

        rnn_in = mnp.concatenate([cpcm, cfeat], axis=-1)

        if rnn1_state is None:
            rnn1_state = mnp.zeros((1, batch_size, self.rnn_units1), dtype=mindspore.float16)

        rnn_in = rnn_in.astype(mindspore.float16)
        gru1_out, rnn1_state = self.rnn(rnn_in, rnn1_state)

        gru1_out = gru1_out.astype(mindspore.float32)
        if noise is not None:
            gru1_out = gru1_out + noise

        rnn_in = mnp.concatenate([gru1_out, cfeat], axis=-1)

        if rnn2_state is None:
            rnn2_state = mnp.zeros((1, batch_size, self.rnn_units2), dtype=mindspore.float16)

        rnn_in = rnn_in.astype(mindspore.float16)
        gru2_out, rnn2_state = self.rnn2(rnn_in, rnn2_state)
        gru2_out = gru2_out.astype(mindspore.float32)


        tree_prob = self.md(gru2_out)
        if self.training:
            ulaw_prob = tree_to_pdf_train(tree_prob)
        else:
            ulaw_prob = tree_to_pdf_infer(tree_prob)
        return ulaw_prob.astype(mindspore.float32), rnn1_state.astype(mindspore.float32), \
               rnn2_state.astype(mindspore.float32)


class LPCNet(nn.Cell):
    """ Implementation of LPCNet """
    def __init__(self, nb_used_features=20, training=False, rnn_units1=384, rnn_units2=16,
                 adaptation=False, quantize=False, flag_e2e=False):
        super().__init__()

        self.encoder = Encoder(nb_used_features=nb_used_features, training=training)
        self.decoder = Decoder(rnn_units1=rnn_units1, rnn_units2=rnn_units2,
                               adaptation=adaptation, quantize=quantize, flag_e2e=flag_e2e)

    def construct(self, pcm, feat, pitch, noise):
        cfeat = self.encoder(feat, pitch)
        ulaw_prob, _, _ = self.decoder(pcm, cfeat, noise=noise)

        return ulaw_prob


class WithLossLPCNet(nn.Cell):
    """ Implementation of LPCNet with loss function """
    def __init__(self, nb_used_features=20, training=False, rnn_units1=384, rnn_units2=16,
                 adaptation=False, quantize=False, flag_e2e=False, loss_fn=SparseCategoricalCrossentropy()):
        super().__init__()
        self.rnn_units1 = rnn_units1
        self.rnn_units2 = rnn_units2

        self.backbone = LPCNet(nb_used_features=nb_used_features, training=training, rnn_units1=rnn_units1,
                               rnn_units2=rnn_units2, adaptation=adaptation, quantize=quantize, flag_e2e=flag_e2e)

        self.loss_fn = loss_fn

    @property
    def backbone_network(self):
        return self.backbone

    def construct(self, in_data, features, periods, out_data, noise=None):
        probs = self.backbone(in_data, features, periods, noise)
        loss = self.loss_fn(probs, out_data)

        return loss
