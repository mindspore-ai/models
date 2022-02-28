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
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import initializer, XavierUniform
import mindspore.nn.probability.distribution as msd

class Encoder(nn.Cell):
    def __init__(self, batchsize, way, input_size, latent_size, droupout_r):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.dropout_r = droupout_r

        self.batchsize = batchsize
        self.way = way

        weight_initializer_enc = initializer(
            XavierUniform(),
            shape=[self.latent_size, self.input_size],
            dtype=mindspore.float32
            )
        weight_initializer_rn = initializer(
            XavierUniform(),
            shape=[2*self.latent_size, 2*self.latent_size],
            dtype=mindspore.float32)

        self.dropout = nn.Dropout(self.dropout_r)
        self.encoder = self.encoder = nn.Dense(self.input_size,
                                               self.latent_size,
                                               weight_init=weight_initializer_enc,
                                               has_bias=False)
        self.relation_network = nn.SequentialCell(
            nn.Dense(2 * self.latent_size, 2 * self.latent_size, weight_init=weight_initializer_rn, has_bias=False),
            nn.ReLU(),
            nn.Dense(2 * self.latent_size, 2 * self.latent_size, weight_init=weight_initializer_rn, has_bias=False),
            nn.ReLU(),
            nn.Dense(2 * self.latent_size, 2 * self.latent_size, weight_init=weight_initializer_rn, has_bias=False),
            nn.ReLU()
        )

        self.get_shape = ops.Shape()
        self.reshape = ops.Reshape()
        self.tile = ops.Tile()
        self.concat3 = ops.Concat(3)
        self.reduce_mean = ops.ReduceMean()
        self.split2_2 = ops.Split(2, 2)
        self.std_guassian = msd.Normal(0.0, 1.0, dtype=mindspore.float32)

    def sample(self, weights):
        mean, var = self.split2_2(weights)
        z = self.std_guassian.sample(self.get_shape(mean))
        return mean + var * z

    def construct(self, inputs, train=True):
        # inputs->[batch, N, K, embedsize]
        if train:
            inputs = self.dropout(inputs)
        encoder_outputs = self.encoder(inputs)
        b_size, N, K, latentsize = self.get_shape(encoder_outputs)

        # construct input for relation network
        t1 = self.reshape(encoder_outputs, (b_size, N * K, latentsize))
        t1 = t1.repeat(N * K, 1)
        t1 = self.reshape(t1, (b_size, N * N, K * K, latentsize))
        t2 = self.tile(encoder_outputs, (1, N, K, 1))
        x = self.concat3((t1, t2))

        # x->[batch, N*N, K*K, latensize]
        x = self.relation_network(x)
        x = self.reshape(x, (b_size, N, N * K * K, 2 * latentsize))
        x = self.reduce_mean(x, 2)

        latens = self.sample(x)
        mean, var = self.split2_2(x)
        return latens, mean, var


class Decoder(nn.Cell):
    def __init__(self, batchsize, way, input_size, latent_size):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.batchsize = batchsize
        self.way = way

        weight_initializer = initializer(
            XavierUniform(),
            shape=[2*self.input_size, self.latent_size,],
            dtype=mindspore.float32
            )
        self.decoder = nn.Dense(self.latent_size, 2 * self.input_size, weight_init=weight_initializer, has_bias=False)

        self.get_shape = ops.Shape()
        self.std_guassian = msd.Normal(0.0, 1.0, dtype=mindspore.float32)
        self.split2_2 = ops.Split(2, 2)

    def construct(self, x):
        weights = self.decoder(x)
        mean, var = self.split2_2(weights)
        z = self.std_guassian.sample(self.get_shape(mean))
        classfer_weights = mean + var * z
        return classfer_weights


class WithLossDecoder(nn.Cell):
    def __init__(self, decoder, classifier):
        super(WithLossDecoder, self).__init__()
        self.decoder = decoder
        self.classifier = classifier

    def construct(self, latents, inputs, target):
        weights = self.decoder(latents)
        loss = self.classifier(inputs, weights, target)
        return loss


class Classifier(nn.Cell):
    def __init__(self, way, input_size):
        super(Classifier, self).__init__()
        self.way = way
        self.input_size = input_size

        self.get_shape = ops.Shape()
        self.reshape = ops.Reshape()
        self.softmax = nn.Softmax(axis=-1)
        self.log = ops.Log()
        self.bmm = ops.BatchMatMul(transpose_b=True)
        self.squeeze1 = ops.Squeeze(axis=1)

    def construct(self, inputs, weights, target):
        criterion = NLLLoss(sparse=True)

        outputs = self.predict(weights, inputs)

        target = self.reshape(target, (self.get_shape(target)[0], -1, self.get_shape(target)[-1]))
        target = self.reshape(target, (-1, self.get_shape(target)[-1]))
        target = self.squeeze1(target)
        loss = criterion(outputs, target)
        return loss

    def predict(self, weights, inputs):
        b_size, _, _, input_size = self.get_shape(inputs)

        inputs = self.reshape(inputs, (b_size, -1, input_size))

        # predict
        outputs = self.bmm(inputs, weights)
        outputs = self.reshape(outputs, (-1, self.get_shape(outputs)[-1]))
        outputs = self.softmax(outputs)
        outputs = self.log(outputs)

        return outputs


class LEO(nn.Cell):
    def __init__(self, batchsize, input_size, latent_size, way, shot, dropout_r, inner_lr_init, finetuning_lr_init):
        super(LEO, self).__init__()

        self.batchsize = batchsize
        self.input_size = input_size
        self.latent_size = latent_size
        self.way = way
        self.shot = shot
        self.dropout_r = dropout_r
        self.inner_lr_init = inner_lr_init
        self.finetuning_lr_init = finetuning_lr_init

        self.inner_lr = mindspore.Parameter(mindspore.Tensor([self.inner_lr_init], dtype=mindspore.float32),
                                            name="inner_lr")
        self.finetuning_lr = mindspore.Parameter(mindspore.Tensor([self.finetuning_lr_init], dtype=mindspore.float32),
                                                 name="finetuning_lr")
        self.encoder = Encoder(self.batchsize, self.way, self.input_size, self.latent_size, self.dropout_r)
        self.decoder = Decoder(self.batchsize, self.way, self.input_size, self.latent_size)
        self.classifier = Classifier(self.way, self.input_size)
        self.withlossdecoder = WithLossDecoder(self.decoder, self.classifier)

        self.get_shape = ops.Shape()
        self.reshape = ops.Reshape()

    def construct(self, inputs, target, train):
        latens, _, _ = self.encoder(inputs, train)
        weights = self.decoder(latens)

        b_size, _, _, input_size = self.get_shape(inputs)

        inputs = self.reshape(inputs, (b_size, -1, input_size))

        # predict
        outputs = self.classifier.predict(inputs, weights)
        return outputs

class NLLLoss(nn.Cell):
    def __init__(self, sparse=False):
        super(NLLLoss, self).__init__()
        self.exp = ops.Exp()
        self.sum = ops.ReduceSum(keep_dims=True)
        self.onehot = ops.OneHot()
        self.on_value = mindspore.Tensor(1.0, mindspore.float32)
        self.off_value = mindspore.Tensor(0.0, mindspore.float32)
        self.div = ops.RealDiv()
        self.log = ops.Log()
        self.sum_cross_entropy = ops.ReduceSum(keep_dims=False)
        self.mul = ops.Mul()
        self.mul2 = ops.Mul()
        self.mean = ops.ReduceMean(keep_dims=False)
        self.sparse = sparse
        self.max = ops.ReduceMax(keep_dims=True)
        self.sub = ops.Sub()
        self.eps = mindspore.Tensor(1e-24, mindspore.float32)

    def construct(self, logit, label):
        if self.sparse:
            label = self.onehot(label, ops.shape(logit)[1], self.on_value, self.off_value)

        loss = self.sum_cross_entropy((self.mul(logit, label)), -1)
        loss = self.mul2(ops.scalar_to_array(-1.0), loss)
        loss = self.mean(loss, -1)

        return loss
