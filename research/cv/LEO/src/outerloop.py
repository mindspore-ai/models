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

import math

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as np
from mindspore.nn.loss.loss import MSELoss

from src.model import LEO


class OuterLoop(nn.Cell):
    def __init__(self, batchsize, input_size, latent_size, way, shot,
                 dropout, kl_weight, encoder_penalty_weight, orthogonality_weight,
                 inner_lr_init, finetuning_lr_init,
                 inner_step, finetune_inner_step, is_meta_training):
        super(OuterLoop, self).__init__()
        self.batchsize = batchsize
        self.input_size = input_size
        self.latent_size = latent_size
        self.way = way
        self.shot = shot
        self.dropout = dropout
        self.kl_weight = mindspore.Tensor(float(kl_weight), mindspore.float32)
        self.encoder_penalty_weight = mindspore.Tensor(float(encoder_penalty_weight), mindspore.float32)
        self.orthogonality_weight = mindspore.Tensor(float(orthogonality_weight), mindspore.float32)
        self.inner_lr_init = inner_lr_init
        self.finetuning_lr_init = finetuning_lr_init
        self.inner_step = inner_step
        self.finetune_inner_step = finetune_inner_step
        self.is_meta_training = is_meta_training

        self.leo = LEO(self.batchsize, self.input_size, self.latent_size, self.way,
                       self.shot, self.dropout, self.inner_lr_init, self.finetuning_lr_init)

        self.leo_decoder_train_params = list(self.leo.decoder.trainable_params())
        self.leo_encoder_encoder_train_params = list(self.leo.encoder.encoder.trainable_params())
        self.leo_relation_net_train_params = list(self.leo.encoder.relation_network.trainable_params())

        self.reduce_mean = ops.ReduceMean()
        self.matmul = ops.MatMul(transpose_b=True)
        self.norm1 = nn.Norm(axis=1, keep_dims=True)
        self.eye = ops.Eye()
        self.get_shape = ops.Shape()
        self.zeros = ops.Zeros()
        self.ones = ops.Ones()
        self.log = ops.Log()
        self.reshape = ops.Reshape()
        self.squeeze1 = ops.Squeeze(axis=1)
        self.softmax = ops.Softmax()
        self.argmax = ops.Argmax(axis=-1)
        self.equal = ops.Equal()
        self.reduce_sum = ops.ReduceSum()
        self.fill = ops.Fill()
        self.l2loss = ops.L2Loss()
        self.zeros_encoder_penalty = mindspore.Tensor(np.zeros([]), mindspore.float32)

        self.scalar1 = mindspore.Tensor(math.log(2 * math.pi), mindspore.float32)

        self.cal_latents_grad_func = ops.GradOperation(get_all=False, get_by_list=False, sens_param=False)(
            self.leo.withlossdecoder)

        self.cal_weights_grad_func = ops.GradOperation(get_all=True, get_by_list=False)(self.leo.classifier)

    def construct(self, train_inputs, train_labels, val_inputs, val_labels, train=True):
        latents, mean, var = self.leo.encoder(train_inputs, train)
        kl = self.cal_kl_divergence(latents, mean, var)
        train_loss, adapted_classifier_weights, encoder_penalty = self.leo_inner_loop(train_inputs, train_labels,
                                                                                      latents)

        val_loss, val_acc = self.finetune_inner_loop(train_inputs, train_labels, val_inputs, val_labels, train_loss,
                                                     adapted_classifier_weights)

        val_loss = val_loss + self.kl_weight * kl
        val_loss = val_loss + self.encoder_penalty_weight * encoder_penalty

        regularization_penalty = self.orthogonality(self.leo_decoder_train_params[0]) * self.orthogonality_weight

        loss = val_loss + regularization_penalty

        return loss, val_acc

    def leo_inner_loop(self, train_inputs, train_labels, latents):
        start_latents = latents
        classifier_weights = self.leo.decoder(latents)
        loss = self.leo.classifier(train_inputs, classifier_weights, train_labels)

        for _ in range(self.inner_step):
            latents_grad = self.cal_latents_grad_func(latents, train_inputs, train_labels)
            latents = latents - self.leo.inner_lr * latents_grad
            classifier_weights = self.leo.decoder(latents)
            loss = self.leo.classifier(train_inputs, classifier_weights, train_labels)

        getmse = MSELoss()
        if self.is_meta_training:
            encoder_penalty = getmse(latents, start_latents)
        else:
            encoder_penalty = self.zeros_encoder_penalty

        return loss, classifier_weights, encoder_penalty

    def finetune_inner_loop(self, train_inputs, train_labels, val_inputs, val_labels, leo_loss, classifier_weights):
        for _ in range(self.finetune_inner_step):
            loss_grad = self.cal_weights_grad_func(train_inputs, classifier_weights, train_labels)[1]
            classifier_weights = classifier_weights - self.leo.finetuning_lr * loss_grad

        val_loss, val_acc = self.cal_target_loss(val_inputs, classifier_weights, val_labels)

        return val_loss, val_acc

    def cal_kl_divergence(self, latens, mean, var):
        return self.reduce_mean(self.cal_log_prob(latens, mean, var) -
                                self.cal_log_prob(latens, self.zeros(self.get_shape(mean), mindspore.float32),
                                                  self.ones(self.get_shape(var), mindspore.float32)))

    def cal_log_prob(self, x, mean, var):
        eps = 1e-20
        x = x.astype(mindspore.float32)
        mean = mean.astype(mindspore.float32)
        var = var.astype(mindspore.float32)
        log_unnormalized = -0.5 * ((x - mean) / (var + eps)) ** 2
        log_normalization = self.log(var + eps) + 0.5 * self.scalar1
        return log_unnormalized - log_normalization

    def cal_target_loss(self, inputs, classfier_weights, target):
        outputs = self.leo.classifier.predict(classfier_weights, inputs)
        target_loss = self.leo.classifier(inputs, classfier_weights, target)

        target = self.reshape(target, (self.get_shape(target)[0], -1, self.get_shape(target)[-1]))
        target = self.reshape(target, (-1, self.get_shape(target)[-1]))
        target = self.squeeze1(target)

        pred = self.argmax(outputs)
        corr = self.equal(pred, target).astype(np.float32)
        corr = self.reduce_sum(corr)
        total = self.reduce_sum(self.fill(mindspore.float32, self.get_shape(pred), 1))

        return target_loss, corr / total

    def orthogonality(self, weight):
        w2 = self.matmul(weight, weight)

        wn = self.norm1(weight) + 1e-20
        correlation_matrix = w2 / self.matmul(wn, wn)

        I = self.eye(self.get_shape(correlation_matrix)[0], self.get_shape(correlation_matrix)[0], mindspore.float32)
        return self.reduce_mean((correlation_matrix - I) ** 2)

    def l2_regularization(self, weighte, weightr, weightd):
        weightr_zero = weightr[0]
        weightr_one = weightr[1]
        weightr_two = weightr[2]
        output = self.l2loss(weighte) + self.l2loss(weightr_zero) + \
                 self.l2loss(weightr_one) + self.l2loss(weightr_two) + self.l2loss(weightd)

        return output
