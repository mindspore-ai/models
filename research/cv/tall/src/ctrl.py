# Copyright 2021 Huawei Technologies Co., Ltd
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
'''CTRL model construction and loss function construction'''
import math

import mindspore
from mindspore import ops, nn
from mindspore.common.initializer import HeUniform, Uniform


class CTRL(nn.Cell):
    """CTRL network structure."""

    # define the operator required
    def __init__(self, visual_dim, sentence_embed_dim, semantic_dim,
                 middle_layer_dim, dropout_rate=1.0):
        super(CTRL, self).__init__()
        self.weight_init = HeUniform(negative_slope=math.sqrt(5))

        self.semantic_dim = semantic_dim
        self.visual_dim = visual_dim
        self.sentence_embed_dim = sentence_embed_dim

        self.v2s_fc = nn.Dense(visual_dim, semantic_dim, weight_init=self.weight_init,
                               bias_init=Uniform(scale=self.get_scale((semantic_dim, visual_dim))),
                               has_bias=True)
        self.s2s_fc = nn.Dense(sentence_embed_dim, semantic_dim, weight_init=self.weight_init,
                               bias_init=Uniform(scale=self.get_scale((semantic_dim, sentence_embed_dim))),
                               has_bias=True)
        self.fc1 = nn.Dense(semantic_dim * 3, semantic_dim)
        self.fc2 = nn.Dense(semantic_dim, 3)
        self.fc_concat = nn.Dense(semantic_dim * 2, semantic_dim, weight_init=self.weight_init,
                                  bias_init=Uniform(scale=self.get_scale((semantic_dim, semantic_dim * 2))),
                                  has_bias=True)

        self.bn1 = nn.BatchNorm1d(num_features=semantic_dim)
        self.bn2 = nn.BatchNorm1d(num_features=semantic_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(keep_prob=dropout_rate)

        self.tile = ops.Tile()
        self.reshape = ops.Reshape()
        self.concat = ops.Concat(axis=2)
        self.expand_dims = ops.ExpandDims()
        self.transpose = ops.Transpose()

    def get_scale(self, shape):
        '''Get uniform distribution initialization parameter scale'''
        fan_in = self._calculate_correct_fan(shape, mode='fan_in')
        res = 1 / math.sqrt(fan_in)
        return res

    def _calculate_correct_fan(self, shape, mode):
        """
        Calculate fan.

        Args:
            shape (tuple): input shape.
            mode (str): only support fan_in and fan_out.

        Returns:
            fan_in or fan_out.
        """
        mode = mode.lower()
        valid_modes = ['fan_in', 'fan_out']
        if mode not in valid_modes:
            raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))
        fan_in, fan_out = self._calculate_fan_in_and_fan_out(shape)
        return fan_in if mode == 'fan_in' else fan_out

    def _calculate_fan_in_and_fan_out(self, shape):
        """
        calculate fan_in and fan_out

        Args:
            shape (tuple): input shape.

        Returns:
            Tuple, a tuple with two elements, the first element is `n_in` and the second element is `n_out`.
        """
        dimensions = len(shape)
        if dimensions < 2:
            raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
        if dimensions == 2:  # Linear
            fan_in = shape[1]
            fan_out = shape[0]
        else:
            num_input_fmaps = shape[1]
            num_output_fmaps = shape[0]
            receptive_field_size = 1
            if dimensions > 2:
                receptive_field_size = shape[2] * shape[3]
            fan_in = num_input_fmaps * receptive_field_size
            fan_out = num_output_fmaps * receptive_field_size
        return fan_in, fan_out

    def construct(self, input_feature):
        '''CTRL model construction'''
        visual_feature, sentence_embed = input_feature[:, :self.visual_dim], input_feature[:, self.visual_dim:]
        batch_size, _ = visual_feature.shape
        transformed_clip = self.v2s_fc(visual_feature)
        transformed_sentence = self.s2s_fc(sentence_embed)
        #print("transformed_sentence")
        #print(transformed_sentence)
        transformed_clip = self.bn1(transformed_clip)
        transformed_sentence = self.bn2(transformed_sentence)

        multiples = (batch_size, 1)
        vv_f = self.tile(transformed_clip, multiples)
        vv_f = self.reshape(vv_f, (batch_size, batch_size, self.semantic_dim))
        multiples = (1, batch_size)
        ss_f = self.tile(transformed_sentence, multiples)
        ss_f = self.reshape(ss_f, (batch_size, batch_size, self.semantic_dim))
        mul_feature = vv_f * ss_f
        add_feature = vv_f + ss_f
        cat_feature = self.concat((vv_f, ss_f))
        cat_feature = self.fc_concat(cat_feature)
        cross_modal_vec = self.concat((mul_feature, add_feature, cat_feature))
        cross_modal_vec = self.expand_dims(cross_modal_vec, 0)
        out = self.reshape(cross_modal_vec, (batch_size*batch_size, 3*1024))
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.reshape(out, (batch_size, batch_size, 3))

        return out

class CTRL_Loss(nn.Cell):
    """CTRL Loss structure."""
    def __init__(self, lambda_reg):
        super(CTRL_Loss, self).__init__()
        self.lambda_reg = lambda_reg
        self.split = ops.Split(2, 3)
        self.reshape = ops.Reshape()
        self.eye = ops.Eye()
        self.ones = ops.Ones()
        self.mul = ops.Mul()
        self.exp = ops.Exp()
        self.log = ops.Log()
        self.mean = ops.ReduceMean(keep_dims=False)
        self.matmul = ops.MatMul()
        self.concat = ops.Concat(axis=1)
        self.abs = ops.Abs()
        self.smooth_loss = nn.SmoothL1Loss()

    def construct(self, output, offset_label):
        '''CTRL loss function construction'''
        batch_size = output.shape[0]
        sim_score_mat, p_reg_mat, l_reg_mat = self.split(output)
        sim_score_mat = self.reshape(sim_score_mat, (batch_size, batch_size))
        p_reg_mat = self.reshape(p_reg_mat, (batch_size, batch_size))
        l_reg_mat = self.reshape(l_reg_mat, (batch_size, batch_size))

        I_2 = 2.0 * self.eye(batch_size, batch_size, mindspore.float32)
        all1 = self.ones((batch_size, batch_size), mindspore.float32)
        mask = all1 - I_2

        I = self.eye(batch_size, batch_size, mindspore.float32)
        batch_para_mat = self.ones((batch_size, batch_size), mindspore.float32) / 64
        para_mat = I + batch_para_mat
        loss_mat = self.log(all1 + self.exp(self.mul(mask, sim_score_mat)))
        loss_mat = self.mul(loss_mat, para_mat)
        loss_align = self.mean(loss_mat)

        l_reg_diag = self.matmul(self.mul(l_reg_mat, I), self.ones((batch_size, 1), mindspore.float32))
        p_reg_diag = self.matmul(self.mul(p_reg_mat, I), self.ones((batch_size, 1), mindspore.float32))
        offset_pred = self.concat((p_reg_diag, l_reg_diag))
        loss_reg = self.mean(self.smooth_loss(offset_pred, offset_label))
        loss = loss_align + self.lambda_reg * loss_reg
        return loss
