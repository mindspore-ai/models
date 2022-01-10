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
"""LSTM_CRF."""
import numpy as np

from mindspore import Tensor, nn, Parameter
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype
from .LSTM import Lstm


STACK_LSTM_CRF_DEVICE = ["CPU"]


class Lstm_CRF(nn.Cell):
    """Lstm_CRF network structure"""
    def __init__(self,
                 vocab_size,
                 tag_to_index,
                 embedding_size,
                 hidden_size,
                 num_layers,
                 weight=None,
                 bidirectional=False,
                 batch_size=1,
                 seq_length=1,
                 dropout=0.0,
                 is_training=True
                 ):
        super(Lstm_CRF, self).__init__()
        self.vocab_size = vocab_size
        self.tag_to_index = tag_to_index
        self.embedding_size = embedding_size
        self.num_hiddens = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.transpose = P.Transpose()

        self.is_training = is_training
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.START_TAG = "<START>"
        self.STOP_TAG = "<STOP>"
        self.tag_to_index[self.START_TAG] = len(self.tag_to_index)
        self.tag_to_index[self.STOP_TAG] = len(self.tag_to_index)
        self.out_size = len(self.tag_to_index)
        self.START_VALUE = Tensor(self.tag_to_index[self.START_TAG], dtype=mstype.int32)
        self.STOP_VALUE = Tensor(self.tag_to_index[self.STOP_TAG], dtype=mstype.int32)

        # Matrix of transition parameters.
        transitions = np.random.normal(size=(self.out_size, self.out_size)).astype(np.float32)
        transitions[self.tag_to_index[self.START_TAG], :] = -10000
        transitions[:, self.tag_to_index[self.STOP_TAG]] = -10000
        self.transitions = Parameter(Tensor(transitions))

        self.lstm = Lstm(vocab_size,
                         embedding_size,
                         hidden_size,
                         out_size=self.out_size,
                         weight=weight,
                         num_layers=num_layers,
                         batch_size=batch_size,
                         dropout=dropout,
                         bidirectional=bidirectional)

        self.cat = P.Concat(axis=-1)
        self.argmax = P.ArgMaxWithValue(axis=-1)
        self.log = P.Log()
        self.exp = P.Exp()
        self.sum = P.ReduceSum()
        self.tile = P.Tile()
        self.reduce_sum = P.ReduceSum(keep_dims=True)
        self.reshape = P.Reshape()
        self.expand = P.ExpandDims()
        self.mean = P.ReduceMean()
        init_alphas = np.ones(shape=(self.batch_size, self.out_size)) * -10000.0
        init_alphas[:, self.tag_to_index[self.START_TAG]] = 0.
        self.init_alphas = Tensor(init_alphas, dtype=mstype.float32)
        self.cast = P.Cast()
        self.reduce_max = P.ReduceMax(keep_dims=True)
        self.on_value = Tensor(1.0, dtype=mstype.float32)
        self.off_value = Tensor(0.0, dtype=mstype.float32)
        self.onehot = P.OneHot()

    def _realpath_score(self, features, label):
        '''
        Compute the emission and transition score for the real path.
        '''
        label = label * 1
        concat_A = self.tile(self.reshape(self.START_VALUE, (1,)), (self.batch_size,))
        concat_A = self.reshape(concat_A, (self.batch_size, 1))
        labels = self.cat((concat_A, label))
        onehot_label = self.onehot(label, self.out_size, self.on_value, self.off_value)
        emits = features * onehot_label
        labels = self.onehot(labels, self.out_size, self.on_value, self.off_value)
        label1 = labels[:, 1:, :]
        label2 = labels[:, :self.seq_length, :]
        label1 = self.expand(label1, 3)
        label2 = self.expand(label2, 2)
        label_trans = label1 * label2
        transitions = self.expand(self.expand(self.transitions, 0), 0)
        trans = transitions * label_trans
        score = self.sum(emits, (1, 2)) + self.sum(trans, (1, 2, 3))
        stop_value_index = labels[:, (self.seq_length-1):self.seq_length, :]
        stop_value = self.transitions[(self.out_size-1):self.out_size, :]
        stop_score = stop_value * self.reshape(stop_value_index, (self.batch_size, self.out_size))
        score = score + self.sum(stop_score, 1)
        score = self.reshape(score, (self.batch_size, -1))
        return score

    def _normalization_factor(self, features):
        '''
        Compute the total score for all the paths.
        '''
        forward_var = self.init_alphas
        forward_var = self.expand(forward_var, 1)
        for idx in range(self.seq_length):
            feat = features[:, idx:(idx+1), :]
            emit_score = self.reshape(feat, (self.batch_size, self.out_size, 1))
            next_tag_var = emit_score + self.transitions + forward_var
            forward_var = self.log_sum_exp(next_tag_var)
            forward_var = self.reshape(forward_var, (self.batch_size, 1, self.out_size))
        terminal_var = forward_var + self.reshape(self.transitions[(self.out_size-1):self.out_size, :], (1, -1))
        alpha = self.log_sum_exp(terminal_var)
        alpha = self.reshape(alpha, (self.batch_size, -1))
        return alpha

    def _decoder(self, features):
        '''
        Viterbi decode for evaluation.
        '''
        backpointers = ()
        forward_var = self.init_alphas
        for idx in range(self.seq_length):
            feat = features[:, idx:(idx+1), :]
            feat = self.reshape(feat, (self.batch_size, self.out_size))
            bptrs_t = ()

            next_tag_var = self.expand(forward_var, 1) + self.transitions
            best_tag_id, best_tag_value = self.argmax(next_tag_var)
            bptrs_t += (best_tag_id,)
            forward_var = best_tag_value + feat

            backpointers += (bptrs_t,)
        terminal_var = forward_var + self.reshape(self.transitions[(self.out_size-1):self.out_size, :], (1, -1))
        best_tag_id, _ = self.argmax(terminal_var)
        return backpointers, best_tag_id

    def log_sum_exp(self, logits):
        '''
        Compute the log_sum_exp score for Normalization factor.
        '''
        max_score = self.reduce_max(logits, -1)  #16 5 5
        score = self.log(self.reduce_sum(self.exp(logits - max_score), -1))
        score = max_score + score
        return score

    def construct(self, feature, label):
        """
        Get the emission scores from the BiLSTM
        """
        emission = self.lstm(feature)
        emission = self.transpose(emission, (1, 0, 2))
        if self.is_training:
            forward_score = self._normalization_factor(emission)
            gold_score = self._realpath_score(emission, label)
            return_value = self.mean(forward_score - gold_score)
        else:
            path_list, tag = self._decoder(emission)
            return_value = path_list, tag
        return return_value


# CRF postprocess
def postprocess(backpointers, best_tag_id):
    '''
    Do postprocess
    '''
    best_tag_id = best_tag_id.asnumpy()
    batch_size = len(best_tag_id)
    best_path = []
    for i in range(batch_size):
        best_path.append([])
        best_local_id = best_tag_id[i]
        best_path[-1].append(best_local_id)
        for bptrs_t in reversed(backpointers):
            bptrs_t = bptrs_t[0].asnumpy()
            local_idx = bptrs_t[i]
            best_local_id = local_idx[best_local_id]
            best_path[-1].append(best_local_id)
        # Pop off the start tag (we dont want to return that to the caller)
        best_path[-1].pop()
        best_path[-1].reverse()
    return best_path
