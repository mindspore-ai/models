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

"""Metric for accuracy evaluation."""
from collections import defaultdict
import edit_distance
from mindspore.nn.metrics.metric import Metric
import numpy as np

ninf = -np.float('inf')


def _logsumexp(a, b):
    '''logsumexp'''
    if a < b:
        a, b = b, a

    if b == ninf:
        return a
    return a + np.log(1 + np.exp(b - a))


def logsumexp(*args):
    '''logsumexp'''
    res = args[0]
    for e in args[1:]:
        res = _logsumexp(res, e)
    return res


def softmax(logits):
    '''softmax'''
    max_value = np.max(logits, axis=2, keepdims=True)
    exp = np.exp(logits - max_value)
    exp_sum = np.sum(exp, axis=2, keepdims=True)
    dist = exp / exp_sum
    return dist


class LER(Metric):
    """LER Metric
    args: beam(Bool): greedy decoder(False) or prefix beam decoder(True)
          default False because beam decoder is very slow
      """

    def __init__(self, beam=False):
        super(LER, self).__init__()
        self.blank = 61
        self.total_distance = 0.0
        self.length = 0
        self.ninf = ninf
        self.beam = beam

    def clear(self):
        '''clear'''
        self.total_distance = 0
        self.length = 0

    def update(self, *inputs):
        '''update metric'''
        if len(inputs) != 3:
            raise ValueError('CRNNAccuracy need 3 inputs (y_pred, y,seqlen), but got {}'.format(len(inputs)))
        seq_len = self._convert_data(inputs[2])
        logits = self._convert_data(inputs[0])
        if not self.beam:
            hyper = self._ctc_greedy_decoder(logits, seq_len)
        elif self.beam:
            hyper = self._prefix_beam_decode(logits, seq_len)
        labels = self._convert_data(inputs[1])
        truth = []
        for label in labels:
            truth.append(self._remove_blank(label))

        for pred, label in zip(hyper, truth):
            distance = edit_distance.SequenceMatcher(a=pred, b=label).distance()
            self.total_distance += distance / len(label)
            self.length += 1

    def eval(self):
        '''eval per epoch'''
        if self.length == 0:
            raise RuntimeError('Accuary can not be calculated, because the number of length is 0.')
        error = self.total_distance / self.length
        return error

    def _remove_blank(self, inputs):
        '''remove blank'''
        sequence = []
        previous = None
        for i in inputs:
            if i != previous:
                sequence.append(i)
                previous = i
        sequence = [l for l in sequence if l < self.blank]
        return sequence

    def _ctc_greedy_decoder(self, y_pred, seq_len):
        '''greedy decode'''
        hyper = []
        _, batch_size, _ = y_pred.shape
        indices = y_pred.argmax(axis=2)
        pred_labels = []
        for i in range(batch_size):
            idx = indices[:, i]
            last_idx = self.blank
            pred_label = []
            for j in range(seq_len[i]):
                cur_idx = idx[j]
                if cur_idx not in [last_idx, self.blank]:
                    pred_label.append(cur_idx)
                    last_idx = cur_idx
            pred_labels.append(pred_label)
        for pred_label in pred_labels:
            hyper.append(self._remove_blank(pred_label))
        return hyper

    def beam_decode(self, log_y, bid, V, seq_len, beam_size=5):
        '''beam decode for one batch'''
        beam = [(tuple(), (self.blank, self.ninf))]
        for t in range(seq_len[bid]):
            new_beam = defaultdict(lambda: (self.ninf, self.ninf))
            for prefix, (p_b, p_nb) in beam:
                for j in range(V):
                    p = log_y[t, bid, j]
                    if j == self.blank:
                        new_p_b, new_p_nb = new_beam[prefix]
                        new_p_b = logsumexp(new_p_b, p_b + p, p_nb + p)
                        new_beam[prefix] = (new_p_b, new_p_nb)
                        continue
                    else:
                        end_t = prefix[-1] if prefix else None
                        new_prefix = prefix + (j,)
                        new_p_b, new_p_nb = new_beam[new_prefix]
                        if j != end_t:
                            new_p_nb = logsumexp(new_p_nb, p_b + p, p_nb + p)
                        else:
                            new_p_nb = logsumexp(new_p_nb, p_b + p)
                        new_beam[new_prefix] = (new_p_b, new_p_nb)
                        if j == end_t:
                            new_p_b, new_p_nb = new_beam[prefix]
                            new_p_nb = logsumexp(new_p_nb, p_nb + p)
                            new_beam[prefix] = (new_p_b, new_p_nb)
            beam = sorted(new_beam.items(), key=lambda x: logsumexp(*x[1]), reverse=True)
            beam = beam[:beam_size]
        return beam[0][0]

    def _prefix_beam_decode(self, y_pred, seq_len, beam_size=5):
        '''prefix beam search decode'''
        hyper = []
        y_pred = softmax(y_pred)
        _, B, V = y_pred.shape
        pred_labels = []
        log_y = np.log(y_pred)
        for i in range(B):
            pred_label = self.beam_decode(log_y, i, V, seq_len, beam_size)
            pred_labels.append(pred_label)
        for pred_label in pred_labels:
            hyper.append(self._remove_blank(pred_label))
        return hyper
