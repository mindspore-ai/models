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
from src.dcrnn_cell import DCGRUCell
import mindspore
from mindspore import dtype as mstype
import mindspore.ops as ops
from mindspore.ops import operations as P


def count_parameters(model):
    print(model.parameters())
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Seq2SeqAttrs:
    def __init__(self, adj_mx, **model_kwargs):
        self.adj_mx = adj_mx
        self.max_diffusion_step = int(model_kwargs.get('max_diffusion_step', 2))
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        self.filter_type = model_kwargs.get('filter_type', 'laplacian')
        self.num_nodes = int(model_kwargs.get('num_nodes', 1))
        self.num_rnn_layers = int(model_kwargs.get('num_rnn_layers', 1))
        self.rnn_units = int(model_kwargs.get('rnn_units'))
        self.hidden_state_size = self.num_nodes * self.rnn_units


class EncoderModel(mindspore.nn.Cell, Seq2SeqAttrs):
    def __init__(self, adj_mx, is_fp16, **model_kwargs):
        mindspore.nn.Cell.__init__(self)
        Seq2SeqAttrs.__init__(self, adj_mx, **model_kwargs)
        self.input_dim = int(model_kwargs.get('input_dim', 1))
        self.seq_len = int(model_kwargs.get('seq_len'))  # for the encoder
        self.zeros = ops.Zeros()
        self.stack = ops.Stack()
        self.dcgru_layers = mindspore.nn.CellList(
            [DCGRUCell(self.rnn_units, adj_mx, self.max_diffusion_step,
                       self.num_nodes, is_fp16, filter_type=self.filter_type),
             DCGRUCell(self.rnn_units, adj_mx, self.max_diffusion_step,
                       self.num_nodes, is_fp16, filter_type=self.filter_type)])

    def construct(self, inputs, hidden_state=None):
        batch_size, _ = inputs.shape
        if hidden_state is None:
            x = (self.num_rnn_layers, batch_size, self.hidden_state_size)
            hidden_state = self.zeros(x, mindspore.float32)
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        stack_hidden_states = self.stack(hidden_states)
        return output, stack_hidden_states


class DecoderModel(mindspore.nn.Cell, Seq2SeqAttrs):
    def __init__(self, adj_mx, is_fp16, **model_kwargs):
        mindspore.nn.Cell.__init__(self)
        Seq2SeqAttrs.__init__(self, adj_mx, **model_kwargs)
        self.output_dim = int(model_kwargs.get('output_dim', 1))
        self.horizon = int(model_kwargs.get('horizon', 1))
        self.projection_layer = mindspore.nn.Dense(self.rnn_units, self.output_dim)
        self.stack = ops.Stack()
        self.dcgru_layers = mindspore.nn.CellList(
            [DCGRUCell(self.rnn_units, adj_mx, self.max_diffusion_step,
                       self.num_nodes, is_fp16, filter_type=self.filter_type),
             DCGRUCell(self.rnn_units, adj_mx, self.max_diffusion_step,
                       self.num_nodes, is_fp16, filter_type=self.filter_type)])

    def construct(self, inputs, hidden_state=None):
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        projected = self.projection_layer(output.view(-1, self.rnn_units))
        output = projected.view(-1, self.num_nodes * self.output_dim)
        return output, self.stack(hidden_states)


class DCRNNModel(mindspore.nn.Cell, Seq2SeqAttrs):
    def __init__(self, adj_mx, is_fp16, batch_size, **model_kwargs):
        super().__init__()
        Seq2SeqAttrs.__init__(self, adj_mx, **model_kwargs)
        self.batch_size = batch_size
        self.encoder_model = EncoderModel(adj_mx, is_fp16, **model_kwargs)
        self.decoder_model = DecoderModel(adj_mx, is_fp16, **model_kwargs)
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        self.use_curriculum_learning = bool(model_kwargs.get('use_curriculum_learning', False))
        self.cast = P.Cast()
        self.transpose = ops.Transpose()
        self.split = ops.Split(0, batch_size*2)
        self.squeeze = ops.Squeeze(0)
        self.reshape = ops.Reshape()
        self.stack = ops.Stack()
        self.zeros = ops.Zeros()

        self.num_nodes = int(model_kwargs.get('num_nodes', 1))
        self.input_dim = int(model_kwargs.get('input_dim', 1))
        self.seq_len = int(model_kwargs.get('seq_len'))  # for the encoder
        self.output_dim = int(model_kwargs.get('output_dim', 1))
        self.use_curriculum_learning = bool(model_kwargs.get('use_curriculum_learning', False))
        self.horizon = int(model_kwargs.get('horizon', 1))  # for the decoder

    def _prepare_data(self, x, y):
        x, y = self._get_x_y(x, y)
        x, y = self._get_x_y_in_correct_dims(x, y)
        return x, y

    def _get_x_y(self, x, y):
        x = self.cast(x, mstype.float32)
        y = self.cast(y, mstype.float32)
        x = self.transpose(x, (1, 0, 2, 3))
        y = self.transpose(y, (1, 0, 2, 3))
        return x, y

    def _get_x_y_in_correct_dims(self, x, y):
        batch_size = x.shape[1]
        x = x.view(self.seq_len, batch_size, self.num_nodes * self.input_dim)
        y = y[..., :self.output_dim].view(self.horizon, batch_size, self.num_nodes * self.output_dim)
        return x, y

    def _compute_sampling_threshold(self, batches_seen):
        index = np.exp(batches_seen / self.cl_decay_steps)
        return self.cl_decay_steps / (self.cl_decay_steps + index)

    def encoder(self, inputs):
        encoder_hidden_state = None
        for t in range(self.encoder_model.seq_len):
            _, encoder_hidden_state = self.encoder_model(inputs[t], encoder_hidden_state)
        return encoder_hidden_state

    def decoder(self, encoder_hidden_state, labels=None):
        batch_size = encoder_hidden_state.shape[1]
        go_symbol = self.zeros((batch_size, self.num_nodes * self.decoder_model.output_dim), mindspore.float32)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol
        outputs = []

        for t in range(self.decoder_model.horizon):
            t = t
            decoder_output, decoder_hidden_state = self.decoder_model(decoder_input, decoder_hidden_state)
            decoder_input = decoder_output
            outputs.append(decoder_output)

        outputs = mindspore.ops.stack(outputs)

        return outputs

    def _separate_data(self, data):

        data = self.cast(data, mstype.float32)
        inputs = []
        labels = []

        (batch_size, xy, a, b, c) = data.shape
        data = self.reshape(data, (batch_size * xy, a, b, c))
        temp = self.split(data)
        tem = []
        for tensor in temp:
            tem.append(self.squeeze(tensor))
        # temp = [self.squeeze(tensor) for tensor in temp]

        for i in range(self.batch_size * 2):
            if i % 2 == 0:
                inputs.append(tem[i])
            else:
                labels.append(tem[i])
        inputs = self.stack(inputs)
        labels = self.stack(labels)
        return inputs, labels

    def construct(self, data):
        inputs, labels = self._separate_data(data)
        inputs, labels = self._prepare_data(inputs, labels)
        encoder_hidden_state = self.encoder(inputs)
        outputs = self.decoder(encoder_hidden_state, labels)
        return outputs
