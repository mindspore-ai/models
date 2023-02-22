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
import pyscipopt as scip
from pyscipopt import SCIP_RESULT

from third_party.logger import logger
from utils import advanced_cut_feature_generator

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class Encoder(nn.Cell):
    """Maps a graph represented as an input sequence
    to a hidden vector"""

    def __init__(self, input_dim, hidden_dim, use_cuda):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.use_cuda = use_cuda
        self.enc_init_hx = mindspore.numpy.zeros(hidden_dim)
        self.enc_init_cx = mindspore.numpy.zeros(hidden_dim)

        self.enc_init_state = (self.enc_init_hx, self.enc_init_cx)

    def construct(self, x, hidden):
        output, hidden = self.lstm(x, hidden)
        return output, hidden

    def init_hidden(self, hidden_dim):
        """Trainable initial hidden state"""
        enc_init_hx = mindspore.numpy.zeros(hidden_dim)

        enc_init_cx = mindspore.numpy.zeros(hidden_dim)

        return (enc_init_hx, enc_init_cx)


class Attention(nn.Cell):
    """A generic attention module for a decoder in seq2seq"""

    def __init__(self, dim, use_tanh=False, C=10, use_cuda=True):
        super(Attention, self).__init__()
        self.use_tanh = use_tanh
        self.project_query = nn.Dense(dim, dim)
        self.project_ref = nn.Conv1d(dim, dim, 1, 1)
        self.C = C
        self.tanh = nn.Tanh()

        self.ones = mindspore.ops.Ones()
        self.v = mindspore.Parameter(self.ones(dim, mindspore.float32))
        self.expand_dims = mindspore.ops.ExpandDims()
        self.tile = mindspore.ops.Tile()
        self.shape = mindspore.ops.Shape()
        self.batmatmul = mindspore.ops.BatchMatMul()

    def construct(self, query, ref):
        """
        Args:
            query: is the hidden state of the decoder at the current
                time step. batch x dim
            ref: the set of hidden states from the encoder.
                sourceL x batch x hidden_dim
        """
        reshape = mindspore.ops.Reshape()
        ref = reshape(
            ref, (self.shape(ref)[1], self.shape(ref)[2], self.shape(ref)[0]))
        q = self.expand_dims(self.project_query(query), 2)
        e = self.project_ref(ref)
        expanded_q = self.tile(q, (1, 1, self.shape(e)[2]))
        broadcast_to = mindspore.ops.BroadcastTo(
            (self.shape(expanded_q)[0], self.shape(self.v)[0]))
        v_view = self.expand_dims(self.v, 0)
        v_view = broadcast_to(v_view)
        v_view = self.expand_dims(v_view, 1)

        squeeze = mindspore.ops.Squeeze(1)
        u = squeeze(self.batmatmul(v_view, self.tanh(expanded_q + e)))
        if self.use_tanh:
            logits = self.C * self.tanh(u)
        else:
            logits = u
        return e, logits


class Decoder(nn.Cell):
    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 tanh_exploration,
                 use_tanh,
                 n_glimpses=1,
                 beam_size=0,
                 use_cuda=True):
        super(Decoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_glimpses = n_glimpses
        self.beam_size = beam_size
        self.use_cuda = use_cuda

        self.input_weights = nn.Dense(embedding_dim, 4 * hidden_dim)
        self.hidden_weights = nn.Dense(hidden_dim, 4 * hidden_dim)

        self.pointer = Attention(
            hidden_dim, use_tanh=use_tanh,
            C=tanh_exploration, use_cuda=self.use_cuda)
        self.glimpse = Attention(
            hidden_dim, use_tanh=False, use_cuda=self.use_cuda)
        self.sm = nn.Softmax()

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.shape = mindspore.ops.Shape()
        self.zeros = mindspore.ops.Zeros()
        self.expand_dims = mindspore.ops.ExpandDims()
        self.tile = mindspore.ops.Tile()
        self.matmul = mindspore.ops.BatchMatMul()

    def apply_mask_to_logits(self, step, logits, mask, prev_idxs):
        if mask is None:
            mask = self.zeros(self.shape(logits), mindspore.float32)

        maskk = mask

        if prev_idxs is not None:
            maskk[[x for x in range(self.shape(logits)[0])],
                  prev_idxs.item()] = 1
            logits[maskk] = mindspore.Tensor(-1e5, mindspore.float32)
        return logits, maskk

    def logprobs(
            self, decoder_input, embedded_inputs,
            hidden, context, max_length, seled_idxes
    ):
        def recurrence(x, hidden, logit_mask, prev_idxs, step):

            hx, cx = hidden
            gates = self.input_weights(x) + self.hidden_weights(hx)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = self.sigmoid(ingate)
            forgetgate = self.sigmoid(forgetgate)
            cellgate = self.tanh(cellgate)
            outgate = self.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * self.tanh(cy)
            g_l = hy
            for _ in range(self.n_glimpses):
                ref, logits = self.glimpse(g_l, context)
                logits, logit_mask = self.apply_mask_to_logits(
                    step, logits, logit_mask, prev_idxs)
                g_l = mindspore.numpy.tensordot(
                    ref, self.sm(logits).unsqueeze(2)).squeeze(2)
            _, logits = self.pointer(g_l, context)
            logits, logit_mask = self.apply_mask_to_logits(
                step, logits, logit_mask, prev_idxs)
            probs = self.sm(logits)
            return hy, cy, probs, logit_mask

        outputs = []
        single_probs = []
        steps = range(max_length)
        inps = []
        idxs = None
        mask = None

        for i in steps:
            hx, cx, probs, mask = recurrence(
                decoder_input, hidden, mask, idxs, i)
            hidden = (hx, cx)
            decoder_input, prob = self.decode_logp(
                probs,
                embedded_inputs,
                seled_idxes[i])
            inps.append(decoder_input)
            idxs = mindspore.numpy.array(
                [seled_idxes[i]], dtype=mindspore.numpy.int64
            ).to(self.pointer.v.device)
            outputs.append(probs)
            single_probs.append(prob)

        return (outputs, single_probs), hidden

    def decode_logp(self, probs, embedded_inputs, idxs):
        batch_size = probs.size(0)
        sels = embedded_inputs[idxs, [i for i in range(batch_size)], :]
        return sels, probs[:, idxs]

    def construct(
            self, decoder_input, embedded_inputs, hidden, context, max_length
    ):
        """
        Args:
            decoder_input: The initial input to the decoder
                size is [batch_size x embedding_dim]. Trainable parameter.
            embedded_inputs: [sourceL x batch_size x embedding_dim]
            hidden: the prev hidden state, size is [batch_size x hidden_dim].
                Initially this is set to (enc_h[-1], enc_c[-1])
            context: encoder outputs, [sourceL x batch_size x hidden_dim]
        """

        def recurrence(x, hidden, logit_mask, prev_idxs, step):

            hx, cx = hidden
            gates = self.input_weights(x) + self.hidden_weights(hx)
            split = mindspore.ops.Split(1, 4)
            ingate, forgetgate, cellgate, outgate = split(gates)

            ingate = self.sigmoid(ingate)

            forgetgate = self.sigmoid(forgetgate)
            cellgate = self.tanh(cellgate)
            outgate = self.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * self.tanh(cy)
            g_l = hy
            _, logits = self.pointer(g_l, context)
            probs = self.sm(logits)
            return hy, cy, probs, logit_mask

        outputs = []
        selections = []
        steps = range(max_length)
        inps = []
        idxs = None
        mask = None
        decode_type = "greedy"
        if decode_type in ["stochastic", "greedy"]:
            for i in steps:
                hx, cx, probs, mask = recurrence(
                    decoder_input, hidden, mask, idxs, i)
                hidden = (hx, cx)
                decoder_input, idxs = self.decode(
                    probs,
                    embedded_inputs,
                    selections,
                    decode_type)
                inps.append(decoder_input)
                outputs.append(probs)
                selections.append(idxs)
            return (outputs, selections), hidden
    def decode(self, probs, embedded_inputs, selections, decode_type):
        """
        Return the next input for the decoder by selecting the
        input with sampling

        Args:
            probs: [batch_size x sourceL]
            embedded_inputs: [sourceL x batch_size x embedding_dim]
            selections: list of all of the previously selected indices during decoding
       Returns:
            Tensor of size [batch_size x sourceL] containing the embeddings
            from the inputs corresponding to the [batch_size] indices
            selected for this iteration of the decoding, as well as the
            corresponding indices
        """
        batch_size = self.shape(probs)[0]
        idxs = mindspore.ops.multinomial(probs, 1)
        squeeze = mindspore.ops.Squeeze(1)
        idxs = squeeze(idxs)

        sels = embedded_inputs[idxs.item(), [i for i in range(batch_size)], :]
        return sels, idxs


class PointerNetwork(nn.Cell):
    """The pointer network, which is the core seq2seq
    model"""

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 n_glimpses,
                 tanh_exploration,
                 use_tanh,
                 beam_size,
                 use_cuda):
        super(PointerNetwork, self).__init__()

        self.embedding_dim = embedding_dim
        self.encoder = Encoder(
            embedding_dim,
            hidden_dim,
            use_cuda)

        self.decoder = Decoder(
            embedding_dim,
            hidden_dim,
            tanh_exploration=tanh_exploration,
            use_tanh=use_tanh,
            n_glimpses=n_glimpses,
            beam_size=beam_size,
            use_cuda=use_cuda)

        self.ones = mindspore.ops.Ones()
        self.decoder_in_0 = mindspore.Parameter(
            self.ones(embedding_dim, mindspore.float32))
        self.expand_dims = mindspore.ops.ExpandDims()
        self.tile = mindspore.ops.Tile()
        self.shape = mindspore.ops.Shape()

    def construct(self, inputs, max_decode_len):
        """ Propagate inputs through the network
        Args:
            inputs: [sourceL x batch_size x embedding_dim]
        """

        (encoder_hx, encoder_cx) = self.encoder.enc_init_state
        encoder_hx = self.expand_dims(self.tile(self.expand_dims(
            encoder_hx, 0), (self.shape(inputs)[1], 1)), 0)
        encoder_cx = self.expand_dims(self.tile(self.expand_dims(
            encoder_cx, 0), (self.shape(inputs)[1], 1)), 0)

        enc_h, (enc_h_t, enc_c_t) = self.encoder(
            inputs, (encoder_hx, encoder_cx))

        dec_init_state = (enc_h_t[-1], enc_c_t[-1])

        decoder_input = self.expand_dims(self.decoder_in_0, 0)
        decoder_input = self.tile(decoder_input, (self.shape(inputs)[1], 1))
        print(self.shape(decoder_input))
        (pointer_probs, input_idxs), _ = self.decoder(
            decoder_input, inputs, dec_init_state, enc_h, max_decode_len)

        return pointer_probs, input_idxs

    def _prob_to_logp(self, prob):
        logprob = 0
        for p in prob:
            logp = mindspore.numpy.log(p)
            logprob += logp

        return logprob

    def logprobs(self, inputs, max_decode_len, seled_idxes):
        """ Propagate inputs through the network
        Args:
            inputs: [sourceL x batch_size x embedding_dim]
        """

        (encoder_hx, encoder_cx) = self.encoder.enc_init_state
        encoder_hx = encoder_hx.unsqueeze(0).repeat(
            inputs.size(1), 1).unsqueeze(0)
        encoder_cx = encoder_cx.unsqueeze(0).repeat(
            inputs.size(1), 1).unsqueeze(0)

        enc_h, (enc_h_t, enc_c_t) = self.encoder(
            inputs, (encoder_hx, encoder_cx))

        dec_init_state = (enc_h_t[-1], enc_c_t[-1])

        decoder_input = self.decoder_in_0.unsqueeze(
            0).repeat(inputs.size(1), 1)

        (pointer_probs, probs), _ = self.decoder.logprobs(
            decoder_input, inputs, dec_init_state,
            enc_h, max_decode_len, seled_idxes
        )
        logprob = self._prob_to_logp(probs)

        pointer_probs_list = [pointer_prob.cpu().detach()
                              for pointer_prob in pointer_probs]
        return pointer_probs_list, logprob


class CriticNetwork(nn.Cell):
    """Useful as a baseline in REINFORCE updates"""

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 n_process_block_iters,
                 tanh_exploration,
                 use_tanh,
                 use_cuda):
        super(CriticNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_process_block_iters = n_process_block_iters

        self.encoder = Encoder(
            embedding_dim,
            hidden_dim,
            use_cuda)

        self.process_block = Attention(
            hidden_dim, use_tanh=use_tanh,
            C=tanh_exploration, use_cuda=use_cuda)
        self.sm = nn.Softmax()
        self.decoder = nn.Sequential(
            nn.Dense(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dense(hidden_dim, 1)
        )

    def construct(self, inputs):
        """
        Args:
            inputs: [embedding_dim x batch_size x sourceL] of embedded inputs
        """

        (encoder_hx, encoder_cx) = self.encoder.enc_init_state
        encoder_hx = encoder_hx.unsqueeze(0).repeat(
            inputs.size(1), 1).unsqueeze(0)
        encoder_cx = encoder_cx.unsqueeze(0).repeat(
            inputs.size(1), 1).unsqueeze(0)

        enc_outputs, (enc_h_t, _) = self.encoder(
            inputs, (encoder_hx, encoder_cx))

        process_block_state = enc_h_t[-1]
        for _ in range(self.n_process_block_iters):
            ref, logits = self.process_block(process_block_state, enc_outputs)
            process_block_state = mindspore.numpy.tensordot(
                ref, self.sm(logits).unsqueeze(2)).squeeze(2)
        out = self.decoder(process_block_state)
        return out


class CutsPercentPolicy(nn.Cell):
    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 n_process_block_iters,
                 tanh_exploration,
                 use_tanh,
                 use_cuda):
        super(CutsPercentPolicy, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_process_block_iters = n_process_block_iters

        self.encoder = Encoder(
            embedding_dim,
            hidden_dim,
            use_cuda)

        self.process_block = Attention(
            hidden_dim, use_tanh=use_tanh,
            C=tanh_exploration, use_cuda=use_cuda
        )
        self.sm = nn.Softmax()
        self.decoder = nn.Sequential(
            nn.Dense(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dense(hidden_dim, 2)
        )
        self.use_cuda = use_cuda

    def construct(self, inputs):
        """
        Args:
            inputs: [embedding_dim x batch_size x sourceL] of embedded inputs
        """

        (encoder_hx, encoder_cx) = self.encoder.enc_init_state
        encoder_hx = encoder_hx.unsqueeze(0).repeat(
            inputs.size(1), 1).unsqueeze(0)
        encoder_cx = encoder_cx.unsqueeze(0).repeat(
            inputs.size(1), 1).unsqueeze(0)

        enc_outputs, (enc_h_t, _) = self.encoder(
            inputs, (encoder_hx, encoder_cx))

        process_block_state = enc_h_t[-1]
        for _ in range(self.n_process_block_iters):
            ref, logits = self.process_block(process_block_state, enc_outputs)
            process_block_state = mindspore.numpy.tensordot(
                ref, self.sm(logits).unsqueeze(2)).squeeze(2)
        out = self.decoder(process_block_state)
        return out

    def action(self, states, deterministic=False):
        mean, log_std = self.get_mean_std(states)
        std = mindspore.numpy.exp(log_std)
        if deterministic:
            action = mean
        else:
            action = mean + std
        tanh_action = mindspore.numpy.tanh(action)

        return tanh_action

    def get_mean_std(self, states):
        out = self.construct(states)
        mean, log_std = mindspore.numpy.chunk(out, 2, -1)
        log_std = mindspore.numpy.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        log_std = log_std.expand(mean.shape)

        return mean, log_std

    def log_prob(self, states, action=None, pretanh_action=None):
        if pretanh_action is None:
            assert action is not None
            pretanh_action = mindspore.numpy.log(
                (1 + action) / (1 - action) + 1e-6) / 2
        else:
            assert pretanh_action is not None
            action = mindspore.numpy.tanh(pretanh_action)
        mean, log_std = self.get_mean_std(states)
        std = mindspore.numpy.exp(log_std)
        pre_log_prob = pretanh_action
        log_prob = pre_log_prob.sum(-1, keepdim=True) - mindspore.numpy.log(
            1 - action * action + 1e-6).sum(-1, keepdim=True)
        info = {}
        info['pre_log_prob'] = pre_log_prob
        info['mean'] = mean
        info['std'] = std

        return log_prob, info


class CutSelectAgent(scip.Cutsel):
    def __init__(
            self,
            scip_model,
            pointer_net,
            value_net,
            sel_cuts_percent,
            device,
            decode_type,
            baseline_type
    ):
        super().__init__()
        self.scip_model = scip_model
        self.policy = pointer_net
        self.value = value_net
        self.sel_cuts_percent = sel_cuts_percent
        self.device = device
        self.decode_type = decode_type
        self.baseline_type = baseline_type

        self.data = {}

    def cutselselect(self, cuts, forcedcuts, root, maxnselectedcuts):
        '''first method called in each iteration in the main solving loop. '''
        logger.log(f"forcedcuts length: {len(forcedcuts)}")
        logger.log(f"len cuts: {len(cuts)}")
        num_cuts = len(cuts)
        if num_cuts <= 1:
            return {
                'cuts': cuts,
                'nselectedcuts': 1,
                'result': SCIP_RESULT.SUCCESS
            }
        sel_cuts_num = min(
            int(num_cuts * self.sel_cuts_percent), int(maxnselectedcuts))
        sel_cuts_num = max(sel_cuts_num, 2)
        cuts_features = advanced_cut_feature_generator(self.scip_model, cuts)
        input_cuts = mindspore.Tensor(cuts_features, mindspore.float32)
        input_cuts = input_cuts.reshape(
            input_cuts.shape[0], 1, input_cuts.shape[1])
        if self.decode_type == 'greedy':
            with mindspore.no_grad():
                pointer_probs, input_idxs = self.policy(
                    input_cuts, sel_cuts_num)
            baseline_value = 0.
        else:
            pointer_probs, input_idxs = self.policy(input_cuts, sel_cuts_num)
            if self.baseline_type == 'net':
                baseline_value = self.value(input_cuts)
            else:
                baseline_value = 0.
        idxes = [input_idx.asnumpy()[0] for input_idx in input_idxs]

        print(idxes)
        idxes = set(idxes)
        all_idxes = list(range(num_cuts))
        not_sel_idxes = list(set(all_idxes).difference(idxes))
        sorted_cuts = [cuts[idx] for idx in idxes]
        not_sel_cuts = [cuts[n_idx] for n_idx in not_sel_idxes]
        sorted_cuts.extend(not_sel_cuts)
        pointer_probs_list = [prob.asnumpy()[:, idx]
                              for prob, idx in zip(pointer_probs, idxes)]
        raw_seq_pointer_probs = [pointer_prob.asnumpy()
                                 for pointer_prob in pointer_probs]
        self.data = {
            "raw_cuts": cuts_features,
            "len_raw_cuts": num_cuts,
            "selected_idx": idxes,
            "pointer_probs": pointer_probs_list,
            "raw_seq_pointer_probs": raw_seq_pointer_probs,
            "baseline_value": baseline_value
        }

        return {
            'cuts': sorted_cuts,
            'nselectedcuts': sel_cuts_num,
            'result': SCIP_RESULT.SUCCESS
        }

    def _get_lp_info(self):
        lp_info = {}
        lp_info['lp_solution_value'] = self.scip_model.getLPObjVal()
        cols = self.scip_model.getLPColsData()
        lp_solution_integer = [
            col.getPrimsol() for col in cols if col.isIntegral()]
        lp_info['lp_solution_integer_var_value'] = lp_solution_integer

        return lp_info

    def get_data(self):
        return self.data

    def free_problem(self):
        self.scip_model.freeProb()
