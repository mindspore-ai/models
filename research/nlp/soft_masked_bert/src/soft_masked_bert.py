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
"""Sofr-Masked BERT"""
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import mindspore as ms
from mindspore import nn
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype
from src.bert_model import BertTransformer, SaturateCast, BertOnlyMLMHead, EmbeddingLookup, EmbeddingPostprocessor, CreateAttentionMaskFromInputMask
from src.finetune_config import bert_cfg, gru_cfg
from src.gru import BidirectionGRU

class DetectionNetwork(nn.Cell):
    def __init__(self, config, batch_size, is_training, if_O3):
        super().__init__()
        self.config = config
        self.rnn = BidirectionGRU(gru_cfg, batch_size)
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Dense(self.config.hidden_size, 1)
        self.cast = P.Cast()
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.O3 = if_O3

    def construct(self, hidden_states):
        if self.O3 is False:
            hidden_states = self.cast(hidden_states, ms.float16) # if not O3
        hidden_states = self.transpose(hidden_states, (1, 0, 2))
        out, _ = self.rnn(hidden_states)
        out = self.transpose(out, (1, 0, 2))
        if self.O3:
            prob = self.linear(out) # if O3
        else:
            prob = self.linear(out.astype("float32")) # if not O3
        prob = self.sigmoid(prob)
        return prob

class BertEmbedding(nn.Cell):
    def __init__(self, config, load_checkpoint_path):
        super(BertEmbedding, self).__init__()
        self.config = config
        self.bert_embedding_lookup = EmbeddingLookup(
            vocab_size=self.config.vocab_size,
            embedding_size=self.config.hidden_size,
            embedding_shape=[-1, self.config.seq_length, self.config.hidden_size],
            use_one_hot_embeddings=False,
            initializer_range=self.config.initializer_range)

        self.bert_embedding_postprocessor = EmbeddingPostprocessor(
            embedding_size=self.config.hidden_size,
            embedding_shape=[-1, self.config.seq_length, self.config.hidden_size],
            use_relative_positions=self.config.use_relative_positions,
            use_token_type=True,
            token_type_vocab_size=self.config.type_vocab_size,
            use_one_hot_embeddings=False,
            initializer_range=0.02,
            max_position_embeddings=self.config.seq_length,
            dropout_prob=self.config.hidden_dropout_prob)

    def construct(self, sentence_tokens, token_type_ids):
        word_embeddings, _ = self.bert_embedding_lookup(sentence_tokens)
        embed = self.bert_embedding_postprocessor(token_type_ids, word_embeddings)
        return embed


class BertCorrectionModel(nn.Cell):
    def __init__(self, config, batch_size, embbedding, param_dict, pretrained):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        corrector = BertTransformer(hidden_size=self.config.hidden_size,
                                    seq_length=self.config.seq_length,
                                    num_hidden_layers=self.config.num_hidden_layers,
                                    hidden_dropout_prob=self.config.hidden_dropout_prob,
                                    attention_probs_dropout_prob=self.config.attention_probs_dropout_prob)
        if pretrained is True:
            load_param_into_net(corrector, param_dict)
        self.corrector = corrector
        self.mask_token_id = 103 # id of the [MASK] token
        self.batch_size = batch_size
        self._create_attention_mask_from_input_mask = CreateAttentionMaskFromInputMask(config, self.batch_size) #?
        self.cast_compute_type = SaturateCast(dst_type=config.compute_type) #?
        # TODO: test whether has to use cls pretrained parameter.
        cls = BertOnlyMLMHead(self.config, param_dict, pretrained)
        self.cls = cls
        self.cast = P.Cast()
        self.oneslike = P.OnesLike()
        self.squeeze = P.Squeeze(-1)
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.loss_fct = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='none')
        self.reshape = P.Reshape()

    def construct(self, prob, embed, mask_embed, cor_labels, original_tokens_mask, original_token_type_ids, \
    batch_max_len, original_batch_len, total_seq_len, batch_size, residual_connection=True):
        cor_embed = prob * mask_embed + (1 - prob) * embed
        # get attention mask
        # use 'original_tokens_mask'
        attention_mask = self._create_attention_mask_from_input_mask(original_tokens_mask)
        encoder_outputs = self.corrector(self.cast_compute_type(cor_embed), attention_mask)
        sequence_output = encoder_outputs[0]
        sequence_output = sequence_output + embed if residual_connection else sequence_output
        prediction_scores = self.cls(sequence_output)
        out = (prediction_scores, sequence_output)
        if cor_labels is not None:
            cor_labels[cor_labels == 0] = -100
            prediction_scores_rp = self.reshape(prediction_scores, (-1, self.vocab_size))
            cor_labels_rp = self.reshape(cor_labels, (-1,))
            cor_loss = self.loss_fct(prediction_scores_rp, cor_labels_rp)
            cor_loss = cor_loss * original_tokens_mask.view(-1)
            cor_loss = self.reduce_sum(cor_loss)
            cor_loss = cor_loss / total_seq_len
            out = (cor_loss,) + out
        return out


class SoftMaskedBertCLS(nn.Cell):
    def __init__(self, batch_size, is_training=True, if_O3=True, \
    load_checkpoint_path="./weight/bert_base.ckpt", pretrained=False):
        super(SoftMaskedBertCLS, self).__init__()
        self.batch_size = batch_size
        self.config = bert_cfg
        self.detector = DetectionNetwork(self.config, batch_size, is_training=True, if_O3=if_O3)
        self.mask_token_id = 103  # id of the [MASK] token
        self.oneslike = P.OnesLike()
        self.squeeze2 = P.Squeeze(-1)
        embedding = BertEmbedding(self.config, load_checkpoint_path)
        param_dict = load_checkpoint(load_checkpoint_path)
        if pretrained is True:
            load_param_into_net(embedding, param_dict)
        self.embedding = embedding
        # correction
        self.corrector = BertCorrectionModel(self.config, self.batch_size, \
        self.embedding, param_dict, pretrained=pretrained)
        self.reshape = P.Reshape()
        self.maskedselect = P.MaskedSelect()
        self.expand_dims = P.ExpandDims()
        self.cast = P.Cast()
        self.reduce_sum1 = P.ReduceSum(keep_dims=True)
        self.reduce_sum2 = P.ReduceSum(keep_dims=False)
        self.loss = nn.BCELoss(reduction='none')
        self.squeeze = P.Squeeze(2)
        self.linear = nn.Dense(512, 512) # for debug
        self.w = 0.8
        self.linear_debug = nn.Dense(768, 1) # for debug
        self.linear_debug2 = nn.Dense(512, 512) # for debug
        self.select = P.Select()
        self.is_training = is_training
        self.print = P.Print() # for debug
        self.argmax = P.Argmax() # for debug
    # ['wrong_ids', 'original_tokens', 'original_tokens_mask', 'correct_tokens', 'correct_tokens_mask',
    # 'original_token_type_ids', 'correct_token_type_ids']
    def construct(self, *inputs):
        det_labels = inputs[0]
        original_tokens = inputs[1]
        original_tokens_mask = inputs[2]
        correct_tokens = inputs[3]
        original_token_type_ids = inputs[5]
        input_shape = original_token_type_ids.shape
        embed = self.embedding(original_tokens, original_token_type_ids) # 3 matmul
        prob = self.detector(embed)
        mask_embed = self.embedding(self.cast(self.oneslike(self.squeeze2(prob)), mstype.int32) * self.mask_token_id,
                                    original_token_type_ids)
        active_loss = self.reshape(original_tokens_mask, (-1, prob.shape[1]))
        batch_seq_len = self.reduce_sum1(active_loss.astype("float32"), 1)
        batch_max_len = batch_seq_len.max()
        original_batch_len = input_shape[1]
        total_seq_len = self.reduce_sum2(batch_seq_len)
        cor_out = self.corrector(prob, embed, mask_embed, correct_tokens, original_tokens_mask, \
        original_token_type_ids, batch_max_len, original_batch_len, total_seq_len, self.batch_size, \
        residual_connection=False)
        prob_ = self.reshape(prob, (-1, prob.shape[1]))
        prob = prob_.astype(mstype.float32)
        det_labels = det_labels.astype(mstype.float32)
        det_loss = self.loss(prob, det_labels)
        det_loss = det_loss * original_tokens_mask
        det_loss = self.reduce_sum2(det_loss)
        det_loss = det_loss / total_seq_len
        outputs = (det_loss, cor_out[0], prob) + cor_out[1:]
        loss = self.w * outputs[1] + (1 - self.w) * outputs[0]
        det_y_hat = (outputs[2] > 0.5).astype("int32")
        cor_y_hat = self.argmax(outputs[3])
        if self.is_training:
            res = loss
        else:
            det_y_hat = (outputs[2] > 0.5).astype("int32")
            cor_y_hat = self.argmax(outputs[3])
            cor_y = correct_tokens
            cor_y_hat *= original_tokens_mask
            res = (original_tokens, cor_y, cor_y_hat, det_y_hat, det_labels, batch_seq_len)
        return res
