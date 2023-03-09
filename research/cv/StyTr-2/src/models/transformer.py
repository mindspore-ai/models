import math
import copy
from typing import Optional
import numpy as np
import mindspore
from mindspore.common.initializer import Initializer, initializer, XavierUniform
from mindspore.common.initializer import _assignment
from mindspore.common.initializer import _calculate_correct_fan
from mindspore.common.initializer import _calculate_gain
from mindspore import nn, ops, Tensor, Parameter
from mindspore import numpy as mnp


class Transformer(nn.Cell):
    def __init__(self, model_dim=512, heads=8, num_encoder_layers=3,
                 num_decoder_layers=3, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layers = TransformerEncoderLayer(model_dim, heads, dim_feedforward,
                                                 dropout, activation, normalize_before)
        encoder_ln = nn.LayerNorm([model_dim]) if normalize_before else None
        self.encoder_c = TransformerEncoder(encoder_layers, num_encoder_layers, encoder_ln)
        self.encoder_s = TransformerEncoder(encoder_layers, num_encoder_layers, encoder_ln)

        decoder_layers = TransformerDecoderLayer(model_dim, heads, dim_feedforward,
                                                 dropout, activation, normalize_before)
        decoder_ln = nn.LayerNorm([model_dim])
        self.decoder = TransformerDecoder(decoder_layers, num_decoder_layers, decoder_ln,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.new_ps = nn.Conv2d(512, 512, (1, 1), has_bias=True, pad_mode='valid')
        self.averagepooling = ops.AdaptiveAvgPool2D(18)
        self.interpolate = nn.ResizeBilinear()
        self.transpose = ops.Transpose()

    def _reset_parameters(self):
        for p in self.get_parameters():
            if p.dim() > 1:
                p = initializer(XavierUniform(), p.shape, mindspore.float32)

    def construct(self, style, mask, content, pos_embed_c, pos_embed_s):
        # content-aware positional embedding

        content_pool = self.averagepooling(content)

        pos_c = self.new_ps(content_pool)

        pos_embed_c = self.interpolate(pos_c, size=style.shape[-2:])

        ###flatten NxCxHxW to HWxNxC

        style = style.view(style.shape[0], style.shape[1], -1)
        style = self.transpose(style, (2, 0, 1))

        if pos_embed_s is not None:
            pos_embed_s = pos_embed_s.view(pos_embed_s.shape[0], pos_embed_s.shape[1], -1)
            pos_embed_s = self.transpose(pos_embed_s, (2, 0, 1))

        content = content.view(content.shape[0], content.shape[1], -1)
        content = self.transpose(content, (2, 0, 1))
        if pos_embed_c is not None:
            pos_embed_c = pos_embed_c.view(pos_embed_c.shape[0], pos_embed_c.shape[1], -1)
            pos_embed_c = self.transpose(pos_embed_c, (2, 0, 1))
        style = self.encoder_s(style, src_key_padding_mask=mask, position=pos_embed_s)
        content = self.encoder_c(content, src_key_padding_mask=mask, position=pos_embed_c)
        hs = self.decoder(content, style, memory_key_padding_mask=mask,
                          position=pos_embed_s, query_position=pos_embed_c)[0]
        ### HWxNxC to NxCxHxW to
        N, B, C = hs.shape
        H = int(np.sqrt(N))
        hs = self.transpose(hs, (1, 2, 0))
        hs = hs.view(B, C, -1, H)
        return hs


class TransformerEncoder(nn.Cell):
    def __init__(self, encoder_layers, num_layers, norm=None):
        super().__init__()
        self.layers = get_clone(encoder_layers, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def construct(self, source,
                  mask: Optional[Tensor] = None,
                  src_key_padding_mask: Optional[Tensor] = None,
                  position: Optional[Tensor] = None):
        out = source

        for layer in self.layers:
            out = layer(out, src_mask=mask,
                        src_key_padding_mask=src_key_padding_mask, position=position)

        if self.norm is not None:
            out = self.norm(out)

        return out


class TransformerDecoder(nn.Cell):
    def __init__(self, decoder_layers, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = get_clone(decoder_layers, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_mediate = return_intermediate
        self.stack = ops.Stack()
        self.expand_dims = ops.ExpandDims()

    def construct(self, target, mem, tgt_mask: Optional[Tensor] = None,
                  memory_mask: Optional[Tensor] = None,
                  tgt_key_padding_mask: Optional[Tensor] = None,
                  memory_key_padding_mask: Optional[Tensor] = None,
                  position: Optional[Tensor] = None,
                  query_position: Optional[Tensor] = None):
        out = target

        mediate = []

        for Layer in self.layers:
            out = Layer(out, mem, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask,
                        position, query_position)
            if self.return_mediate:
                mediate.append(self.norm(out))

        if self.norm is not None:
            out = self.norm(out)
            if self.return_mediate:
                mediate.pop()
                mediate.append(out)

        if self.return_mediate:
            return self.stack(mediate)

        out = self.expand_dims(out, 0)
        return out


class TransformerEncoderLayer(nn.Cell):
    def __init__(self, model_dim, heads, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = MultiheadAttention(model_dim, heads, dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Dense(model_dim, dim_feedforward)
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Dense(dim_feedforward, model_dim)

        self.norm1 = nn.LayerNorm([model_dim])
        self.norm2 = nn.LayerNorm([model_dim])
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, position: Optional[Tensor]):
        return tensor if position is None else tensor + position

    def construct_post(self,
                       source,
                       src_mask: Optional[Tensor] = None,
                       src_key_padding_mask: Optional[Tensor] = None,
                       position: Optional[Tensor] = None):
        q = k = self.with_pos_embed(source, position)
        source2 = self.self_attn(q, k, value=source, key_padding_mask=src_key_padding_mask)

        source = source + self.dropout1(source2)
        source = self.norm1(source)
        source2 = self.linear2(self.dropout(self.activation(self.linear1(source))))
        source = source + self.dropout2(source2)
        source = self.norm2(source)
        return source

    def construct_before(self, source,
                         src_mask: Optional[Tensor] = None,
                         src_key_padding_mask: Optional[Tensor] = None,
                         position: Optional[Tensor] = None):
        source2 = self.norm1(source)
        q = k = self.with_pos_embed(source2, position)
        source2 = self.self_attn(q, k, value=source2, key_padding_mask=src_key_padding_mask)

        source = source + self.dropout1(source2)
        source2 = self.norm2(source)
        source2 = self.linear2(self.dropout(self.activation(self.linear1(source2))))
        source = source + self.dropout2(source2)
        return source

    def construct(self, source,
                  src_mask: Optional[Tensor] = None,
                  src_key_padding_mask: Optional[Tensor] = None,
                  position: Optional[Tensor] = None):
        if self.normalize_before:
            return self.construct_before(source, src_mask, src_key_padding_mask, position)
        return self.construct_post(source, src_mask, src_key_padding_mask, position)


class TransformerDecoderLayer(nn.Cell):
    def __init__(self, model_dim, heads, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # model_dim embedding dim
        self.self_attn = MultiheadAttention(model_dim, heads, dropout)
        self.multihead_attn = MultiheadAttention(model_dim, heads, dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Dense(model_dim, dim_feedforward)
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Dense(dim_feedforward, model_dim)

        self.norm1 = nn.LayerNorm([model_dim])
        self.norm2 = nn.LayerNorm([model_dim])
        self.norm3 = nn.LayerNorm([model_dim])
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.dropout3 = nn.Dropout(p=dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, position: Optional[Tensor]):
        return tensor if position is None else tensor + position

    def construct_post(self, target, mem,
                       tgt_mask: Optional[Tensor] = None,
                       memory_mask: Optional[Tensor] = None,
                       tgt_key_padding_mask: Optional[Tensor] = None,
                       memory_key_padding_mask: Optional[Tensor] = None,
                       position: Optional[Tensor] = None,
                       query_position: Optional[Tensor] = None):
        q = self.with_pos_embed(target, query_position)
        k = self.with_pos_embed(mem, position)
        v = mem
        target2 = self.self_attn(q, k, v, key_padding_mask=tgt_key_padding_mask)
        target = target + self.dropout1(target2)
        target = self.norm1(target)
        target2 = self.multihead_attn(query=self.with_pos_embed(target, query_position),
                                      key=self.with_pos_embed(mem, position),
                                      value=mem, key_padding_mask=memory_key_padding_mask)

        target = target + self.dropout2(target2)
        target = self.norm2(target)
        target2 = self.linear2(self.dropout(self.activation(self.linear1(target))))
        target = target + self.dropout3(target2)
        target = self.norm3(target)
        return target

    def construct_before(self, target, mem,
                         tgt_mask: Optional[Tensor] = None,
                         memory_mask: Optional[Tensor] = None,
                         tgt_key_padding_mask: Optional[Tensor] = None,
                         memory_key_padding_mask: Optional[Tensor] = None,
                         position: Optional[Tensor] = None,
                         query_position: Optional[Tensor] = None):
        target2 = self.norm1(target)
        q = k = self.with_pos_embed(target2, query_position)
        target2 = self.self_attn(q, k, value=target2, key_padding_mask=tgt_key_padding_mask)
        target = target + self.dropout1(target2)
        target2 = self.norm2(target)
        target2 = self.multihead_attn(query=self.with_pos_embed(target2, query_position),
                                      key=self.with_pos_embed(mem, position),
                                      value=mem, key_padding_mask=memory_key_padding_mask)
        target = target + self.dropout2(target2)
        target2 = self.norm3(target)
        target2 = self.linear2(self.dropout(self.activation(self.linear1(target2))))
        target = target + self.dropout3(target2)
        return target

    def construct(self, target, mem,
                  tgt_mask: Optional[Tensor] = None,
                  memory_mask: Optional[Tensor] = None,
                  tgt_key_padding_mask: Optional[Tensor] = None,
                  memory_key_padding_mask: Optional[Tensor] = None,
                  position: Optional[Tensor] = None,
                  query_position: Optional[Tensor] = None):
        if self.normalize_before:
            return self.construct_before(target, mem, tgt_mask, memory_mask,
                                         tgt_key_padding_mask, memory_key_padding_mask, position, query_position)
        return self.construct_post(target, mem, tgt_mask, memory_mask,
                                   tgt_key_padding_mask, memory_key_padding_mask, position, query_position)


class MultiheadAttention(nn.Cell):
    """multi head attention"""

    def __init__(self, embed_dim, num_heads, dropout=0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.q_in_proj_weight = Parameter(initializer('xavier_uniform',
                                                      [embed_dim, embed_dim],
                                                      mindspore.float32))
        self.k_in_proj_weight = Parameter(initializer('xavier_uniform',
                                                      [embed_dim, embed_dim],
                                                      mindspore.float32))
        self.v_in_proj_weight = Parameter(initializer('xavier_uniform',
                                                      [embed_dim, embed_dim],
                                                      mindspore.float32))

        self.q_in_proj_bias = Parameter(initializer('zeros',
                                                    [embed_dim],
                                                    mindspore.float32))
        self.k_in_proj_bias = Parameter(initializer('zeros',
                                                    [embed_dim],
                                                    mindspore.float32))
        self.v_in_proj_bias = Parameter(initializer('zeros',
                                                    [embed_dim],
                                                    mindspore.float32))

        self.out_proj = nn.Dense(embed_dim, embed_dim, weight_init=KaimingUniform())
        self.drop = nn.Dropout(p=dropout)
        self.BatchMatMul = ops.BatchMatMul()
        self.Tile = ops.Tile()
        self.expand_dims = ops.ExpandDims()
        self.softmax = nn.Softmax(axis=-1)

    def construct(self,
                  query: Tensor,
                  key: Tensor,
                  value: Tensor,
                  key_padding_mask: Tensor,
                  need_weights: bool = True):
        """construct"""
        tgt_len, bsz, embed_dim = query.shape
        scaling = self.head_dim ** -0.5

        q = linear(query, self.q_in_proj_weight, self.q_in_proj_bias)
        k = linear(key, self.k_in_proj_weight, self.k_in_proj_bias)
        v = linear(value, self.v_in_proj_weight, self.v_in_proj_bias)

        q = q * scaling

        q = q.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(1, 0, 2)
        k = k.view(-1, bsz * self.num_heads, self.head_dim).transpose(1, 0, 2)
        v = v.view(-1, bsz * self.num_heads, self.head_dim).transpose(1, 0, 2)

        src_len = k.shape[1]

        attn_output_weights = self.BatchMatMul(q, k.transpose(0, 2, 1))

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            key_padding_mask = self.Tile(
                self.expand_dims(self.expand_dims(key_padding_mask, 1), 2),
                (1, self.num_heads, tgt_len, 1)
            )
            attn_output_weights = attn_output_weights - key_padding_mask * 10000.
            attn_output_weights = attn_output_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_output_weights = self.softmax(attn_output_weights)
        attn_output_weights = self.drop(attn_output_weights)

        attn_output = self.BatchMatMul(attn_output_weights, v)
        attn_output = attn_output.transpose(1, 0, 2).view(tgt_len, bsz, embed_dim)
        attn_output = linear(attn_output, self.out_proj.weight, self.out_proj.bias)
        return attn_output


class KaimingUniform(Initializer):
    """
    Initialize the array with He kaiming algorithm.
    Args:
        a: the negative slope of the rectifier used after this layer (only
            used with ``'leaky_relu'``)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function, recommended to use only with
            ``'relu'`` or ``'leaky_relu'`` (default).
    """

    def __init__(self, a=math.sqrt(5), mode='fan_in', nonlinearity='leaky_relu'):
        super().__init__()
        self.mode = mode
        self.gain = _calculate_gain(nonlinearity, a)

    def _initialize(self, arr):
        fan = _calculate_correct_fan(arr.shape, self.mode)
        bound = math.sqrt(3.0) * self.gain / math.sqrt(fan)
        data = np.random.uniform(-bound, bound, arr.shape)
        _assignment(arr, data)


def linear(input_arr, weight, bias=None):
    """
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.

    Shape:

        - Input: :math:`(N, *, in_features)` N is the batch size, `*` means any number of
          additional dimensions
        - Weight: :math:`(out_features, in_features)`
        - Bias: :math:`(out_features)`
        - Output: :math:`(N, *, out_features)`
    """
    if input_arr.ndim == 2 and bias is not None:
        # fused op is marginally faster
        ret = ops.BatchMatMul()(input_arr, weight.T) + bias
    else:
        output = mnp.matmul(input_arr, weight.T)
        if bias is not None:
            output += bias
        ret = output
    return ret


def get_clone(cell, N):
    return nn.CellList([copy.deepcopy(cell) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.ReLU()
    # if activation == "gelu":
    #     return F.gelu
    # if activation == "glu":
    #     return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
