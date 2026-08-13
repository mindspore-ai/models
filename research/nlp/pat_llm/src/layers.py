"""Core MindSpore layers used by PAT-LLM."""

# Neural-network cells expose several hyperparameters and child cells by
# design; keeping them explicit makes the architecture auditable.
# pylint: disable=too-many-arguments,too-many-instance-attributes

import math
from math import sqrt

import mindspore.common.dtype as mstype
from mindspore import Parameter, Tensor, nn, ops
import numpy as np


class PatchEmbedding(nn.Cell):
    """Tokenise each variable into overlapping local patches."""

    def __init__(self, seq_len, d_model, patch_len, stride, dropout=0.1):
        super().__init__()
        if patch_len <= 0 or stride <= 0:
            raise ValueError("patch_len and stride must be positive")
        if patch_len > seq_len + stride:
            raise ValueError("patch_len exceeds the padded sequence length")
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = (seq_len - patch_len) // stride + 2
        self.projection = nn.Dense(patch_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def construct(self, values):
        """Embed a batch shaped [batch, variables, time]."""
        if values.ndim != 3:
            raise ValueError("PatchEmbedding expects [batch, variables, time]")
        padding = ops.tile(values[:, :, -1:], (1, 1, self.stride))
        padded = ops.concat((values, padding), axis=-1)
        patches = tuple(
            padded[:, :, start : start + self.patch_len]
            for start in range(
                0, self.n_patches * self.stride, self.stride
            )
        )
        return self.dropout(self.projection(ops.stack(patches, axis=2)))


class ProjectedMultiheadAttention(nn.Cell):
    """Attention with projected heads for non-divisible model dimensions."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        if d_model <= 0 or n_heads <= 0:
            raise ValueError("d_model and n_heads must be positive")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        if self.d_head == 0:
            raise ValueError("n_heads cannot exceed d_model")
        projected_dim = n_heads * self.d_head
        self.query_projection = nn.Dense(d_model, projected_dim)
        self.key_projection = nn.Dense(d_model, projected_dim)
        self.value_projection = nn.Dense(d_model, projected_dim)
        self.output_projection = nn.Dense(projected_dim, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def construct(self, query, key, value):
        """Apply scaled dot-product attention and average head weights."""
        batch_size, query_length, _ = query.shape
        key_length = key.shape[1]
        query = ops.reshape(
            self.query_projection(query),
            (batch_size, query_length, self.n_heads, self.d_head),
        )
        key = ops.reshape(
            self.key_projection(key),
            (batch_size, key_length, self.n_heads, self.d_head),
        )
        value = ops.reshape(
            self.value_projection(value),
            (batch_size, key_length, self.n_heads, self.d_head),
        )
        query = ops.transpose(query, (0, 2, 1, 3))
        key = ops.transpose(key, (0, 2, 3, 1))
        value = ops.transpose(value, (0, 2, 1, 3))
        weights = ops.softmax(
            ops.matmul(query, key) / sqrt(self.d_head), axis=-1
        )
        output = ops.matmul(self.dropout(weights), value)
        output = ops.transpose(output, (0, 2, 1, 3))
        output = ops.reshape(
            output,
            (batch_size, query_length, self.n_heads * self.d_head),
        )
        return self.output_projection(output), ops.mean(weights, axis=1)


class FeedForward(nn.Cell):
    """Position-wise feed-forward block."""

    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.net = nn.SequentialCell(
            nn.Dense(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Dense(d_ff, d_model),
            nn.Dropout(p=dropout),
        )

    def construct(self, values):
        """Apply the position-wise network."""
        return self.net(values)


class MultiPatchAttention(nn.Cell):
    """Two-stage cross-variable and cross-temporal patch attention."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.stage1_attention = ProjectedMultiheadAttention(
            d_model, n_heads, dropout
        )
        self.stage1_attention_norm = nn.LayerNorm((d_model,))
        self.stage1_ffn = FeedForward(d_model, d_ff, dropout)
        self.stage1_ffn_norm = nn.LayerNorm((d_model,))
        self.stage2_attention = ProjectedMultiheadAttention(
            d_model, n_heads, dropout
        )
        self.stage2_attention_norm = nn.LayerNorm((d_model,))
        self.stage2_ffn = FeedForward(d_model, d_ff, dropout)
        self.stage2_ffn_norm = nn.LayerNorm((d_model,))
        self.dropout = nn.Dropout(p=dropout)

    def construct(self, patch_embeddings):
        """Refine patch embeddings and return both attention maps."""
        if patch_embeddings.ndim != 4:
            raise ValueError(
                "patch embeddings must be [batch, variables, patches, hidden]"
            )
        batch_size, n_vars, n_patches, d_model = patch_embeddings.shape
        flattened = ops.reshape(
            patch_embeddings, (batch_size, n_vars * n_patches, d_model)
        )
        latest = patch_embeddings[:, :, -1, :]
        variable_context, stage1_weights = self.stage1_attention(
            latest, flattened, flattened
        )
        variable_context = self.stage1_attention_norm(
            latest + self.dropout(variable_context)
        )
        variable_context = self.stage1_ffn_norm(
            variable_context + self.stage1_ffn(variable_context)
        )
        refined, stage2_weights = self.stage2_attention(
            flattened, variable_context, variable_context
        )
        refined = self.stage2_attention_norm(
            flattened + self.dropout(refined)
        )
        refined = self.stage2_ffn_norm(refined + self.stage2_ffn(refined))
        refined = ops.reshape(
            refined, (batch_size, n_vars, n_patches, d_model)
        )
        return refined, stage1_weights, stage2_weights


class AttentionPromptFeatures(nn.Cell):
    """Extract important variables and patches from MPA attention maps."""

    def __init__(
        self, n_vars, n_patches, top_k=5, top_m=3, recent_fraction=0.2
    ):
        super().__init__()
        self.n_vars = n_vars
        self.n_patches = n_patches
        self.top_k = min(top_k, n_vars)
        self.top_m = min(top_m, n_patches)
        self.recent_count = max(
            1, math.ceil(n_patches * recent_fraction)
        )
        cross_mask = np.ones((n_vars, n_vars), dtype=np.float32)
        np.fill_diagonal(cross_mask, 0.0)
        self.cross_mask = Tensor(cross_mask, mstype.float32)

    def construct(self, stage1_attention, stage2_attention):
        """Select important variables, patches, and recent attention."""
        batch_size = stage1_attention.shape[0]
        stage1 = ops.reshape(
            stage1_attention,
            (batch_size, self.n_vars, self.n_vars, self.n_patches),
        )
        stage1 = ops.transpose(stage1, (0, 1, 3, 2))
        cross_variable = ops.sum(stage1, dim=2)
        cross_variable = cross_variable * ops.cast(
            self.cross_mask, cross_variable.dtype
        )
        variable_importance = ops.sum(cross_variable, dim=1)

        stage2 = ops.reshape(
            stage2_attention,
            (batch_size, self.n_vars, self.n_patches, self.n_vars),
        )
        temporal = ops.diagonal(stage2, dim1=1, dim2=3)
        temporal = ops.transpose(temporal, (0, 2, 1))
        temporal_importance = ops.mean(temporal, dim=1)
        recent_weight = ops.mean(
            temporal_importance[:, -self.recent_count :], dim=1
        )
        _, top_variables = ops.topk(
            variable_importance, self.top_k, dim=-1
        )
        _, top_patches = ops.topk(
            temporal_importance, self.top_m, dim=-1
        )
        return top_variables, top_patches, recent_weight


class AttentionAsPrompt:
    """Textualise attention features on the Python host."""

    def __init__(self, variable_names, n_patches):
        self.variable_names = list(variable_names)
        self.n_patches = n_patches

    def __call__(self, top_variables, top_patches, recent_weight):
        variable_indices = top_variables.asnumpy().tolist()
        patch_indices = top_patches.asnumpy().tolist()
        recent_values = recent_weight.asnumpy().astype(float).tolist()
        prompts = []
        for batch_index, selected_variables in enumerate(variable_indices):
            influential = [
                self.variable_names[index] for index in selected_variables
            ]
            periods = []
            for patch_index in patch_indices[batch_index]:
                label = self._period_label(patch_index)
                if label not in periods:
                    periods.append(label)
            prompts.append(
                "Attention-derived structure: top influential variables: "
                f"{', '.join(influential)}; focus periods: "
                f"{', '.join(periods)} (recent weight: "
                f"{recent_values[batch_index]:.4f})."
            )
        return prompts

    def _period_label(self, patch_index):
        if self.n_patches == 1:
            return "very-recent"
        relative_position = patch_index / (self.n_patches - 1)
        if relative_position >= 0.8:
            return "very-recent"
        if relative_position >= 0.6:
            return "recent"
        if relative_position >= 0.4:
            return "mid-term"
        if relative_position >= 0.2:
            return "early"
        return "initial"


class ReprogrammingLayer(nn.Cell):
    """Map temporal tokens into the frozen LLM embedding space."""

    def __init__(
        self, d_model, n_heads, d_llm, d_keys=None, dropout=0.1
    ):
        super().__init__()
        self.d_keys = d_keys or d_model // n_heads
        self.n_heads = n_heads
        projected_dim = self.d_keys * n_heads
        self.query_projection = nn.Dense(d_model, projected_dim)
        self.key_projection = nn.Dense(d_llm, projected_dim)
        self.value_projection = nn.Dense(d_llm, projected_dim)
        self.output_projection = nn.Dense(projected_dim, d_llm)
        self.dropout = nn.Dropout(p=dropout)

    def construct(self, target, source, value):
        """Cross-attend temporal queries to vocabulary prototypes."""
        batch_size, target_length, _ = target.shape
        source_length = source.shape[0]
        query = ops.reshape(
            self.query_projection(target),
            (batch_size, target_length, self.n_heads, self.d_keys),
        )
        key = ops.reshape(
            self.key_projection(source),
            (source_length, self.n_heads, self.d_keys),
        )
        value = ops.reshape(
            self.value_projection(value),
            (source_length, self.n_heads, self.d_keys),
        )
        query = ops.transpose(query, (0, 2, 1, 3))
        key = ops.transpose(key, (1, 2, 0))
        attention = ops.softmax(
            ops.matmul(query, key) / sqrt(self.d_keys), axis=-1
        )
        value = ops.transpose(value, (1, 0, 2))
        output = ops.matmul(self.dropout(attention), value)
        output = ops.transpose(output, (0, 2, 1, 3))
        output = ops.reshape(
            output,
            (batch_size, target_length, self.n_heads * self.d_keys),
        )
        return self.output_projection(output)


class FlattenHead(nn.Cell):
    """Project LLM features from every variable to the forecast horizon."""

    def __init__(self, n_patches, d_ff, target_window, dropout=0.0):
        super().__init__()
        self.input_size = n_patches * d_ff
        self.projection = nn.Dense(self.input_size, target_window)
        self.dropout = nn.Dropout(p=dropout)

    def construct(self, values):
        """Flatten per-variable features and predict the horizon."""
        batch_size, n_vars, _, _ = values.shape
        values = ops.reshape(
            values, (batch_size, n_vars, self.input_size)
        )
        return self.dropout(self.projection(values))


class Normalize(nn.Cell):
    """Reversible instance normalisation."""

    def __init__(self, num_features, eps=1e-5, affine=False):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = Parameter(
                ops.ones((num_features,), mstype.float32),
                name="affine_weight",
            )
            self.affine_bias = Parameter(
                ops.zeros((num_features,), mstype.float32),
                name="affine_bias",
            )

    def construct(self, values):
        """Normalise each sample along its temporal axis."""
        mean = ops.stop_gradient(ops.mean(values, axis=1, keep_dims=True))
        variance = ops.mean(
            ops.square(values - mean), axis=1, keep_dims=True
        )
        stdev = ops.stop_gradient(ops.sqrt(variance + self.eps))
        normalised = (values - mean) / stdev
        if self.affine:
            normalised = (
                normalised * self.affine_weight + self.affine_bias
            )
        return normalised, mean, stdev

    def denormalize(self, values, mean, stdev):
        """Restore the prediction to the original per-sample scale."""
        if self.affine:
            values = (values - self.affine_bias) / (
                self.affine_weight + self.eps * self.eps
            )
        return values * stdev + mean
