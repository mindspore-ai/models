"""PAT-LLM network implemented with MindSpore and MindNLP."""

# PAT-LLM explicitly exposes its encoder, prompt, reprogramming, LLM, and
# output components. The forward pass also keeps named intermediate tensors
# to mirror the method described in the manuscript.
# pylint: disable=too-many-instance-attributes,too-many-locals

import mindspore.common.dtype as mstype
from mindspore import Tensor, nn, ops
from mindnlp.transformers import AutoConfig, AutoModel, AutoTokenizer
import numpy as np

from .layers import (
    AttentionAsPrompt,
    AttentionPromptFeatures,
    FlattenHead,
    MultiPatchAttention,
    Normalize,
    PatchEmbedding,
    ReprogrammingLayer,
)


class PATLLM(nn.Cell):
    """Textualized Multi-Patch Attention forecasting network."""

    def __init__(self, config):
        super().__init__()
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.n_vars = config.enc_in
        self.d_ff = config.d_ff
        self.description = config.description
        self.target_index = config.target_index
        self.variable_names = list(config.variable_names)

        self.patch_embedding = PatchEmbedding(
            config.seq_len,
            config.d_model,
            config.patch_len,
            config.stride,
            config.dropout,
        )
        self.n_patches = self.patch_embedding.n_patches
        self.multi_patch_attention = MultiPatchAttention(
            config.d_model,
            config.n_heads,
            config.d_ff,
            config.dropout,
        )
        self.prompt_features = AttentionPromptFeatures(
            self.n_vars,
            self.n_patches,
            config.aap_top_k,
            config.aap_top_m,
            config.aap_recent_fraction,
        )
        self.attention_as_prompt = AttentionAsPrompt(
            self.variable_names, self.n_patches
        )
        self.normalizer = Normalize(self.n_vars, affine=False)

        llm_config = AutoConfig.from_pretrained(
            config.llm_model_path,
            local_files_only=not config.allow_download,
            trust_remote_code=config.trust_remote_code,
        )
        if hasattr(llm_config, "num_hidden_layers"):
            llm_config.num_hidden_layers = config.llm_layers
        elif hasattr(llm_config, "n_layer"):
            llm_config.n_layer = config.llm_layers
        llm_config.output_hidden_states = True
        self.llm_model = AutoModel.from_pretrained(
            config.llm_model_path,
            config=llm_config,
            ms_dtype=getattr(mstype, config.llm_dtype),
            local_files_only=not config.allow_download,
            trust_remote_code=config.trust_remote_code,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.llm_model_path,
            local_files_only=not config.allow_download,
            trust_remote_code=config.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                self.llm_model.resize_token_embeddings(len(self.tokenizer))
        for parameter in self.llm_model.get_parameters():
            parameter.requires_grad = False

        embedding_layer = self.llm_model.get_input_embeddings()
        word_embeddings = self._embedding_weight(embedding_layer)
        self.vocab_size, self.d_llm = word_embeddings.shape
        if self.d_ff > self.d_llm:
            raise ValueError("d_ff cannot exceed the LLM hidden dimension")
        self.mapping_layer = nn.Dense(
            self.vocab_size, config.num_prototypes
        )
        self.reprogramming_layer = ReprogrammingLayer(
            config.d_model,
            config.n_heads,
            self.d_llm,
            dropout=config.dropout,
        )
        self.output_projection = FlattenHead(
            self.n_patches,
            self.d_ff,
            self.pred_len,
            config.dropout,
        )

    def set_train(self, mode=True):
        """Keep the frozen LLM in evaluation mode during PAT-LLM training."""
        super().set_train(mode)
        self.llm_model.set_train(False)
        return self

    def construct(self, x_enc):
        """Forecast a multivariate batch shaped [batch, time, variables]."""
        x_enc, mean, stdev = self.normalizer(x_enc)
        patches = self.patch_embedding(ops.transpose(x_enc, (0, 2, 1)))
        refined, stage1_attention, stage2_attention = (
            self.multi_patch_attention(patches)
        )
        top_variables, top_patches, recent_weight = self.prompt_features(
            stage1_attention, stage2_attention
        )
        structure_prompts = self.attention_as_prompt(
            top_variables, top_patches, recent_weight
        )
        prompt_embeddings, prompt_attention_mask = self._embed_prompts(
            self._build_prompts(x_enc, structure_prompts)
        )

        embedding_layer = self.llm_model.get_input_embeddings()
        word_embeddings = ops.cast(
            self._embedding_weight(embedding_layer), mstype.float32
        )
        prototypes = self.mapping_layer(
            ops.transpose(word_embeddings, (1, 0))
        )
        prototypes = ops.transpose(prototypes, (1, 0))
        batch_size = refined.shape[0]
        temporal_tokens = ops.reshape(
            refined,
            (batch_size, self.n_vars * self.n_patches, -1),
        )
        reprogrammed = self.reprogramming_layer(
            temporal_tokens, prototypes, prototypes
        )
        reprogrammed = ops.cast(reprogrammed, prompt_embeddings.dtype)
        llm_input = ops.concat((prompt_embeddings, reprogrammed), axis=1)
        temporal_attention_mask = ops.ones(
            (batch_size, self.n_vars * self.n_patches), mstype.int32
        )
        attention_mask = ops.concat(
            (prompt_attention_mask, temporal_attention_mask), axis=1
        )
        llm_output = self.llm_model(
            inputs_embeds=llm_input,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden_state = (
            llm_output.last_hidden_state
            if hasattr(llm_output, "last_hidden_state")
            else llm_output[0]
        )
        hidden_state = hidden_state[
            :, -self.n_vars * self.n_patches :, : self.d_ff
        ]
        hidden_state = ops.reshape(
            hidden_state,
            (batch_size, self.n_vars, self.n_patches, self.d_ff),
        )
        prediction = self.output_projection(hidden_state)
        prediction = ops.transpose(prediction, (0, 2, 1))
        return self.normalizer.denormalize(prediction, mean, stdev)

    def _build_prompts(self, x_enc, structure_prompts):
        values = x_enc.asnumpy()
        target = values[:, :, self.target_index]
        prompts = []
        for batch_index, series in enumerate(target):
            trend = "upward" if np.diff(series).sum() > 0 else "downward"
            lags = self._top_lags(series, top_k=5)
            prompts.append(
                "<|start_prompt|>"
                f"Dataset description: {self.description} "
                f"Task description: forecast the next {self.pred_len} steps "
                f"from the previous {self.seq_len} steps. "
                f"Target statistics: minimum {series.min():.6g}, "
                f"maximum {series.max():.6g}, "
                f"median {np.median(series):.6g}, trend {trend}, "
                f"top lags {lags}. {structure_prompts[batch_index]}"
                "<|end_prompt|>"
            )
        return prompts

    def _embed_prompts(self, prompts):
        tokenized = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="ms",
        )
        input_ids = tokenized["input_ids"]
        if not isinstance(input_ids, Tensor):
            input_ids = Tensor(np.asarray(input_ids), mstype.int32)
        attention_mask = tokenized.get("attention_mask")
        if attention_mask is None:
            attention_mask = ops.ones(input_ids.shape, mstype.int32)
        elif not isinstance(attention_mask, Tensor):
            attention_mask = Tensor(
                np.asarray(attention_mask), mstype.int32
            )
        else:
            attention_mask = ops.cast(attention_mask, mstype.int32)
        embeddings = self.llm_model.get_input_embeddings()(input_ids)
        return embeddings, attention_mask

    @staticmethod
    def _embedding_weight(embedding_layer):
        if hasattr(embedding_layer, "weight"):
            return embedding_layer.weight
        if hasattr(embedding_layer, "embedding_table"):
            return embedding_layer.embedding_table
        raise AttributeError("LLM input embedding has no accessible weight")

    @staticmethod
    def _top_lags(series, top_k):
        spectrum = np.fft.rfft(series)
        correlation = np.fft.irfft(
            spectrum * np.conj(spectrum), n=len(series)
        )
        count = min(top_k, len(correlation))
        return np.argsort(correlation)[-count:][::-1].tolist()
