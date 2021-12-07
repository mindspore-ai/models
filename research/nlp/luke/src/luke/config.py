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
"""luke config"""
import copy

roberta_base_config = {
    "architectures": [
        "RobertaForMaskedLM"
    ],
    "attention_probs_dropout_prob": 0.1,
    "bos_token_id": 0,
    "do_sample": False,
    "eos_token_id": 2,
    "eos_token_ids": 0,
    "finetuning_task": None,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 768,
    "id2label": {
        "0": "LABEL_0",
        "1": "LABEL_1"
    },
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "is_decoder": False,
    "label2id": {
        "LABEL_0": 0,
        "LABEL_1": 1
    },
    "layer_norm_eps": 1e-05,
    "length_penalty": 1.0,
    "max_length": 20,
    "max_position_embeddings": 514,
    "model_type": "roberta",
    "num_attention_heads": 12,
    "num_beams": 1,
    "num_hidden_layers": 12,
    "num_labels": 2,
    "num_return_sequences": 1,
    "output_attentions": False,
    "output_hidden_states": False,
    "output_past": True,
    "pad_token_id": 1,
    "pruned_heads": {},
    "repetition_penalty": 1.0,
    "temperature": 1.0,
    "top_k": 50,
    "top_p": 1.0,
    "torchscript": False,
    "type_vocab_size": 1,
    "use_bfloat16": False,
    "vocab_size": 50265
}
roberta_large_config = {
    'output_attentions': False,
    'output_hidden_states': False,
    'output_past': True,
    'torchscript': False,
    'use_bfloat16': False,
    'pruned_heads': {},
    'is_decoder': False,
    'max_length': 20,
    'do_sample': False,
    'num_beams': 1,
    'temperature': 1.0,
    'top_k': 50,
    'top_p': 1.0,
    'repetition_penalty': 1.0,
    'bos_token_id': 0,
    'pad_token_id': 1,
    'eos_token_ids': 0,
    'length_penalty': 1.0,
    'num_return_sequences': 1,
    'architectures': ['RobertaForMaskedLM'],
    'finetuning_task': None, 'num_labels': 2,
    'id2label': {0: 'LABEL_0', 1: 'LABEL_1'},
    'label2id': {'LABEL_0': 0, 'LABEL_1': 1},
    'eos_token_id': 2,
    'model_type': 'roberta',
    'vocab_size': 50265,
    'hidden_size': 1024,
    'num_hidden_layers': 24,
    'num_attention_heads': 16,
    'hidden_act': 'gelu',
    'intermediate_size': 4096,
    'hidden_dropout_prob': 0.1,
    'attention_probs_dropout_prob': 0.1,
    'max_position_embeddings': 514,
    'type_vocab_size': 1,
    'initializer_range': 0.02,
    'layer_norm_eps': 1e-05}


class PretrainedConfig:
    """pre train config"""

    def __init__(self, **kwargs):
        """init fun"""
        # Attributes with defaults
        self.output_attentions = kwargs.pop("output_attentions", False)
        self.output_hidden_states = kwargs.pop("output_hidden_states", False)
        self.output_past = kwargs.pop("output_past", True)  # Not used by all models
        self.torchscript = kwargs.pop("torchscript", False)  # Only used by PyTorch models
        self.use_bfloat16 = kwargs.pop("use_bfloat16", False)
        self.pruned_heads = kwargs.pop("pruned_heads", {})

        # Is decoder is used in encoder-decoder models to differentiate encoder from decoder
        self.is_decoder = kwargs.pop("is_decoder", False)

        # Parameters for sequence generation
        self.max_length = kwargs.pop("max_length", 20)
        self.do_sample = kwargs.pop("do_sample", False)
        self.num_beams = kwargs.pop("num_beams", 1)
        self.temperature = kwargs.pop("temperature", 1.0)
        self.top_k = kwargs.pop("top_k", 50)
        self.top_p = kwargs.pop("top_p", 1.0)
        self.repetition_penalty = kwargs.pop("repetition_penalty", 1.0)
        self.bos_token_id = kwargs.pop("bos_token_id", 0)
        self.pad_token_id = kwargs.pop("pad_token_id", 0)
        self.eos_token_ids = kwargs.pop("eos_token_ids", 0)
        self.length_penalty = kwargs.pop("length_penalty", 1.0)
        self.num_return_sequences = kwargs.pop("num_return_sequences", 1)

        # Fine-tuning task arguments
        self.architectures = kwargs.pop("architectures", None)
        self.finetuning_task = kwargs.pop("finetuning_task", None)
        self.num_labels = kwargs.pop("num_labels", 2)
        self.id2label = kwargs.pop("id2label", {i: "LABEL_{}".format(i) for i in range(self.num_labels)})
        self.id2label = dict((int(key), value) for key, value in self.id2label.items())
        self.label2id = kwargs.pop("label2id", dict(zip(self.id2label.values(), self.id2label.keys())))
        self.label2id = dict((key, int(value)) for key, value in self.label2id.items())

        # Additional attributes without default values
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                raise err


class BertConfig(PretrainedConfig):
    """bert config"""

    def __init__(
            self,
            vocab_size=30522,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=2,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            **kwargs
    ):
        """init fun"""
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps


class LukeConfig(BertConfig):
    """luke config"""

    def __init__(
            self, vocab_size: int, entity_vocab_size: int, bert_model_name: str, entity_emb_size: int = None, **kwargs
    ):
        """init fun"""
        super(LukeConfig, self).__init__(vocab_size, **kwargs)

        self.entity_vocab_size = entity_vocab_size
        self.bert_model_name = bert_model_name
        if entity_emb_size is None:
            self.entity_emb_size = self.hidden_size
        else:
            self.entity_emb_size = entity_emb_size

    def to_dict(self):
        """to dict"""
        output = copy.deepcopy(self.__dict__)
        return output
