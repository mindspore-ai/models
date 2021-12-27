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

"""
The tensorflow Albert parameter name corresponds to the dictionary of MindSpore albert
"""

trans_dict_tf = {
    'bert.embeddings.LayerNorm.beta':
        'albert.albert.albert_embedding_postprocessor.layernorm.beta',
    'bert.embeddings.LayerNorm.gamma':
        'albert.albert.albert_embedding_postprocessor.layernorm.gamma',
    'bert.embeddings.position_embeddings':
        'albert.albert.albert_embedding_postprocessor.full_position_embeddings',
    'bert.embeddings.token_type_embeddings':
        'albert.albert.albert_embedding_postprocessor.embedding_table',
    'bert.embeddings.word_embeddings': 'albert.albert.albert_embedding_lookup.embedding_table',
    'bert.encoder.embedding_hidden_mapping_in.bias':
        'albert.albert.albert_encoder.embedding_hidden_mapping_in.bias',
    'bert.encoder.embedding_hidden_mapping_in.kernel':
        'albert.albert.albert_encoder.embedding_hidden_mapping_in.weight',
    'bert.encoder.transformer.group_0.inner_group_0.LayerNorm.beta':
        'albert.albert.albert_encoder.group.0.inner_group.0.attention.output.layernorm.beta',
    'bert.encoder.transformer.group_0.inner_group_0.LayerNorm.gamma':
        'albert.albert.albert_encoder.group.0.inner_group.0.attention.output.layernorm.gamma',
    'bert.encoder.transformer.group_0.inner_group_0.LayerNorm_1.beta':
        'albert.albert.albert_encoder.group.0.inner_group.0.output.layernorm.beta',
    'bert.encoder.transformer.group_0.inner_group_0.LayerNorm_1.gamma':
        'albert.albert.albert_encoder.group.0.inner_group.0.output.layernorm.gamma',
    'bert.encoder.transformer.group_0.inner_group_0.attention_1.output.dense.bias':
        'albert.albert.albert_encoder.group.0.inner_group.0.attention.output.dense.bias',
    'bert.encoder.transformer.group_0.inner_group_0.attention_1.output.dense.kernel':
        'albert.albert.albert_encoder.group.0.inner_group.0.attention.output.dense.weight',
    'bert.encoder.transformer.group_0.inner_group_0.attention_1.self.key.bias':
        'albert.albert.albert_encoder.group.0.inner_group.0.attention.attention.key_layer.bias',
    'bert.encoder.transformer.group_0.inner_group_0.attention_1.self.key.kernel':
        'albert.albert.albert_encoder.group.0.inner_group.0.attention.attention.key_layer.weight',
    'bert.encoder.transformer.group_0.inner_group_0.attention_1.self.query.bias':
        'albert.albert.albert_encoder.group.0.inner_group.0.attention.attention.query_layer.bias',
    'bert.encoder.transformer.group_0.inner_group_0.attention_1.self.query.kernel':
        'albert.albert.albert_encoder.group.0.inner_group.0.attention.attention.query_layer.weight',
    'bert.encoder.transformer.group_0.inner_group_0.attention_1.self.value.bias':
        'albert.albert.albert_encoder.group.0.inner_group.0.attention.attention.value_layer.bias',
    'bert.encoder.transformer.group_0.inner_group_0.attention_1.self.value.kernel':
        'albert.albert.albert_encoder.group.0.inner_group.0.attention.attention.value_layer.weight',
    'bert.encoder.transformer.group_0.inner_group_0.ffn_1.intermediate.dense.bias':
        'albert.albert.albert_encoder.group.0.inner_group.0.intermediate.bias',
    'bert.encoder.transformer.group_0.inner_group_0.ffn_1.intermediate.dense.kernel':
        'albert.albert.albert_encoder.group.0.inner_group.0.intermediate.weight',
    'bert.encoder.transformer.group_0.inner_group_0.ffn_1.intermediate.output.dense.bias':
        'albert.albert.albert_encoder.group.0.inner_group.0.output.dense.bias',
    'bert.encoder.transformer.group_0.inner_group_0.ffn_1.intermediate.output.dense.kernel':
        'albert.albert.albert_encoder.group.0.inner_group.0.output.dense.weight',
    'bert.pooler.dense.bias': 'albert.albert.dense.bias',
    'bert.pooler.dense.kernel': 'albert.albert.dense.weight',
    'cls.predictions.transform.dense.bias': 'albert.cls1.dense.bias',
    'cls.predictions.transform.dense.kernel': 'albert.cls1.dense.weight',
    'cls.predictions.transform.LayerNorm.beta': 'albert.cls1.layernorm.beta',
    'cls.predictions.transform.LayerNorm.gamma': 'albert.cls1.layernorm.gamma',
    'cls.predictions.output_bias': 'albert.cls1.output_bias',
    'cls.seq_relationship.output_bias': 'albert.cls2.dense.bias',
    'cls.seq_relationship.output_weights': 'albert.cls2.dense.weight',
    'global_step': 'global_step'
}
