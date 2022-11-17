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
"""
    # tf and ms bert large checkpoint param transfer table
    # key:   tf
    # value: ms
"""
param_name_dict = {
    'bert/embeddings/word_embeddings': 'bert.bert.bert_embedding_lookup.embedding_table',
    'bert/embeddings/token_type_embeddings': 'bert.bert.bert_embedding_postprocessor.'
                                             'token_type_embedding.embedding_table',
    'bert/embeddings/position_embeddings': 'bert.bert.bert_embedding_postprocessor.'
                                           'full_position_embedding.embedding_table',
    'bert/embeddings/LayerNorm/gamma': 'bert.bert.bert_embedding_postprocessor.layernorm.gamma',
    'bert/embeddings/LayerNorm/beta': 'bert.bert.bert_embedding_postprocessor.layernorm.beta',
    'bert/encoder/layer_0/attention/self/query/kernel': 'bert.bert.bert_encoder.layers.0.attention.attention'
                                                        '.query_layer.weight',
    'bert/encoder/layer_0/attention/self/query/bias': 'bert.bert.bert_encoder.layers.0.attention.attention'
                                                      '.query_layer.bias',
    'bert/encoder/layer_0/attention/self/key/kernel': 'bert.bert.bert_encoder.layers.0.attention.attention.key_layer'
                                                      '.weight',
    'bert/encoder/layer_0/attention/self/key/bias': 'bert.bert.bert_encoder.layers.0.attention.attention'
                                                    '.key_layer.bias',
    'bert/encoder/layer_0/attention/self/value/kernel': 'bert.bert.bert_encoder.layers.0.attention.attention'
                                                        '.value_layer.weight',
    'bert/encoder/layer_0/attention/self/value/bias': 'bert.bert.bert_encoder.layers.0.attention.attention'
                                                      '.value_layer.bias',
    'bert/encoder/layer_0/attention/output/dense/kernel': 'bert.bert.bert_encoder.layers.0.attention.output.dense'
                                                          '.weight',
    'bert/encoder/layer_0/attention/output/dense/bias': 'bert.bert.bert_encoder.layers.0.attention.output.dense.bias',
    'bert/encoder/layer_0/attention/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.0.attention.output'
                                                             '.layernorm.gamma',
    'bert/encoder/layer_0/attention/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.0.attention.output'
                                                            '.layernorm.beta',
    'bert/encoder/layer_0/intermediate/dense/kernel': 'bert.bert.bert_encoder.layers.0.intermediate.weight',
    'bert/encoder/layer_0/intermediate/dense/bias': 'bert.bert.bert_encoder.layers.0.intermediate.bias',
    'bert/encoder/layer_0/output/dense/kernel': 'bert.bert.bert_encoder.layers.0.output.dense.weight',
    'bert/encoder/layer_0/output/dense/bias': 'bert.bert.bert_encoder.layers.0.output.dense.bias',
    'bert/encoder/layer_0/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.0.output.layernorm.gamma',
    'bert/encoder/layer_0/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.0.output.layernorm.beta',
    'bert/encoder/layer_1/attention/self/query/kernel': 'bert.bert.bert_encoder.layers.1.attention.attention'
                                                        '.query_layer.weight',
    'bert/encoder/layer_1/attention/self/query/bias': 'bert.bert.bert_encoder.layers.1.attention.attention'
                                                      '.query_layer.bias',
    'bert/encoder/layer_1/attention/self/key/kernel': 'bert.bert.bert_encoder.layers.1.attention.attention.key_layer'
                                                      '.weight',
    'bert/encoder/layer_1/attention/self/key/bias': 'bert.bert.bert_encoder.layers.1.attention'
                                                    '.attention.key_layer.bias',
    'bert/encoder/layer_1/attention/self/value/kernel': 'bert.bert.bert_encoder.layers.1.attention.attention'
                                                        '.value_layer.weight',
    'bert/encoder/layer_1/attention/self/value/bias': 'bert.bert.bert_encoder.layers.1.attention.attention'
                                                      '.value_layer.bias',
    'bert/encoder/layer_1/attention/output/dense/kernel': 'bert.bert.bert_encoder.layers.1.attention.output.dense'
                                                          '.weight',
    'bert/encoder/layer_1/attention/output/dense/bias': 'bert.bert.bert_encoder.layers.1.attention.output.dense.bias',
    'bert/encoder/layer_1/attention/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.1.attention.output'
                                                             '.layernorm.gamma',
    'bert/encoder/layer_1/attention/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.1.attention.output'
                                                            '.layernorm.beta',
    'bert/encoder/layer_1/intermediate/dense/kernel': 'bert.bert.bert_encoder.layers.1.intermediate.weight',
    'bert/encoder/layer_1/intermediate/dense/bias': 'bert.bert.bert_encoder.layers.1.intermediate.bias',
    'bert/encoder/layer_1/output/dense/kernel': 'bert.bert.bert_encoder.layers.1.output.dense.weight',
    'bert/encoder/layer_1/output/dense/bias': 'bert.bert.bert_encoder.layers.1.output.dense.bias',
    'bert/encoder/layer_1/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.1.output.layernorm.gamma',
    'bert/encoder/layer_1/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.1.output.layernorm.beta',
    'bert/encoder/layer_2/attention/self/query/kernel': 'bert.bert.bert_encoder.layers.2.attention.attention'
                                                        '.query_layer.weight',
    'bert/encoder/layer_2/attention/self/query/bias': 'bert.bert.bert_encoder.layers.2.attention.attention'
                                                      '.query_layer.bias',
    'bert/encoder/layer_2/attention/self/key/kernel': 'bert.bert.bert_encoder.layers.2.attention.attention.key_layer'
                                                      '.weight',
    'bert/encoder/layer_2/attention/self/key/bias': 'bert.bert.bert_encoder.layers.2.attention'
                                                    '.attention.key_layer.bias',
    'bert/encoder/layer_2/attention/self/value/kernel': 'bert.bert.bert_encoder.layers.2.attention.attention'
                                                        '.value_layer.weight',
    'bert/encoder/layer_2/attention/self/value/bias': 'bert.bert.bert_encoder.layers.2.attention.attention'
                                                      '.value_layer.bias',
    'bert/encoder/layer_2/attention/output/dense/kernel': 'bert.bert.bert_encoder.layers.2.attention.output.dense'
                                                          '.weight',
    'bert/encoder/layer_2/attention/output/dense/bias': 'bert.bert.bert_encoder.layers.2.attention.output.dense.bias',
    'bert/encoder/layer_2/attention/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.2.attention.output'
                                                             '.layernorm.gamma',
    'bert/encoder/layer_2/attention/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.2.attention.output'
                                                            '.layernorm.beta',
    'bert/encoder/layer_2/intermediate/dense/kernel': 'bert.bert.bert_encoder.layers.2.intermediate.weight',
    'bert/encoder/layer_2/intermediate/dense/bias': 'bert.bert.bert_encoder.layers.2.intermediate.bias',
    'bert/encoder/layer_2/output/dense/kernel': 'bert.bert.bert_encoder.layers.2.output.dense.weight',
    'bert/encoder/layer_2/output/dense/bias': 'bert.bert.bert_encoder.layers.2.output.dense.bias',
    'bert/encoder/layer_2/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.2.output.layernorm.gamma',
    'bert/encoder/layer_2/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.2.output.layernorm.beta',
    'bert/encoder/layer_3/attention/self/query/kernel': 'bert.bert.bert_encoder.layers.3.attention.attention'
                                                        '.query_layer.weight',
    'bert/encoder/layer_3/attention/self/query/bias': 'bert.bert.bert_encoder.layers.3.attention.attention'
                                                      '.query_layer.bias',
    'bert/encoder/layer_3/attention/self/key/kernel': 'bert.bert.bert_encoder.layers.3.attention.attention.key_layer'
                                                      '.weight',
    'bert/encoder/layer_3/attention/self/key/bias': 'bert.bert.bert_encoder.layers.3.attention'
                                                    '.attention.key_layer.bias',
    'bert/encoder/layer_3/attention/self/value/kernel': 'bert.bert.bert_encoder.layers.3.attention.attention'
                                                        '.value_layer.weight',
    'bert/encoder/layer_3/attention/self/value/bias': 'bert.bert.bert_encoder.layers.3.attention.attention'
                                                      '.value_layer.bias',
    'bert/encoder/layer_3/attention/output/dense/kernel': 'bert.bert.bert_encoder.layers.3.attention.output.dense'
                                                          '.weight',
    'bert/encoder/layer_3/attention/output/dense/bias': 'bert.bert.bert_encoder.layers.3.attention.output.dense.bias',
    'bert/encoder/layer_3/attention/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.3.attention.output'
                                                             '.layernorm.gamma',
    'bert/encoder/layer_3/attention/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.3.attention.output'
                                                            '.layernorm.beta',
    'bert/encoder/layer_3/intermediate/dense/kernel': 'bert.bert.bert_encoder.layers.3.intermediate.weight',
    'bert/encoder/layer_3/intermediate/dense/bias': 'bert.bert.bert_encoder.layers.3.intermediate.bias',
    'bert/encoder/layer_3/output/dense/kernel': 'bert.bert.bert_encoder.layers.3.output.dense.weight',
    'bert/encoder/layer_3/output/dense/bias': 'bert.bert.bert_encoder.layers.3.output.dense.bias',
    'bert/encoder/layer_3/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.3.output.layernorm.gamma',
    'bert/encoder/layer_3/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.3.output.layernorm.beta',
    'bert/encoder/layer_4/attention/self/query/kernel': 'bert.bert.bert_encoder.layers.4.attention.attention'
                                                        '.query_layer.weight',
    'bert/encoder/layer_4/attention/self/query/bias': 'bert.bert.bert_encoder.layers.4.attention.attention'
                                                      '.query_layer.bias',
    'bert/encoder/layer_4/attention/self/key/kernel': 'bert.bert.bert_encoder.layers.4.attention.attention.key_layer'
                                                      '.weight',
    'bert/encoder/layer_4/attention/self/key/bias': 'bert.bert.bert_encoder.layers.4'
                                                    '.attention.attention.key_layer.bias',
    'bert/encoder/layer_4/attention/self/value/kernel': 'bert.bert.bert_encoder.layers.4.attention.attention'
                                                        '.value_layer.weight',
    'bert/encoder/layer_4/attention/self/value/bias': 'bert.bert.bert_encoder.layers.4.attention.attention'
                                                      '.value_layer.bias',
    'bert/encoder/layer_4/attention/output/dense/kernel': 'bert.bert.bert_encoder.layers.4.attention.output.dense'
                                                          '.weight',
    'bert/encoder/layer_4/attention/output/dense/bias': 'bert.bert.bert_encoder.layers.4.attention.output.dense.bias',
    'bert/encoder/layer_4/attention/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.4.attention.output'
                                                             '.layernorm.gamma',
    'bert/encoder/layer_4/attention/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.4.attention.output'
                                                            '.layernorm.beta',
    'bert/encoder/layer_4/intermediate/dense/kernel': 'bert.bert.bert_encoder.layers.4.intermediate.weight',
    'bert/encoder/layer_4/intermediate/dense/bias': 'bert.bert.bert_encoder.layers.4.intermediate.bias',
    'bert/encoder/layer_4/output/dense/kernel': 'bert.bert.bert_encoder.layers.4.output.dense.weight',
    'bert/encoder/layer_4/output/dense/bias': 'bert.bert.bert_encoder.layers.4.output.dense.bias',
    'bert/encoder/layer_4/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.4.output.layernorm.gamma',
    'bert/encoder/layer_4/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.4.output.layernorm.beta',
    'bert/encoder/layer_5/attention/self/query/kernel': 'bert.bert.bert_encoder.layers.5.attention.attention'
                                                        '.query_layer.weight',
    'bert/encoder/layer_5/attention/self/query/bias': 'bert.bert.bert_encoder.layers.5.attention.attention'
                                                      '.query_layer.bias',
    'bert/encoder/layer_5/attention/self/key/kernel': 'bert.bert.bert_encoder.layers.5.attention.attention.key_layer'
                                                      '.weight',
    'bert/encoder/layer_5/attention/self/key/bias': 'bert.bert.bert_encoder.layers.5.attention'
                                                    '.attention.key_layer.bias',
    'bert/encoder/layer_5/attention/self/value/kernel': 'bert.bert.bert_encoder.layers.5.attention.attention'
                                                        '.value_layer.weight',
    'bert/encoder/layer_5/attention/self/value/bias': 'bert.bert.bert_encoder.layers.5.attention.attention'
                                                      '.value_layer.bias',
    'bert/encoder/layer_5/attention/output/dense/kernel': 'bert.bert.bert_encoder.layers.5.attention.output.dense'
                                                          '.weight',
    'bert/encoder/layer_5/attention/output/dense/bias': 'bert.bert.bert_encoder.layers.5.attention.output.dense.bias',
    'bert/encoder/layer_5/attention/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.5.attention.output'
                                                             '.layernorm.gamma',
    'bert/encoder/layer_5/attention/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.5.attention.output'
                                                            '.layernorm.beta',
    'bert/encoder/layer_5/intermediate/dense/kernel': 'bert.bert.bert_encoder.layers.5.intermediate.weight',
    'bert/encoder/layer_5/intermediate/dense/bias': 'bert.bert.bert_encoder.layers.5.intermediate.bias',
    'bert/encoder/layer_5/output/dense/kernel': 'bert.bert.bert_encoder.layers.5.output.dense.weight',
    'bert/encoder/layer_5/output/dense/bias': 'bert.bert.bert_encoder.layers.5.output.dense.bias',
    'bert/encoder/layer_5/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.5.output.layernorm.gamma',
    'bert/encoder/layer_5/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.5.output.layernorm.beta',
    'bert/encoder/layer_6/attention/self/query/kernel': 'bert.bert.bert_encoder.layers.6.attention.attention'
                                                        '.query_layer.weight',
    'bert/encoder/layer_6/attention/self/query/bias': 'bert.bert.bert_encoder.layers.6.attention.attention'
                                                      '.query_layer.bias',
    'bert/encoder/layer_6/attention/self/key/kernel': 'bert.bert.bert_encoder.layers.6.attention.attention.key_layer'
                                                      '.weight',
    'bert/encoder/layer_6/attention/self/key/bias': 'bert.bert.bert_encoder.layers.6.attention'
                                                    '.attention.key_layer.bias',
    'bert/encoder/layer_6/attention/self/value/kernel': 'bert.bert.bert_encoder.layers.6.attention.attention'
                                                        '.value_layer.weight',
    'bert/encoder/layer_6/attention/self/value/bias': 'bert.bert.bert_encoder.layers.6.attention.attention'
                                                      '.value_layer.bias',
    'bert/encoder/layer_6/attention/output/dense/kernel': 'bert.bert.bert_encoder.layers.6.attention.output.dense'
                                                          '.weight',
    'bert/encoder/layer_6/attention/output/dense/bias': 'bert.bert.bert_encoder.layers.6.attention.output.dense.bias',
    'bert/encoder/layer_6/attention/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.6.attention.output'
                                                             '.layernorm.gamma',
    'bert/encoder/layer_6/attention/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.6.attention.output'
                                                            '.layernorm.beta',
    'bert/encoder/layer_6/intermediate/dense/kernel': 'bert.bert.bert_encoder.layers.6.intermediate.weight',
    'bert/encoder/layer_6/intermediate/dense/bias': 'bert.bert.bert_encoder.layers.6.intermediate.bias',
    'bert/encoder/layer_6/output/dense/kernel': 'bert.bert.bert_encoder.layers.6.output.dense.weight',
    'bert/encoder/layer_6/output/dense/bias': 'bert.bert.bert_encoder.layers.6.output.dense.bias',
    'bert/encoder/layer_6/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.6.output.layernorm.gamma',
    'bert/encoder/layer_6/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.6.output.layernorm.beta',
    'bert/encoder/layer_7/attention/self/query/kernel': 'bert.bert.bert_encoder.layers.7.attention.attention'
                                                        '.query_layer.weight',
    'bert/encoder/layer_7/attention/self/query/bias': 'bert.bert.bert_encoder.layers.7.attention.attention'
                                                      '.query_layer.bias',
    'bert/encoder/layer_7/attention/self/key/kernel': 'bert.bert.bert_encoder.layers.7.attention.attention.key_layer'
                                                      '.weight',
    'bert/encoder/layer_7/attention/self/key/bias': 'bert.bert.bert_encoder.layers.7.attention'
                                                    '.attention.key_layer.bias',
    'bert/encoder/layer_7/attention/self/value/kernel': 'bert.bert.bert_encoder.layers.7.attention.attention'
                                                        '.value_layer.weight',
    'bert/encoder/layer_7/attention/self/value/bias': 'bert.bert.bert_encoder.layers.7.attention.attention'
                                                      '.value_layer.bias',
    'bert/encoder/layer_7/attention/output/dense/kernel': 'bert.bert.bert_encoder.layers.7.attention.output.dense'
                                                          '.weight',
    'bert/encoder/layer_7/attention/output/dense/bias': 'bert.bert.bert_encoder.layers.7.attention.output.dense.bias',
    'bert/encoder/layer_7/attention/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.7.attention.output'
                                                             '.layernorm.gamma',
    'bert/encoder/layer_7/attention/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.7.attention.output'
                                                            '.layernorm.beta',
    'bert/encoder/layer_7/intermediate/dense/kernel': 'bert.bert.bert_encoder.layers.7.intermediate.weight',
    'bert/encoder/layer_7/intermediate/dense/bias': 'bert.bert.bert_encoder.layers.7.intermediate.bias',
    'bert/encoder/layer_7/output/dense/kernel': 'bert.bert.bert_encoder.layers.7.output.dense.weight',
    'bert/encoder/layer_7/output/dense/bias': 'bert.bert.bert_encoder.layers.7.output.dense.bias',
    'bert/encoder/layer_7/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.7.output.layernorm.gamma',
    'bert/encoder/layer_7/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.7.output.layernorm.beta',
    'bert/encoder/layer_8/attention/self/query/kernel': 'bert.bert.bert_encoder.layers.8.attention.attention'
                                                        '.query_layer.weight',
    'bert/encoder/layer_8/attention/self/query/bias': 'bert.bert.bert_encoder.layers.8.attention.attention'
                                                      '.query_layer.bias',
    'bert/encoder/layer_8/attention/self/key/kernel': 'bert.bert.bert_encoder.layers.8.attention.attention.key_layer'
                                                      '.weight',
    'bert/encoder/layer_8/attention/self/key/bias': 'bert.bert.bert_encoder.layers.8.attention'
                                                    '.attention.key_layer.bias',
    'bert/encoder/layer_8/attention/self/value/kernel': 'bert.bert.bert_encoder.layers.8.attention.attention'
                                                        '.value_layer.weight',
    'bert/encoder/layer_8/attention/self/value/bias': 'bert.bert.bert_encoder.layers.8.attention.attention'
                                                      '.value_layer.bias',
    'bert/encoder/layer_8/attention/output/dense/kernel': 'bert.bert.bert_encoder.layers.8.attention.output.dense'
                                                          '.weight',
    'bert/encoder/layer_8/attention/output/dense/bias': 'bert.bert.bert_encoder.layers.8.attention.output.dense.bias',
    'bert/encoder/layer_8/attention/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.8.attention.output'
                                                             '.layernorm.gamma',
    'bert/encoder/layer_8/attention/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.8.attention.output'
                                                            '.layernorm.beta',
    'bert/encoder/layer_8/intermediate/dense/kernel': 'bert.bert.bert_encoder.layers.8.intermediate.weight',
    'bert/encoder/layer_8/intermediate/dense/bias': 'bert.bert.bert_encoder.layers.8.intermediate.bias',
    'bert/encoder/layer_8/output/dense/kernel': 'bert.bert.bert_encoder.layers.8.output.dense.weight',
    'bert/encoder/layer_8/output/dense/bias': 'bert.bert.bert_encoder.layers.8.output.dense.bias',
    'bert/encoder/layer_8/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.8.output.layernorm.gamma',
    'bert/encoder/layer_8/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.8.output.layernorm.beta',
    'bert/encoder/layer_9/attention/self/query/kernel': 'bert.bert.bert_encoder.layers.9.attention.attention'
                                                        '.query_layer.weight',
    'bert/encoder/layer_9/attention/self/query/bias': 'bert.bert.bert_encoder.layers.9.attention.attention'
                                                      '.query_layer.bias',
    'bert/encoder/layer_9/attention/self/key/kernel': 'bert.bert.bert_encoder.layers.9.attention.attention.key_layer'
                                                      '.weight',
    'bert/encoder/layer_9/attention/self/key/bias': 'bert.bert.bert_encoder.layers.9.attention'
                                                    '.attention.key_layer.bias',
    'bert/encoder/layer_9/attention/self/value/kernel': 'bert.bert.bert_encoder.layers.9.attention.attention'
                                                        '.value_layer.weight',
    'bert/encoder/layer_9/attention/self/value/bias': 'bert.bert.bert_encoder.layers.9.attention.attention'
                                                      '.value_layer.bias',
    'bert/encoder/layer_9/attention/output/dense/kernel': 'bert.bert.bert_encoder.layers.9.attention.output.dense'
                                                          '.weight',
    'bert/encoder/layer_9/attention/output/dense/bias': 'bert.bert.bert_encoder.layers.9.attention.output.dense.bias',
    'bert/encoder/layer_9/attention/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.9.attention.output'
                                                             '.layernorm.gamma',
    'bert/encoder/layer_9/attention/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.9.attention.output'
                                                            '.layernorm.beta',
    'bert/encoder/layer_9/intermediate/dense/kernel': 'bert.bert.bert_encoder.layers.9.intermediate.weight',
    'bert/encoder/layer_9/intermediate/dense/bias': 'bert.bert.bert_encoder.layers.9.intermediate.bias',
    'bert/encoder/layer_9/output/dense/kernel': 'bert.bert.bert_encoder.layers.9.output.dense.weight',
    'bert/encoder/layer_9/output/dense/bias': 'bert.bert.bert_encoder.layers.9.output.dense.bias',
    'bert/encoder/layer_9/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.9.output.layernorm.gamma',
    'bert/encoder/layer_9/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.9.output.layernorm.beta',
    'bert/encoder/layer_10/attention/self/query/kernel': 'bert.bert.bert_encoder.layers.10.attention.attention'
                                                         '.query_layer.weight',
    'bert/encoder/layer_10/attention/self/query/bias': 'bert.bert.bert_encoder.layers.10.attention.attention'
                                                       '.query_layer.bias',
    'bert/encoder/layer_10/attention/self/key/kernel': 'bert.bert.bert_encoder.layers.10.attention.attention'
                                                       '.key_layer.weight',
    'bert/encoder/layer_10/attention/self/key/bias': 'bert.bert.bert_encoder.layers.10.attention.attention.key_layer'
                                                     '.bias',
    'bert/encoder/layer_10/attention/self/value/kernel': 'bert.bert.bert_encoder.layers.10.attention.attention'
                                                         '.value_layer.weight',
    'bert/encoder/layer_10/attention/self/value/bias': 'bert.bert.bert_encoder.layers.10.attention.attention'
                                                       '.value_layer.bias',
    'bert/encoder/layer_10/attention/output/dense/kernel': 'bert.bert.bert_encoder.layers.10.attention.output.dense'
                                                           '.weight',
    'bert/encoder/layer_10/attention/output/dense/bias': 'bert.bert.bert_encoder.layers.10.attention.output.dense.bias',
    'bert/encoder/layer_10/attention/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.10.attention.output'
                                                              '.layernorm.gamma',
    'bert/encoder/layer_10/attention/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.10.attention.output'
                                                             '.layernorm.beta',
    'bert/encoder/layer_10/intermediate/dense/kernel': 'bert.bert.bert_encoder.layers.10.intermediate.weight',
    'bert/encoder/layer_10/intermediate/dense/bias': 'bert.bert.bert_encoder.layers.10.intermediate.bias',
    'bert/encoder/layer_10/output/dense/kernel': 'bert.bert.bert_encoder.layers.10.output.dense.weight',
    'bert/encoder/layer_10/output/dense/bias': 'bert.bert.bert_encoder.layers.10.output.dense.bias',
    'bert/encoder/layer_10/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.10.output.layernorm.gamma',
    'bert/encoder/layer_10/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.10.output.layernorm.beta',
    'bert/encoder/layer_11/attention/self/query/kernel': 'bert.bert.bert_encoder.layers.11.attention.attention'
                                                         '.query_layer.weight',
    'bert/encoder/layer_11/attention/self/query/bias': 'bert.bert.bert_encoder.layers.11.attention.attention'
                                                       '.query_layer.bias',
    'bert/encoder/layer_11/attention/self/key/kernel': 'bert.bert.bert_encoder.layers.11.attention.attention'
                                                       '.key_layer.weight',
    'bert/encoder/layer_11/attention/self/key/bias': 'bert.bert.bert_encoder.layers.11.attention.attention.key_layer'
                                                     '.bias',
    'bert/encoder/layer_11/attention/self/value/kernel': 'bert.bert.bert_encoder.layers.11.attention.attention'
                                                         '.value_layer.weight',
    'bert/encoder/layer_11/attention/self/value/bias': 'bert.bert.bert_encoder.layers.11.attention.attention'
                                                       '.value_layer.bias',
    'bert/encoder/layer_11/attention/output/dense/kernel': 'bert.bert.bert_encoder.layers.11.attention.output.dense'
                                                           '.weight',
    'bert/encoder/layer_11/attention/output/dense/bias': 'bert.bert.bert_encoder.layers.11.attention.output.dense.bias',
    'bert/encoder/layer_11/attention/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.11.attention.output'
                                                              '.layernorm.gamma',
    'bert/encoder/layer_11/attention/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.11.attention.output'
                                                             '.layernorm.beta',
    'bert/encoder/layer_11/intermediate/dense/kernel': 'bert.bert.bert_encoder.layers.11.intermediate.weight',
    'bert/encoder/layer_11/intermediate/dense/bias': 'bert.bert.bert_encoder.layers.11.intermediate.bias',
    'bert/encoder/layer_11/output/dense/kernel': 'bert.bert.bert_encoder.layers.11.output.dense.weight',
    'bert/encoder/layer_11/output/dense/bias': 'bert.bert.bert_encoder.layers.11.output.dense.bias',
    'bert/encoder/layer_11/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.11.output.layernorm.gamma',
    'bert/encoder/layer_11/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.11.output.layernorm.beta',
    'bert/pooler/dense/kernel': 'bert.bert.dense.weight',
    'bert/pooler/dense/bias': 'bert.bert.dense.bias',
    'cls/predictions/output_bias': 'bert.cls1.output_bias',
    'cls/predictions/transform/dense/kernel': 'bert.cls1.dense.weight',
    'cls/predictions/transform/dense/bias': 'bert.cls1.dense.bias',
    'cls/predictions/transform/LayerNorm/gamma': 'bert.cls1.layernorm.gamma',
    'cls/predictions/transform/LayerNorm/beta': 'bert.cls1.layernorm.beta',
    'cls/seq_relationship/output_weights': 'bert.cls2.dense.weight',
    'cls/seq_relationship/output_bias': 'bert.cls2.dense.bias',
}

# Weights need to be transposed while transfer
transpose_list = [
    'bert.bert.bert_encoder.layers.0.attention.attention.query_layer.weight',
    'bert.bert.bert_encoder.layers.0.attention.attention.key_layer.weight',
    'bert.bert.bert_encoder.layers.0.attention.attention.value_layer.weight',
    'bert.bert.bert_encoder.layers.0.attention.output.dense.weight',
    'bert.bert.bert_encoder.layers.0.intermediate.weight',
    'bert.bert.bert_encoder.layers.0.output.dense.weight',
    'bert.bert.bert_encoder.layers.1.attention.attention.query_layer.weight',
    'bert.bert.bert_encoder.layers.1.attention.attention.key_layer.weight',
    'bert.bert.bert_encoder.layers.1.attention.attention.value_layer.weight',
    'bert.bert.bert_encoder.layers.1.attention.output.dense.weight',
    'bert.bert.bert_encoder.layers.1.intermediate.weight',
    'bert.bert.bert_encoder.layers.1.output.dense.weight',
    'bert.bert.bert_encoder.layers.2.attention.attention.query_layer.weight',
    'bert.bert.bert_encoder.layers.2.attention.attention.key_layer.weight',
    'bert.bert.bert_encoder.layers.2.attention.attention.value_layer.weight',
    'bert.bert.bert_encoder.layers.2.attention.output.dense.weight',
    'bert.bert.bert_encoder.layers.2.intermediate.weight',
    'bert.bert.bert_encoder.layers.2.output.dense.weight',
    'bert.bert.bert_encoder.layers.3.attention.attention.query_layer.weight',
    'bert.bert.bert_encoder.layers.3.attention.attention.key_layer.weight',
    'bert.bert.bert_encoder.layers.3.attention.attention.value_layer.weight',
    'bert.bert.bert_encoder.layers.3.attention.output.dense.weight',
    'bert.bert.bert_encoder.layers.3.intermediate.weight',
    'bert.bert.bert_encoder.layers.3.output.dense.weight',
    'bert.bert.bert_encoder.layers.4.attention.attention.query_layer.weight',
    'bert.bert.bert_encoder.layers.4.attention.attention.key_layer.weight',
    'bert.bert.bert_encoder.layers.4.attention.attention.value_layer.weight',
    'bert.bert.bert_encoder.layers.4.attention.output.dense.weight',
    'bert.bert.bert_encoder.layers.4.intermediate.weight',
    'bert.bert.bert_encoder.layers.4.output.dense.weight',
    'bert.bert.bert_encoder.layers.5.attention.attention.query_layer.weight',
    'bert.bert.bert_encoder.layers.5.attention.attention.key_layer.weight',
    'bert.bert.bert_encoder.layers.5.attention.attention.value_layer.weight',
    'bert.bert.bert_encoder.layers.5.attention.output.dense.weight',
    'bert.bert.bert_encoder.layers.5.intermediate.weight',
    'bert.bert.bert_encoder.layers.5.output.dense.weight',
    'bert.bert.bert_encoder.layers.6.attention.attention.query_layer.weight',
    'bert.bert.bert_encoder.layers.6.attention.attention.key_layer.weight',
    'bert.bert.bert_encoder.layers.6.attention.attention.value_layer.weight',
    'bert.bert.bert_encoder.layers.6.attention.output.dense.weight',
    'bert.bert.bert_encoder.layers.6.intermediate.weight',
    'bert.bert.bert_encoder.layers.6.output.dense.weight',
    'bert.bert.bert_encoder.layers.7.attention.attention.query_layer.weight',
    'bert.bert.bert_encoder.layers.7.attention.attention.key_layer.weight',
    'bert.bert.bert_encoder.layers.7.attention.attention.value_layer.weight',
    'bert.bert.bert_encoder.layers.7.attention.output.dense.weight',
    'bert.bert.bert_encoder.layers.7.intermediate.weight',
    'bert.bert.bert_encoder.layers.7.output.dense.weight',
    'bert.bert.bert_encoder.layers.8.attention.attention.query_layer.weight',
    'bert.bert.bert_encoder.layers.8.attention.attention.key_layer.weight',
    'bert.bert.bert_encoder.layers.8.attention.attention.value_layer.weight',
    'bert.bert.bert_encoder.layers.8.attention.output.dense.weight',
    'bert.bert.bert_encoder.layers.8.intermediate.weight',
    'bert.bert.bert_encoder.layers.8.output.dense.weight',
    'bert.bert.bert_encoder.layers.9.attention.attention.query_layer.weight',
    'bert.bert.bert_encoder.layers.9.attention.attention.key_layer.weight',
    'bert.bert.bert_encoder.layers.9.attention.attention.value_layer.weight',
    'bert.bert.bert_encoder.layers.9.attention.output.dense.weight',
    'bert.bert.bert_encoder.layers.9.intermediate.weight',
    'bert.bert.bert_encoder.layers.9.output.dense.weight',
    'bert.bert.bert_encoder.layers.10.attention.attention.query_layer.weight',
    'bert.bert.bert_encoder.layers.10.attention.attention.key_layer.weight',
    'bert.bert.bert_encoder.layers.10.attention.attention.value_layer.weight',
    'bert.bert.bert_encoder.layers.10.attention.output.dense.weight',
    'bert.bert.bert_encoder.layers.10.intermediate.weight',
    'bert.bert.bert_encoder.layers.10.output.dense.weight',
    'bert.bert.bert_encoder.layers.11.attention.attention.query_layer.weight',
    'bert.bert.bert_encoder.layers.11.attention.attention.key_layer.weight',
    'bert.bert.bert_encoder.layers.11.attention.attention.value_layer.weight',
    'bert.bert.bert_encoder.layers.11.attention.output.dense.weight',
    'bert.bert.bert_encoder.layers.11.intermediate.weight',
    'bert.bert.bert_encoder.layers.11.output.dense.weight',
    'bert.bert.dense.weight',
    'bert.cls1.dense.weight',
]
