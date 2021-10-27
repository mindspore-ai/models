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
Configurations.
"""

from easydict import EasyDict as edict


# Returns a minimal configuration for testing.
get_testing = edict({
    'patches_grid': None,
    'patches_size': 16,
    'hidden_size': 1,
    'transformer_mlp_dim': 1,
    'transformer_num_heads': 1,
    'transformer_num_layers': 1,
    'transformer_attention_dropout_rate': 1.0,
    'transformer_dropout_rate': 0.9,
    'classifier': 'token',
    'representation_size': None,
})


# Returns the ViT-B/16 configuration.
get_b16_config = edict({
    'patches_grid': None,
    'patches_size': 16,
    'hidden_size': 768,
    'transformer_mlp_dim': 3072,
    'transformer_num_heads': 12,
    'transformer_num_layers': 12,
    'transformer_attention_dropout_rate': 1.0,
    'transformer_dropout_rate': 1.0,  # 0.9
    'classifier': 'token',
    'representation_size': None,
})


# Returns the Resnet50 + ViT-B/16 configuration.
get_r50_b16_config = edict({
    'patches_grid': 14,
    'resnet_num_layers': (3, 4, 9),
    'resnet_width_factor': 1,
})


# Returns the ViT-B/32 configuration.
get_b32_config = edict({
    'patches_grid': None,
    'patches_size': 32,
    'hidden_size': 768,
    'transformer_mlp_dim': 3072,
    'transformer_num_heads': 12,
    'transformer_num_layers': 12,
    'transformer_attention_dropout_rate': 1.0,
    'transformer_dropout_rate': 0.9,
    'classifier': 'token',
    'representation_size': None,
})


# Returns the ViT-L/16 configuration.
get_l16_config = edict({
    'patches_grid': None,
    'patches_size': 16,
    'hidden_size': 1024,
    'transformer_mlp_dim': 4096,
    'transformer_num_heads': 16,
    'transformer_num_layers': 24,
    'transformer_attention_dropout_rate': 1.0,
    'transformer_dropout_rate': 0.9,
    'classifier': 'token',
    'representation_size': None,
})


# Returns the ViT-L/32 configuration.
get_l32_config = edict({
    'patches_grid': None,
    'patches_size': 32,
    'hidden_size': 1024,
    'transformer_mlp_dim': 4096,
    'transformer_num_heads': 16,
    'transformer_num_layers': 24,
    'transformer_attention_dropout_rate': 1.0,
    'transformer_dropout_rate': 0.9,
    'classifier': 'token',
    'representation_size': None,
})


# Returns the ViT-L/16 configuration.
get_h14_config = edict({
    'patches_grid': None,
    'patches_size': 14,
    'hidden_size': 1280,
    'transformer_mlp_dim': 5120,
    'transformer_num_heads': 16,
    'transformer_num_layers': 32,
    'transformer_attention_dropout_rate': 1.0,
    'transformer_dropout_rate': 0.9,
    'classifier': 'token',
    'representation_size': None,
})
