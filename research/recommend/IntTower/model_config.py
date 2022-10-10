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

mlp_layers = [300, 300, 128]
feblock_size = 256
head_num = 4
user_embedding_dim = 129
item_embedding_dim = 33
sparse_embedding_dim = 32
use_multi_layer = True
user_sparse_field = 4
keep_rate = 0.9
epoch = 10
batch_size = 2048
seed = 3047
lr = 0.0005
