# Copyright 2023 Huawei Technologies Co., Ltd
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
config parameters
"""

CONFIG = {
    # selection [CPU, GPU]
    'device_platform': 'GPU',
    'seed': 666,
    'batch_size': 128,
    'max_episode_steps': 12,
    'hidden_dim': 256,
    'uniform_action_size': 1000,
    'replay_buffer_size': 8000,
    'max_frames': 8334,
    'start_timesteps': 130,
    'test_epoch': 0,
    'frame_idx': 0,
    'sample_flow_num': 99,
    'repeat_episode_num': 5,
    'sample_episode_num': 1000,
    'momentum': 0.9,
    'critic_lr': 3e-4,
    'transaction_lr': 3e-5,
}
