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
"""QMIX default parameter config"""

import mindspore as ms
from mindspore_rl.environment import StarCraft2Environment
from mindspore_rl.core.replay_buffer import ReplayBuffer
from src.qmix_actor import QMIXActor, QMIXPolicy
from src.qmix_learner import QMIXLearner


def env_info(env_name):
    env_params = {'sc2_args': {'map_name': env_name}}
    temp_env = StarCraft2Environment(env_params, 0)
    info = temp_env.env_info
    return info


class ParamConfig():

    def __init__(self, env_name):
        ENV_NAME = env_name
        SEED = 42
        BATCH_SIZE = 64
        env_param = env_info(ENV_NAME)
        state_dim = env_param["state_shape"]
        obs_dim = env_param["obs_shape"]
        action_dim = env_param["n_actions"]
        agent_num = env_param["n_agents"]
        episode_limit = env_param["episode_limit"]
        self.collect_env_params = {
            'sc2_args': {
                'map_name': ENV_NAME,
                'seed': SEED
            }
        }
        self.eval_env_params = {'sc2_args': {'map_name': ENV_NAME}}
        # parameter config for network
        self.policy_params = {
            # parameter for agent network
            'rnn_type': 'GRU',
            'rnn_layer_num': 1,
            'hidden_dim': 64,
            # params for mixer network
            'embed_dim': 32,
            'hypernet_embed': 64,
            # params for epsilon greedy
            'epsi_start': 1.0,
            'epsi_end': 0.05,
            'all_steps': 50000
        }

        self.learner_params = {
            'lr': 0.0005,
            'gamma': 0.99,
            'batch_size': BATCH_SIZE,
        }

        self.trainer_params = {
            'hidden_dim': 64,
            'batch_size': BATCH_SIZE,
            'summary_path': './log/{}'.format(ENV_NAME),
            'ckpt_path': './out/{}'.format(ENV_NAME)
        }

        self.algorithm_config = {
            'actor': {
                'number': 1,
                'type': QMIXActor,
                'policies': ['collect_policy', 'eval_policy'],
            },
            'learner': {
                'number': 1,
                'type': QMIXLearner,
                'params': self.learner_params,
                'networks': ['agents', 'mixer']
            },
            'policy_and_network': {
                'type': QMIXPolicy,
                'params': self.policy_params
            },
            'collect_environment': {
                'number': 1,
                'type': StarCraft2Environment,
                'params': self.collect_env_params
            },
            'eval_environment': {
                'number': 1,
                'type': StarCraft2Environment,
                'params': self.eval_env_params
            },
            'replay_buffer': {
                'number':
                1,
                'type':
                ReplayBuffer,
                'capacity':
                5000,
                # obs, state, action, reward, done, hidden_state, avaial_action, filled
                'data_shape': [(episode_limit + 1, agent_num,
                                obs_dim + agent_num + action_dim),
                               (episode_limit + 1, state_dim),
                               (episode_limit + 1, agent_num, 1),
                               (episode_limit + 1, 1), (episode_limit + 1, 1),
                               (episode_limit + 1, agent_num, 64),
                               (episode_limit + 1, agent_num, action_dim),
                               (episode_limit + 1, 1)],
                'data_type': [
                    ms.float32, ms.float32, ms.int32, ms.float32, ms.bool_,
                    ms.float32, ms.int32, ms.int32
                ],
                'sample_size':
                BATCH_SIZE,
            }
        }
