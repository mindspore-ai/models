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
"""train QMIX model"""

import argparse
from configs.default import ParamConfig
from src.qmix_trainer import QMIXTrainer
from src.utils import RecordCb
from mindspore import context
from mindspore_rl.core import Session
from mindspore_rl.utils.callback import CheckpointCallback, LossCallback


def alg_config(env):
    config = ParamConfig(env)
    return config


def parse_args():
    parser = argparse.ArgumentParser(description='QMIX training')
    parser.add_argument('--episode',
                        type=int,
                        default=1000000,
                        help='total episode numbers.')
    parser.add_argument('--device',
                        type=str,
                        default='GPU',
                        choices=['CPU', 'GPU', 'Auto'],
                        help='target device to train QMIX(default=GPU).')
    parser.add_argument(
        '--env_name',
        type=str,
        default='2s3z',
        choices=['1c3s5z', '2s3z', '3m', '3s5z', '5m_vs_6m', '8m'],
        help='SMAC environment to train QMIX(default=2s3z)')
    args, _ = parser.parse_known_args()
    return args


def initCB(config):
    record_cb = RecordCb(config.trainer_params['summary_path'], 100, 200, 20)
    loss_cb = LossCallback()
    ckpt_cb = CheckpointCallback(1, config.trainer_params['ckpt_path'])
    cblst = [loss_cb, record_cb, ckpt_cb]
    return cblst


def train(args):
    """start to train qmix algorithm"""
    if args.device != 'Auto':
        context.set_context(device_target=args.device)
    context.set_context(mode=context.GRAPH_MODE)
    config = alg_config(args.env_name)
    qmix_session = Session(config.algorithm_config)
    cblst = initCB(config)
    qmix_session.run(class_type=QMIXTrainer,
                     episode=args.episode,
                     params=config.trainer_params,
                     callbacks=cblst)


if __name__ == "__main__":
    arglst = parse_args()
    train(arglst)
