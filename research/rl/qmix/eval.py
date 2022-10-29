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
"""evaluate QMIX model"""

import argparse
from configs.default import ParamConfig
from src.utils import EvalCb
from src.qmix_trainer import QMIXTrainer
from mindspore_rl.core import Session
from mindspore import context


def alg_config(env):
    config = ParamConfig(env)
    return config


def parse_args():
    parser = argparse.ArgumentParser(description='QMIX evaluation')
    parser.add_argument('--device',
                        type=str,
                        default='CPU',
                        choices=['Ascend', 'CPU', 'GPU', 'Auto'],
                        help='target device to train QMIX(default=GPU).')
    parser.add_argument('--ckpt_path',
                        type=str,
                        default=None,
                        help='The ckpt file to eval.')
    parser.add_argument(
        '--env_name',
        type=str,
        default='2s3z',
        choices=['1c3s5z', '2s3z', '3m', '3s5z', '5m_vs_6m', '8m'],
        help='SMAC environment to eval QMIX(default=2s3z)')
    args, _ = parser.parse_known_args()
    return args


def eval_qmix(args):
    if args.device != 'Auto':
        context.set_context(device_target=args.device)
    context.set_context(mode=context.GRAPH_MODE)
    config = alg_config(args.env_name)
    config.trainer_params.update({'ckpt_path': args.ckpt_path})
    qmix_session = Session(config.algorithm_config)
    eval_cb = EvalCb(times=50)
    qmix_session.run(class_type=QMIXTrainer,
                     episode=1,
                     params=config.trainer_params,
                     callbacks=[eval_cb])


if __name__ == "__main__":
    arglst = parse_args()
    eval_qmix(arglst)
