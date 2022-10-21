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
"""ONNX verify"""
import argparse
import gym
from src.config import config
from mindspore import context
import onnxruntime as ort
import numpy as np

parser = argparse.ArgumentParser(description='ONNX ddpg Example')
parser.add_argument('--device_target', type=str, default="GPU", choices=['Ascend', 'GPU'],
                    help='device where the code will be implemented (default: Ascend)')
parser.add_argument('--device_id', type=int, default=0, help='if is test, must provide\
                    path where the trained ckpt file')
args = parser.parse_args()

context.set_context(device_id=args.device_id)
EP_TEST = config.EP_TEST
STEP_TEST = config.STEP_TEST
REWORD_SCOPE = 16.2736044

def create_session(checkpoint_path, target_device):
    '''
    create onnx session
    '''
    if target_device == 'GPU':
        providers = ['CUDAExecutionProvider']
    elif target_device == 'CPU':
        providers = ['CPUExecutionProvider']
    else:
        raise ValueError(f'Unsupported target device {target_device!r}. Expected one of: "CPU", "GPU"')
    session = ort.InferenceSession(checkpoint_path, providers=providers)
    input_name = session.get_inputs()[0].name

    return session, input_name

def verify():
    """ verify"""
    env = gym.make('Pendulum-v1')
    env = env.unwrapped
    env.seed(1)

    session, input_name = create_session('./actornet.onnx', 'GPU')
    rewards = []
    for i in range(EP_TEST):
        reward_sum = 0
        state = env.reset()
        for j in range(STEP_TEST):
            state = np.expand_dims(state, axis=0)
            state = state.astype(np.float32)

            action = session.run(None, {input_name: state})[0]
            action = action[np.argmax(action)]

            next_state, reward, _, _, _ = env.step(action)
            reward_sum += reward
            state = next_state
            if j == STEP_TEST - 1:
                print('Episode: ', i, ' Reward:', reward_sum / REWORD_SCOPE)
                rewards.append(reward_sum)
                break
    print('Final Average Reward: ', sum(rewards) / (len(rewards) * REWORD_SCOPE))


if __name__ == '__main__':
    verify()
