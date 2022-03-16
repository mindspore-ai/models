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
"""Ascend310verify"""
import os
import argparse
import numpy as np
import gym


parser = argparse.ArgumentParser(description='test')
parser.add_argument("--run_main", type=str, default=None, help="310 run file")
parser.add_argument("--model_path", type=str, default="test.mindir", help="mindir file path")
parser.add_argument("--output_path", type=str, default="output", help="save outputs path")
parser.add_argument("--device_id", type=int, default=0, help="310 device id")
args = parser.parse_args()
REWORD_SCOPE = 16.2736044

def preprocess(result_path):
    """
        preprocess
        Args:
            result_path(str): data path
    """
    env = gym.make('Pendulum-v0')
    env = env.unwrapped
    env.seed(1)
    rewards = []
    for _ in range(5):
        state = env.reset()
        reward_sum = 0
        for j in range(100):
            state = np.expand_dims(state, axis=0)
            state = state.astype(np.float32)
            state.tofile(os.path.join(result_path, "state.bin"))
            os.system("{} --model_path={} --output_path={} --device_id={}".format(args.run_main,
                                                                                  args.model_path,
                                                                                  args.output_path,
                                                                                  args.device_id))
            action = np.fromfile(os.path.join(result_path, "action.bin"), dtype=np.float32).reshape((1, 1))
            action = np.asarray(action)
            action = action[np.argmax(action)]
            next_state, reward, _, _ = env.step(action)
            reward_sum += reward
            state = next_state
            print('step', j + 1)
            if j == 99:
                print(reward_sum / REWORD_SCOPE)
                rewards.append(reward_sum / REWORD_SCOPE)
    np.savetxt('rewards.txt', rewards)


if __name__ == '__main__':
    preprocess(args.output_path)
