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
grid env
"""
import math

import numpy as np
from gym import spaces
from gym import Env


class MultiStepTwoGoalPointEnv(Env):
    def __init__(self):
        self.threshold = 10.0
        high = np.array([self.threshold, self.threshold], dtype=np.float32)
        low = np.array([0.0, 0.0], dtype=np.float32)
        self.goal1 = [5.0, 10.0]
        self.goal2 = [10.0, 5.0]
        self.observation_space = spaces.Box(low=low, high=high, shape=(2,))
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        self.state = [0.0, 0.0]
        self.time = 1.0
        self.cnt_step = 0
        self.step_max = 12

    def step(self, action):
        action = (action + 1) * 45.0
        action = max(0.0, action)
        action = min(90.0, action)
        x = self.state[0]
        y = self.state[1]
        costheta = math.cos(action * math.pi / 180.0)
        sintheta = math.sin(action * math.pi / 180.0)
        x = x + costheta * self.time
        y = y + sintheta * self.time
        done = False
        self.cnt_step += 1
        self.state = np.array([x, y])
        reward = 0
        dist = 0
        if self.cnt_step == self.step_max:
            dist1 = ((x - self.goal1[0]) ** 2 + (y - self.goal1[1]) ** 2) ** 0.3
            dist2 = ((x - self.goal2[0]) ** 2 + (y - self.goal2[1]) ** 2) ** 0.3
            dist = dist1 + dist2
            reward1 = 1.0 / (dist1 + 0.5)
            reward2 = 1.0 / (dist2 + 0.5)
            reward = reward1 + reward2
            done = True
        return self.state, reward, done, {'dist', dist}

    def reset(self):
        self.state = np.zeros((1, 2), dtype=np.float32)[0]
        self.cnt_step = 0
        return self.state
