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
"""callback utils to record info and restore it into summary file"""

import mindspore as ms
from mindspore import Tensor
from mindspore_rl.utils.callback import Callback


class RecordCb(Callback):

    def __init__(self, summary_dir, interval, eval_rate, times):
        self._summary_dir = summary_dir
        self.interval = interval
        self.iswin_buffer = []
        self.dead_allies_buffer = []
        self.dead_enemies_buffer = []
        if not isinstance(eval_rate, int) or eval_rate < 0:
            raise ValueError(
                "The arg of 'evaluation_frequency' must be int and >= 0, but get ",
                eval_rate)
        self._eval_rate = eval_rate
        self.times = times

    def __enter__(self):
        self.summary_record = ms.SummaryRecord(self._summary_dir)
        return self

    def __exit__(self, *exc_args):
        self.summary_record.close()

    def begin(self, params):
        params.eval_rate = self._eval_rate

    def episode_end(self, params):
        iswin, dead_allies, dead_enemies = params.others
        self.iswin_buffer.append(iswin.asnumpy()[0])
        self.dead_allies_buffer.append(dead_allies.asnumpy()[0])
        self.dead_enemies_buffer.append(dead_enemies.asnumpy()[0])
        if (params.cur_episode + 1) % self.interval == 0:
            self.summary_record.add_value(
                "scalar", 'win_rate',
                Tensor(sum(self.iswin_buffer) / self.interval))
            self.summary_record.add_value(
                "scalar", 'avg_dead_allies',
                Tensor(sum(self.dead_allies_buffer) / self.interval))
            self.summary_record.add_value(
                "scalar", 'avg_dead_enemies',
                Tensor(sum(self.dead_enemies_buffer) / self.interval))
            self.iswin_buffer = []
            self.dead_allies_buffer = []
            self.dead_enemies_buffer = []
        if self._eval_rate != 0 and params.cur_episode > 0 and \
                params.cur_episode % self._eval_rate == 0:
            eval_iswin_buffer = []
            eval_dead_alies_buffer = []
            eval_dead_enemies_buffer = []
            for _ in range(self.times):
                iswin, dead_allies, dead_enemies = params.evaluate()
                eval_iswin_buffer.append(iswin.asnumpy()[0])
                eval_dead_alies_buffer.append(dead_allies.asnumpy()[0])
                eval_dead_enemies_buffer.append(dead_enemies.asnumpy()[0])
            self.summary_record.add_value(
                "scalar", 'eval_win_rate',
                Tensor(sum(eval_iswin_buffer) / self.times))
            self.summary_record.add_value(
                "scalar", 'eval_avg_dead_allies',
                Tensor(sum(eval_dead_alies_buffer) / self.times))
            self.summary_record.add_value(
                "scalar", 'eval_avg_dead_enemies',
                Tensor(sum(eval_dead_enemies_buffer) / self.times))
        loss = params.loss
        # loss check already been done in LossCallback
        self.summary_record.add_value("scalar", 'loss', loss)
        reward = params.total_rewards
        self.summary_record.add_value("scalar", 'reward', reward)
        self.summary_record.record(params.cur_episode)


class EvalCb(Callback):

    def __init__(self, times):
        self.times = times

    def begin(self, params):
        params.eval_rate = 1

    def episode_end(self, params):
        eval_iswin_buffer = []
        eval_dead_alies_buffer = []
        eval_dead_enemies_buffer = []
        for _ in range(self.times):
            iswin, dead_allies, dead_enemies = params.evaluate()
            eval_iswin_buffer.append(iswin.asnumpy()[0])
            eval_dead_alies_buffer.append(dead_allies.asnumpy()[0])
            eval_dead_enemies_buffer.append(dead_enemies.asnumpy()[0])
        print("-----------------------------------------")
        print("Evaluation of {} episodes finish".format(self.times))
        print('eval_avg_win_rate:{}'.format(
            sum(eval_iswin_buffer) / self.times))
        print('eval_avg_dead_alies:{}'.format(
            sum(eval_dead_alies_buffer) / self.times))
        print('eval_avg_dead_enemies:{}'.format(
            sum(eval_dead_enemies_buffer) / self.times))
        print("-----------------------------------------")
