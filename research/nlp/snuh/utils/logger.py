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
"""Logger module."""
import math
import statistics as stat

class Logger:
    """Logger class"""
    def log(self, string):
        print(string)

    def log_perfs(self, perfs):
        """print log"""
        valid_perfs = [perf for perf in perfs if not math.isinf(perf)]
        best_perf = max(valid_perfs)
        self.log('-' * 89)
        self.log('%d perfs: %s' % (len(perfs), str(perfs)))
        self.log('perf max: %g' % best_perf)
        self.log('perf min: %g' % min(valid_perfs))
        self.log('perf avg: %g' % stat.mean(valid_perfs))
        self.log('perf std: %g' % (stat.stdev(valid_perfs)\
            if len(valid_perfs) > 1 else 0.0))
        self.log('(excluded %d out of %d runs that produced -inf)' %
                 (len(perfs) - len(valid_perfs), len(perfs)))
        self.log('-' * 89)
