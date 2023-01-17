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
from multiprocessing import Process, Queue

import numpy as np

from Env import KESEnv


class KESEnvCo(KESEnv):
    def __init__(self, dataset, model_name='CoKT', dataset_name='assist09', workers=4):
        super(KESEnvCo, self).__init__(dataset, model_name, dataset_name)
        self.dataset = dataset
        self.his = None

        self.process_list = []
        self.worker_queue = Queue()
        self.index_queue = Queue()
        for _ in range(workers):
            p = Process(target=self.worker, args=(self.worker_queue, self.index_queue))  # 实例化进程对象
            p.start()
            self.process_list.append(p)

    def begin_episode(self, targets, initial_logs):
        self.put_index(initial_logs)
        logs = self.get_next(initial_logs)
        return super(KESEnvCo, self).begin_episode(logs)

    def worker(self, q1: Queue, q2: Queue):
        while True:
            if not q2.empty():
                skill = q2.get()
                q1.put(self.dataset.get_query(-1, skill, range(self.his_len(), len(skill))))

    @staticmethod
    def collate_fn(data_):
        r_his, r_skill_y, r_len = zip(*data_)
        r_his, r_skill_y, r_len = [np.concatenate(_, 0) for _ in (r_his, r_skill_y, r_len)]
        return r_his, r_skill_y, r_len

    def get_next(self, items):
        batch_size = items.shape[0]
        r = []
        while len(r) < batch_size:
            try:
                x = self.worker_queue.get()
                r.append(x)
            except TimeoutError:
                continue
        r_his, r_skill_y, r_len = self.collate_fn(r)
        return items, r_his, r_skill_y, r_len

    def put_index(self, index, add_to_his=True):
        index = index.asnumpy()
        if not self.his is None:
            index = np.concatenate((self.his, index), axis=1)
        if add_to_his:
            self.his = index
        for i in index:
            self.index_queue.put(i)

    def his_len(self):
        return 0 if self.his is None else self.his.shape[1]

    def __del__(self):
        for p in self.process_list:
            p.terminate()
