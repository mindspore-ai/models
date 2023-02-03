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
import numpy as np
from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm


def save_to_es(dataset='assist09'):
    es = Elasticsearch(hosts=['http://localhost:9200/']).options(
        request_timeout=20,
        retry_on_timeout=True,
        ignore_status=[400, 404]
    )

    def handler_es():
        data_dir = f"./data/{dataset}/{dataset}.npz"
        with np.load(data_dir, allow_pickle=True) as data:
            y, skill, real_len = data['y'], data['skill'], data['real_len']
        users = range(len(real_len))
        train_num = int(0.8 * len(users))
        train_test_split = {'train': users[:train_num], 'test': users[train_num:]}
        for mode, users_mode in train_test_split.items():
            index = f'{dataset}_{mode}'
            es.indices.delete(index=index)
            es.indices.create(index=index)
            for sid in tqdm(users_mode):
                skill_str = skill[sid][:real_len[sid]].astype(str)
                history = ''
                for rid in range(real_len[sid]):
                    if rid == 0:
                        history = skill_str[rid]
                    else:
                        history = f'{history} {skill_str[rid]}'
                    action = {'_index': index, 'user': sid, 'loc': rid, 'skill': skill[sid][rid],
                              'y': y[sid][:rid + 1], 'history': history}
                    yield action

    helpers.bulk(client=es, actions=handler_es())
    print("Save to ES Finished!")
    es.close()


def check_es(dataset='assist09'):
    es = Elasticsearch(hosts=['http://localhost:9200/']).options(
        request_timeout=20,
        retry_on_timeout=True,
        ignore_status=[400, 404]
    )
    result = es.search(index=f'{dataset}_train')
    print(result)
    queries = []
    np.random.seed(1)
    skill = np.random.randint(0, 10, size=(20,))
    print(skill)
    query = [{'index': f'{dataset}_train'},
             {'query': {
                 'bool':
                     {
                         'must':
                             [
                                 {'term': {'skill': skill[-1]}},
                                 {'match': {'history': ' '.join(skill.astype('str'))}}
                             ],
                         'must_not':
                             {'term': {'user': 23}}
                     }}}]
    queries += query
    skill = np.random.randint(0, 10, size=(20,))
    print(skill)
    query = [{'index': f'{dataset}_train'},
             {
                 'query': {
                     'bool':
                         {
                             'must':
                                 [
                                     {'term': {'skill': skill[-1]}},
                                     {'match': {'history': ' '.join(skill.astype('str'))}}
                                 ],
                             'must_not':
                                 {'term': {'user': 33}}
                         }}
             }]
    queries += query
    result = es.msearch(searches=queries)
    print(result)


if __name__ == '__main__':
    dataset_ = 'junyi'
    save_to_es(dataset_)
    check_es(dataset_)
