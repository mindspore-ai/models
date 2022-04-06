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

import csv
import os
import stat

from config import config
from dataset import create_eval_dataset, EvalNews, EvalUsers, EvalCandidateNews, MINDPreprocess


def do_Convert(infer_path):
    """ converter for mxbase dataset """
    if config.neg_sample == 4:
        config.neg_sample = -1
    if config.batch_size != 1:
        config.batch_size = 1
    config.embedding_file = os.path.join(config.dataset_path, config.embedding_file)
    config.word_dict_path = os.path.join(config.dataset_path, config.word_dict_path)
    config.category_dict_path = os.path.join(config.dataset_path, config.category_dict_path)
    config.subcategory_dict_path = os.path.join(config.dataset_path, config.subcategory_dict_path)
    config.uid2index_path = os.path.join(config.dataset_path, config.uid2index_path)

    mindpreprocess = MINDPreprocess(vars(config),
                                    dataset_path=os.path.join(config.dataset_path,
                                                              "MIND{}_dev".format(config.dataset)))
    news_data = create_dataset(mindpreprocess, EvalNews, 1)
    user_data = create_dataset(mindpreprocess, EvalUsers, 1)
    eval_data = create_dataset(mindpreprocess, EvalCandidateNews, 1)
    create_newsdata2file(news_data, infer_path)
    create_userdata2file(user_data, infer_path)
    create_evaldata2file(eval_data, infer_path)
    print("===create mxbase_data success====", end='\r')


def create_newsdata2file(news_data, infer_path):
    """ news_data convert to csv file """
    iterator = news_data.create_dict_iterator(output_numpy=True)
    news_dataset_size = news_data.get_dataset_size()
    rows = []
    for count, data in enumerate(iterator):
        row = [data["news_id"][0][0], data["category"][0][0], data["subcategory"][0][0]]
        title_num = " ".join([str(num) for num in data["title"][0]])
        row.append(title_num)
        abstract_num = " ".join([str(num) for num in data["abstract"][0]])
        row.append(abstract_num)
        rows.append(row)
        print(f"===create News data==== [ {count} / {news_dataset_size} ]", end='\r')

    filepath = os.path.join(infer_path, "mxbase_data/newsdata.csv")
    if not os.path.exists(filepath):
        os.makedirs(os.path.join(infer_path, "mxbase_data/"))
    with open(filepath, 'w', newline='') as inf:
        writer = csv.writer(inf)
        writer.writerows(rows)
    inf.close()
    os.chmod(filepath, stat.S_IRWXO)
    print(f"===create news data==== [ {news_dataset_size} / {news_dataset_size} ]")


def create_userdata2file(user_data, infer_path):
    """ user_data convert to csv file """
    iterator = user_data.create_dict_iterator(output_numpy=True)
    user_dataset_size = user_data.get_dataset_size()
    rows = []
    for count, data in enumerate(iterator):
        row = [data["uid"][0]]
        for newses in data["history"]:
            hisstr = " ".join([str(num[0]) for num in newses])
            row.append(hisstr)
        rows.append(row)
        print(f"===create user data==== [ {count} / {user_dataset_size} ]", end='\r')

    filepath = os.path.join(infer_path, "mxbase_data/userdata.csv")
    with open(filepath, 'w', newline='') as t:
        writer = csv.writer(t)
        writer.writerows(rows)
    t.close()
    print(f"===create user data==== [ {user_dataset_size} / {user_dataset_size} ]", end='\r')


def create_evaldata2file(eval_data, infer_path):
    """Prediction data convert to csv file """
    iterator = eval_data.create_dict_iterator(output_numpy=True)
    eval_data_size = eval_data.get_dataset_size()
    rows = []
    for count, data in enumerate(iterator):
        row = []
        row.append(data["uid"])
        candidate_nid = " ".join([str(nid) for nid in data["candidate_nid"]])
        row.append(candidate_nid)
        labels = " ".join([str(label) for label in data["labels"]])
        row.append(labels)
        rows.append(row)
        print(f"===create eval data==== [ {count} / {eval_data_size} ]", end='\r')
    filepath = os.path.join(infer_path, "mxbase_data/evaldata.csv")
    with open(filepath, 'w', newline='') as t:
        writer = csv.writer(t)
        writer.writerows(rows)
    t.close()
    print(f"===create eval data==== [ {eval_data_size} / {eval_data_size} ]", end='\r')


def create_dataset(mindpreprocess, datatype, batch_size):
    """create_dataset"""
    dataset = create_eval_dataset(mindpreprocess, datatype, batch_size)
    return dataset


if __name__ == '__main__':
    filePath = os.path.abspath(os.path.dirname(__file__))
    input_path = os.path.abspath(os.path.join(filePath, "../../data/"))
    do_Convert(input_path)
