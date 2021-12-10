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
"""Preprocess NAML."""
import os

from model_utils.config import config
from src.dataset import MINDPreprocess
from src.dataset import create_eval_dataset, EvalNews, EvalUsers, EvalCandidateNews

def export_bin():
    '''pre process function.'''
    config.phase = "eval"
    config.neg_sample = config.eval_neg_sample
    config.embedding_file = os.path.join(config.dataset_path, config.embedding_file)
    config.word_dict_path = os.path.join(config.dataset_path, config.word_dict_path)
    config.category_dict_path = os.path.join(config.dataset_path, config.category_dict_path)
    config.subcategory_dict_path = os.path.join(config.dataset_path, config.subcategory_dict_path)
    config.uid2index_path = os.path.join(config.dataset_path, config.uid2index_path)
    config.train_dataset_path = os.path.join(config.dataset_path, config.train_dataset_path)
    config.eval_dataset_path = os.path.join(config.dataset_path, config.eval_dataset_path)
    args = config
    mindpreprocess = MINDPreprocess(vars(args), dataset_path=args.eval_dataset_path)
    base_path = args.preprocess_path

    data_dir = base_path + '/news_test_data/'
    news_id_folder = base_path + "/news_id_data/"
    if not os.path.exists(news_id_folder):
        os.makedirs(news_id_folder)
    print("======== news_id_folder is ", news_id_folder, flush=True)

    category_folder = os.path.join(data_dir, "00_category_data")
    if not os.path.exists(category_folder):
        os.makedirs(category_folder)
    print("======== category_folder is ", category_folder, flush=True)

    subcategory_folder = os.path.join(data_dir, "01_subcategory_data")
    if not os.path.exists(subcategory_folder):
        os.makedirs(subcategory_folder)

    title_folder = os.path.join(data_dir, "02_title_data")
    if not os.path.exists(title_folder):
        os.makedirs(title_folder)

    abstract_folder = os.path.join(data_dir, "03_abstract_data")
    if not os.path.exists(abstract_folder):
        os.makedirs(abstract_folder)
    dataset = create_eval_dataset(mindpreprocess, EvalNews, batch_size=args.batch_size)
    iterator = dataset.create_dict_iterator(output_numpy=True)
    idx = 0
    for idx, data in enumerate(iterator):
        news_id = data["news_id"]
        category = data["category"]
        subcategory = data["subcategory"]
        title = data["title"]
        abstract = data["abstract"]
        file_name = "naml_news_" + str(idx) + ".bin"
        news_id_file_path = os.path.join(news_id_folder, file_name)
        news_id.tofile(news_id_file_path)
        category_file_path = os.path.join(category_folder, file_name)
        category.tofile(category_file_path)
        subcategory_file_path = os.path.join(subcategory_folder, file_name)
        subcategory.tofile(subcategory_file_path)
        title_file_path = os.path.join(title_folder, file_name)
        title.tofile(title_file_path)
        abstract_file_path = os.path.join(abstract_folder, file_name)
        abstract.tofile(abstract_file_path)

    data_dir = base_path + '/users_test_data/'
    user_id_folder = os.path.join(data_dir, "00_user_id_data")
    print("======== user_id_folder is ", user_id_folder)
    if not os.path.exists(user_id_folder):
        os.makedirs(user_id_folder)

    history_folder = os.path.join(data_dir, "01_history_data")
    if not os.path.exists(history_folder):
        os.makedirs(history_folder)

    dataset = create_eval_dataset(mindpreprocess, EvalUsers, batch_size=args.batch_size)
    iterator = dataset.create_dict_iterator(output_numpy=True)

    for idx, data in enumerate(iterator):
        user_id = data["uid"]
        history = data["history"]
        file_name = "naml_users_" + str(idx) + ".bin"
        user_id_file_path = os.path.join(user_id_folder, file_name)
        user_id.tofile(user_id_file_path)
        history_file_path = os.path.join(history_folder, file_name)
        history.tofile(history_file_path)
    data_dir = base_path + '/browsed_news_test_data/'
    user_id_folder = os.path.join(data_dir, "00_user_id_data")
    if not os.path.exists(user_id_folder):
        os.makedirs(user_id_folder)

    candidate_nid_folder = os.path.join(data_dir, "01_candidate_nid_data")
    if not os.path.exists(candidate_nid_folder):
        os.makedirs(candidate_nid_folder)

    labels_folder = os.path.join(data_dir, "02_labels_data")
    print("======== labels_folder is ", labels_folder)
    if not os.path.exists(labels_folder):
        os.makedirs(labels_folder)
    dataset = create_eval_dataset(mindpreprocess, EvalCandidateNews, batch_size=args.batch_size)
    iterator = dataset.create_dict_iterator(output_numpy=True)
    for idx, data in enumerate(iterator):
    # 'uid', 'candidate_nid', 'labels'
        uid = data["uid"]
        candidate_nid = data["candidate_nid"]
        labels = data["labels"]
        file_name = "naml_browsed_news_" + str(idx) + ".bin"
        user_id_file_path = os.path.join(user_id_folder, file_name)
        uid.tofile(user_id_file_path)
        candidate_nid_file_path = os.path.join(candidate_nid_folder, file_name)
        candidate_nid.tofile(candidate_nid_file_path)
        labels_file_path = os.path.join(labels_folder, file_name)
        labels.tofile(labels_file_path)

if __name__ == "__main__":
    export_bin()
