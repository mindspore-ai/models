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
"""
sample script of preprocessing txt data for autodis infer
"""
import collections
import pickle
import os
import argparse
class StatsDict():
    """preprocessed data"""

    def __init__(self, field_size, dense_dim, slot_dim, skip_id_convert):
        self.field_size = field_size
        self.dense_dim = dense_dim
        self.slot_dim = slot_dim
        self.skip_id_convert = bool(skip_id_convert)

        self.val_cols = ["val_{}".format(i + 1) for i in range(self.dense_dim)]
        self.cat_cols = ["cat_{}".format(i + 1) for i in range(self.slot_dim)]

        self.val_min_dict = {col: 0 for col in self.val_cols}
        self.val_max_dict = {col: 0 for col in self.val_cols}

        self.cat_count_dict = {col: collections.defaultdict(int) for col in self.cat_cols}

        self.oov_prefix = "OOV"

        self.cat2id_dict = {}
        self.cat2id_dict.update({col: i for i, col in enumerate(self.val_cols)})
        self.cat2id_dict.update(
            {self.oov_prefix + col: i + len(self.val_cols) for i, col in enumerate(self.cat_cols)})

    def load_dict(self, dict_path, prefix=""):
        with open(os.path.join(dict_path, "{}val_max_dict.pkl".format(prefix)), "rb") as file_wrt:
            self.val_max_dict = pickle.load(file_wrt)
        with open(os.path.join(dict_path, "{}val_min_dict.pkl".format(prefix)), "rb") as file_wrt:
            self.val_min_dict = pickle.load(file_wrt)
        with open(os.path.join(dict_path, "{}cat_count_dict.pkl".format(prefix)), "rb") as file_wrt:
            self.cat_count_dict = pickle.load(file_wrt)
        print("val_max_dict.items()[:50]:{}".format(list(self.val_max_dict.items())))
        print("val_min_dict.items()[:50]:{}".format(list(self.val_min_dict.items())))

    def get_cat2id(self, threshold=100):
        for key, cat_count_d in self.cat_count_dict.items():
            new_cat_count_d = dict(filter(lambda x: x[1] > threshold, cat_count_d.items()))
            for cat_str, _ in new_cat_count_d.items():
                self.cat2id_dict[key + "_" + cat_str] = len(self.cat2id_dict)
        print("cat2id_dict.size:{}".format(len(self.cat2id_dict)))
        print("cat2id.dict.items()[:50]:{}".format(list(self.cat2id_dict.items())[:50]))

    def map_cat2id(self, values, cats):
        """Cat to id"""

        def minmax_scale_value(i, val):
            max_v = float(self.val_max_dict["val_{}".format(i + 1)])
            return float(val) * 1.0 / max_v

        id_list = []
        weight_list = []
        for i, val in enumerate(values):
            if val == "":
                id_list.append(i)
                weight_list.append(0)
            else:
                key = "val_{}".format(i + 1)
                id_list.append(self.cat2id_dict[key])
                weight_list.append(minmax_scale_value(i, float(val)))

        for i, cat_str in enumerate(cats):
            key = "cat_{}".format(i + 1) + "_" + cat_str
            if key in self.cat2id_dict:
                if self.skip_id_convert is True:
                    # For the synthetic data, if the generated id is between [0, max_vcoab], but the num examples is l
                    # ess than vocab_size/ slot_nums the id will still be converted to [0, real_vocab], where real_vocab
                    # the actually the vocab size, rather than the max_vocab. So a simple way to alleviate this
                    # problem is skip the id convert, regarding the synthetic data id as the final id.
                    id_list.append(cat_str)
                else:
                    id_list.append(self.cat2id_dict[key])
            else:
                id_list.append(self.cat2id_dict[self.oov_prefix + "cat_{}".format(i + 1)])
            weight_list.append(1.0)
        return id_list, weight_list

def parse_args():
    """set and check parameters."""
    parser = argparse.ArgumentParser(description="autodis process")
    parser.add_argument('--data_dir', type=str, default='../data/input/origin_data')
    parser.add_argument('--dst_dir', type=str, default='../data/input')
    parser.add_argument('--data_input', type=str, default="test.txt")
    parser.add_argument('--dense_dim', type=int, default=13)
    parser.add_argument('--slot_dim', type=int, default=26)
    parser.add_argument("--skip_id_convert", type=int, default=0)
    parser.add_argument("--threshold", type=int, default=100)
    args, _ = parser.parse_known_args()
    return args

def run():
    """
    preprocessing txt data
    """
    args = parse_args()
    stats = StatsDict(field_size=args.dense_dim+args.slot_dim, dense_dim=args.dense_dim, \
                        slot_dim=args.slot_dim, skip_id_convert=args.skip_id_convert)
    stats.load_dict(dict_path="./stats_dict", prefix="")
    stats.get_cat2id(threshold=args.threshold)
    fi = open(os.path.join(args.data_dir, args.data_input), "r")
    fo1 = open(os.path.join(args.dst_dir, "label.txt"), "w")
    fo2 = open(os.path.join(args.dst_dir, "ids.txt"), "w")
    fo3 = open(os.path.join(args.dst_dir, "wts.txt"), "w")
    for line in fi:
        line = line.strip("\n")
        items = line.split("\t")
        label = float(items[0])
        values = items[1:1 + args.dense_dim]
        cats = items[1 + args.dense_dim:]
        ids, wts = stats.map_cat2id(values, cats)
        fo1.write(str(int(label))+"\n")
        fo2.write("\t".join(str(id) for id in ids)+"\n")
        fo3.write("\t".join(str(wt) for wt in wts)+"\n")
    fo1.close()
    fo2.close()
    fo3.close()

if __name__ == '__main__':
    run()
