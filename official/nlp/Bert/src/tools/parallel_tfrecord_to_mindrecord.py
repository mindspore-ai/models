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
import os
from multiprocessing import Pool
from argparse import ArgumentParser
import mindspore.dataset as ds
from mindspore.mindrecord import FileWriter

def tf_2_mr(item):
    item_path = item
    if not os.path.exists(args.output_mindrecord_dir):
        os.makedirs(args.output_mindrecord_dir, exist_ok=True)
    mindrecord_path = os.path.join(args.output_mindrecord_dir,
                                   item[item.rfind('/') + 1:item.rfind('.')] + '.mindrecord')
    print("Start convert {} to {}.".format(item_path, mindrecord_path))
    writer = FileWriter(file_name=mindrecord_path, shard_num=1, overwrite=True)
    nlp_schema = {"input_ids": {"type": "int64", "shape": [-1]},
                  "input_mask": {"type": "int64", "shape": [-1]},
                  "segment_ids": {"type": "int64", "shape": [-1]},
                  "next_sentence_labels": {"type": "int64", "shape": [-1]},
                  "masked_lm_positions": {"type": "int64", "shape": [-1]},
                  "masked_lm_ids": {"type": "int64", "shape": [-1]},
                  "masked_lm_weights": {"type": "float32", "shape": [-1]}}
    writer.add_schema(nlp_schema, "it is a preprocessed nlp dataset")

    tf_objs = ds.TFRecordDataset(item_path, shuffle=False)
    data = []
    index = 0
    for tf_obj in tf_objs.create_dict_iterator(output_numpy=True):
        sample = {"input_ids": tf_obj["input_ids"],
                  "input_mask": tf_obj["input_mask"],
                  "segment_ids": tf_obj["segment_ids"],
                  "next_sentence_labels": tf_obj["next_sentence_labels"],
                  "masked_lm_positions": tf_obj["masked_lm_positions"],
                  "masked_lm_ids": tf_obj["masked_lm_ids"],
                  "masked_lm_weights": tf_obj["masked_lm_weights"]}
        data.append(sample)
        index += 1
        if index % 2000 == 0:
            writer.write_raw_data(data)
            data = []
    if data:
        writer.write_raw_data(data)

    writer.commit()
    print("Convert {} to {} success.".format(item_path, mindrecord_path))

def parse_args():
    parser = ArgumentParser(description="Parallel tfrecord to mindrecord")
    parser.add_argument("--pool_nums", type=int, default="8",
                        help="pool nums convert tfrecord to mindrecord, can increase this value to speed up.")
    parser.add_argument("--input_tfrecord_dir", type=str, default="",
                        help="The input data dir contain the .tfrecord files, it is best to use an absolute path.")
    parser.add_argument("--output_mindrecord_dir", type=str, default="",
                        help="The output data dir to save mindrecord, it is best to use an absolute path.")
    args_opt = parser.parse_args()
    return args_opt

args = parse_args()
if __name__ == "__main__":
    pool = Pool(args.pool_nums)
    files = os.listdir(args.input_tfrecord_dir)
    items = []
    for file_name in files:
        items.append(os.path.join(args.input_tfrecord_dir, file_name))
    for single_file in items:
        pool.apply_async(tf_2_mr, (single_file,))
    pool.close()
    pool.join()
