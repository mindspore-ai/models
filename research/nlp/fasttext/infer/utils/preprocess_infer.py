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
"""FastText for Evaluation"""
import argparse
import os
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as deC
from mindspore import context
import numpy as np

parser = argparse.ArgumentParser(description='FastText Classification')
parser.add_argument('--data_name', type=str, default='ag')
parser.add_argument('--device_target', default='CPU', type=str)
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--dataset_path', type=str)
parser.add_argument('--test_buckets', default=[467], type=list)
parser.add_argument('--outputdir', default='', type=str)
args = parser.parse_args()

if args.data_name == "ag":
    target_label1 = ['0', '1', '2', '3']
elif args.data_name == 'dbpedia':
    target_label1 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']
elif args.data_name == 'yelp_p':
    target_label1 = ['0', '1']

context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target=args.device_target)

def load_infer_dataset(batch_size, datafile, bucket):
    """data loader for infer"""
    def batch_per_bucket(bucket_length, input_file):
        input_file = input_file + '/test_dataset_bs_' + str(bucket_length) + '.mindrecord'
        if not input_file:
            raise FileNotFoundError("input file parameter must not be empty.")

        data_set = ds.MindDataset(input_file,
                                  columns_list=['src_tokens', 'src_tokens_length', 'label_idx'])
        type_cast_op = deC.TypeCast(mstype.int32)
        data_set = data_set.map(operations=type_cast_op, input_columns="src_tokens")
        data_set = data_set.map(operations=type_cast_op, input_columns="src_tokens_length")
        data_set = data_set.map(operations=type_cast_op, input_columns="label_idx")

        data_set = data_set.batch(batch_size, drop_remainder=False)
        return data_set
    for i, _ in enumerate(bucket):
        bucket_len = bucket[i]
        ds_per = batch_per_bucket(bucket_len, datafile)
        if i == 0:
            data_set = ds_per
        else:
            data_set = data_set + ds_per

    return data_set

def w2txt(file, data):
    with open(file, "w") as f:
        for i in range(data.shape[0]):
            s = ' '.join(str(num) for num in data[i])
            f.write(s+"\n")

if __name__ == '__main__':
    dataset = load_infer_dataset(batch_size=args.batch_size, datafile=args.dataset_path, bucket=args.test_buckets)
    src_tokens_sents = []
    target_sens_sents = []
    for batch in dataset.create_dict_iterator(output_numpy=True, num_epochs=1):
        src_tokens = batch['src_tokens'].astype(np.int32)
        target_sens = batch['label_idx'].astype(np.int32)
        src_tokens_shape = src_tokens.shape
        target_sens_shape = target_sens.shape
        for index in range(src_tokens_shape[0]):
            src_tokens_sents.append(src_tokens[index].astype(np.int32))
        for index in range(target_sens_shape[0]):
            target_sens_sents.append(target_sens[index].astype(np.int32))

    src_tokens_sents = np.array(src_tokens_sents).astype(np.int32)
    target_sens_sents = np.array(target_sens_sents).astype(np.int32)
    w2txt(os.path.join(args.outputdir, "src_tokens.txt"), src_tokens_sents)
    w2txt(os.path.join(args.outputdir, "target_sens.txt"), target_sens_sents)
