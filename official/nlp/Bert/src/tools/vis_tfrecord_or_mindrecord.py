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
from argparse import ArgumentParser
import tensorflow as tf
import mindspore.mindrecord as mm

def vis_tfrecord(file_name):
    tfe = tf.contrib.eager
    tfe.enable_eager_execution()
    raw_dataset = tf.data.TFRecordDataset(file_name)
    # raw_dataset is iterator: you can use raw_dataset.take(n) to get n data
    for raw_record in raw_dataset:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        print("Start print tfrecord example:", example, flush=True)


def vis_mindrecord(file_name):
    fr = mm.FileReader(file_name, num_consumer=4)
    for data in fr.get_next():
        print("value is:", data)
        # you can use break here to get one data
    fr.close()


def main():
    """
    vis tfrecord or vis mindrecord
    """
    parser = ArgumentParser(description='vis tfrecord or vis mindrecord.')
    parser.add_argument("--file_name", type=str, default='./output/AE_wiki_00.tfrecord', help="the file name.")
    parser.add_argument("--vis_option", type=str, default='vis_tfrecord', choices=['vis_tfrecord', 'vis_mindrecord'],
                        help="option of transfer vis_tfrecord or vis_mindrecord, default is vis_tfrecord.")
    args = parser.parse_args()
    if args.vis_option == 'vis_tfrecord':
        print("start vis tfrecord: ", args.file_name, flush=True)
        vis_tfrecord(args.file_name)
    elif args.vis_option == 'vis_mindrecord':
        print("start vis mindrecord: ", args.file_name, flush=True)
        vis_mindrecord(args.file_name)
    else:
        raise ValueError("Unsupported vis option: ", args.vis_option)

if __name__ == "__main__":
    main()
