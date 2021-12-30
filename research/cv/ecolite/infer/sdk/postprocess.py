# Copyright(C) 2021. Huawei Technologies Co.,Ltd
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

"""post process for 310 inference"""
import os
import argparse
import pickle
import numpy as np

parser = argparse.ArgumentParser(description="ecolite inference")
parser.add_argument("--result_path", type=str, required=True, help="result files path.")
parser.add_argument("--label_path", type=str, required=True, help="image file path.")
parser.add_argument('--batch_size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
args = parser.parse_args()
batch_size = args.batch_size
num_classes = 101


def get_result(result_path, label_path):
    """Get final accuracy result"""
    files = os.listdir(result_path)
    top1 = 0
    top5 = 0
    total_data = batch_size * len(files)
    for i in range(len(files)):
        video_label_id = files[i].split('_')[-2]
        file = 'eval_predict_' + str(video_label_id) + '_.bin'
        data_path = os.path.join(result_path, file)
        result = np.fromfile(data_path, dtype=np.float32).reshape(batch_size, num_classes)
        predict = np.argsort(result, axis=-1)[:, -5:]
        print("the post-processing result of the file", file, ":")
        print(predict)
        label_file_path = label_path + 'eval_label_'
        label_file_path += str(video_label_id)
        label_file_path += '.pkl'
        pkllabelfile = open(label_file_path, 'rb')
        label = pickle.load(pkllabelfile)
        for batch in range(batch_size):
            if predict[batch][-1] == label[batch]:
                top1 += 1
                top5 += 1
            elif label[batch] in predict[batch][-5:]:
                top5 += 1
    print(f"Total data: {total_data}, top1 accuracy: {top1 / total_data}, top5 accuracy: {top5 / total_data}.")


if __name__ == '__main__':
    get_result(args.result_path, args.label_path)
