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

"""config"""
import argparse
import ast


def parse_args():
    """parse_args"""
    parser = argparse.ArgumentParser(description='Recommend System with DIEN')
    # path
    parser.add_argument('--mindrecord_path', type=str, default='./dataset_mindrecord/',
                        help='mindrecord format dataset path')
    parser.add_argument('--dataset_type', type=str, default='Books', help='dataset type')
    parser.add_argument('--dataset_file_path', type=str, default='./Books/', help='dataset files path')
    parser.add_argument('--pretrained_ckpt_path', type=str, default=None, help='Pretrained ckpt path')
    parser.add_argument('--binary_files_path', type=str, default='./ascend310_data',
                        help='the generated binary files path')
    parser.add_argument('--target_binary_files_path', type=str, default=None, help='the target binary files path')
    parser.add_argument('--save_checkpoint_path', type=str, default=None, help='checkpoint path')
    parser.add_argument('--train_mindrecord_path', type=str, default='./dataset_mindrecord/Books_train.mindrecord',
                        help='train dataset mindrecord path')
    parser.add_argument('--test_mindrecord_path', type=str, default='./dataset_mindrecord/Books_test.mindrecord',
                        help='test dataset mindrecord path')
    parser.add_argument('--MINDIR_file_name', type=str, default='DIEN_Electronics', help='MINDIR file path')

    # modelArt args
    parser.add_argument('--is_modelarts', type=ast.literal_eval, default=False)
    parser.add_argument('--data_url', type=str, default=None, help='Dataset path in the obs')
    parser.add_argument('--train_url', type=str, default=None, help='Training outputs path in the obs')
    parser.add_argument('--run_distribute', type=ast.literal_eval, default=False, help='Run distribute')

    # train args
    parser.add_argument('--epoch_size', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_len', type=int, default=100)
    parser.add_argument('--base_lr', type=float, default=0.001, help='Base learning rate')

    # device args
    parser.add_argument('--device_target', type=str, default='Ascend', help='Device target')
    parser.add_argument('--device_id', type=int, default=0)

    args = parser.parse_args()
    return args
