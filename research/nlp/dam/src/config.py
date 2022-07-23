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
# ===========================================================================
"""Net Config"""
import argparse
import ast
import six


def parse_args():
    """
    Deep Attention Matching Network Training Args. Default:Ubuntu Corpus
    """
    parser = argparse.ArgumentParser("DAM Training Args")
    parser.add_argument('--model_name', type=str, default="DAM_ubuntu", help='The model name.')  # douban: DAM_douban
    parser.add_argument('--device_target', type=str, default="Ascend", help="run platform, only support Ascend")
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--parallel', type=ast.literal_eval, default=False)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--is_emb_init', type=ast.literal_eval, default=True)
    parser.add_argument('--do_eval', type=ast.literal_eval, default=False,
                        help="Whether side training changes verification.")
    parser.add_argument('--file_format', type=str, default='MINDIR')
    # net args
    parser.add_argument('--max_turn_num', type=int, default=9)
    parser.add_argument('--max_turn_len', type=int, default=50)
    parser.add_argument('--stack_num', type=int, default=5)
    parser.add_argument('--attention_type', type=str, default="dot")
    parser.add_argument('--vocab_size', type=int, default=434512)  # 172130 for douban data
    parser.add_argument('--emb_size', type=int, default=200)
    parser.add_argument('--channel1_dim', type=int, default=32)  # 16 for douban data
    parser.add_argument('--channel2_dim', type=int, default=16)
    parser.add_argument('--is_mask', type=ast.literal_eval, default=True)
    parser.add_argument('--is_layer_norm', type=ast.literal_eval, default=True)
    parser.add_argument('--is_positional', type=ast.literal_eval, default=False)
    parser.add_argument('--drop_attention', type=int, default=None)
    # path
    parser.add_argument('--data_root', type=str, default="./data/ubuntu")  # douban: ./data/douban
    parser.add_argument('--output_path', type=str, default="./output/ubuntu")  # douban: ./output/duban
    parser.add_argument('--train_data', type=str, default="data_train.mindrecord")
    parser.add_argument('--eval_data', type=str, default="data_val.mindrecord")
    parser.add_argument('--test_data', type=str, default="data_test.mindrecord")
    parser.add_argument('--emb_init', type=str, default='word_embedding.pkl')
    parser.add_argument('--loss_file_name', type=str, default="loss.log")
    parser.add_argument('--eval_file_name', type=str, default="eval.log")
    parser.add_argument('--ckpt_path', type=str, default=None, help="the path of checkpoint,")
    parser.add_argument('--ckpt_name', type=str, default=None, help="the name of checkpoint,")
    # train args
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training.')
    parser.add_argument('--eval_batch_size', type=int, default=200, help='Batch size for training.')  # douban: 256
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate used to train.')
    parser.add_argument('--decay_rate', type=float, default=0.9, help='The decay rate.')
    parser.add_argument('--decay_steps', type=int, default=400, help='A value used to calculate decayed learning rate.')
    parser.add_argument('--loss_scale', type=float, default=1.0)
    parser.add_argument('--epoch_size', type=int, default=2)
    # moderArt args
    parser.add_argument('--modelArts', type=ast.literal_eval, default=False,
                        help="Whether to use modelArts for training")
    parser.add_argument('--data_url', type=str, default=None, help='Location of data.')
    parser.add_argument('--train_url', type=str, default=None, help='Location of training outputs.')

    parser.add_argument('--version', type=str, default="V001", help="Training version.")

    args = parser.parse_args()
    return args


def print_arguments(args):
    """
    Print Config
    """
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(six.iteritems(vars(args))):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


if __name__ == "__main__":
    print_arguments(parse_args())
