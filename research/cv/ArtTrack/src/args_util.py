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

import argparse
import re
from typing import List
import ast

from src.config import load_config, merge_a_into_b
from src.log import setup_log

TARGET_MPII_SINGLE = 'mpii_single'
TARGET_COCO_MULTI = 'coco_multi'
TARGET_TF2MS = 'tf2ms'
TARGET_MAT2JSON = 'mat2json'
TARGET_PAIRWISE = 'pairwise'


def to_number_or_str(value: str):
    if value is None:
        return None
    try:
        if re.match(r"^([-+])?\d+$", value) is not None:
            return int(value)

        return float(value)
    except ValueError:
        return value


def compose_option(option: List[str]):
    result = dict()
    for o in option:
        kv = o.split('=', 1)
        key = kv[0]
        value = kv[1] if len(kv) == 2 else None
        keys = key.split('.')
        cursor = result
        for k in keys[:-1]:
            last_cursor = cursor
            cursor = cursor.get(k, None)
            if cursor is None:
                cursor = dict()
                last_cursor[k] = cursor
        cursor[keys[-1]] = to_number_or_str(value)
    return result


def setup_config(args):
    cfg = None
    if args.config is not None:
        cfg = load_config(args.config)
    if args.option:
        option = compose_option(args.option)
        if cfg is not None:
            merge_a_into_b(option, cfg)
    return cfg


def command(func):
    def _command(p, a):
        if a.log:
            setup_log(a.log)
        return func(p, a, setup_config(a))

    return _command


def join_targets(targets):
    return ', '.join(targets)


def create_arg_parser():
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument('-c', '--config', type=str, nargs='?')
    common_parser.add_argument('--log', default="log.yaml", type=str, nargs='?', help="log config file")
    common_parser.add_argument('--option', type=str, nargs='+', help="extra option will override config file."
                                                                     "example: context.device_target=GPU")

    parser = argparse.ArgumentParser(description='tool')
    subparsers = parser.add_subparsers(metavar="COMMAND", dest='command')
    # preprocess
    parser_pre = subparsers.add_parser(
        'preprocess', aliases=['pre'], help='preprocess', parents=[common_parser])
    pre_targets = [TARGET_MPII_SINGLE, TARGET_TF2MS, TARGET_MAT2JSON, TARGET_PAIRWISE]
    parser_pre.add_argument('target', metavar='TARGET',
                            choices=pre_targets,
                            help='option choices: %s' % join_targets(pre_targets), nargs='?')
    # preprocess single
    pre_group_single = parser_pre.add_argument_group(TARGET_MPII_SINGLE)
    pre_group_single.add_argument('--dataset-dir', default='.', type=str, nargs='?')
    pre_group_single.add_argument('--dataset-name', default='mpii_human_pose_v1_u12_1', type=str, nargs='?')
    pre_group_single.add_argument('--save-dir', default=None, type=str, nargs='?')
    pre_group_single.add_argument('--image-dir', default=None, type=str, nargs='?')
    pre_group_single.add_argument('--split', default=False, action='store_true',
                                  help="split dataset to train and eval.")
    pre_group_single.add_argument('--eval-ratio', default=0.2, type=float, nargs='?')
    # preprocess tf2ms
    pre_group_tf2ms = parser_pre.add_argument_group(TARGET_TF2MS)
    pre_group_tf2ms.add_argument('--checkpoint', type=str, nargs='?', help='path to tf parameter')
    pre_group_tf2ms.add_argument('--output', default='out/tf2ms.ckpt', type=str, nargs='?')
    pre_group_tf2ms.add_argument('--map', default='config/tf2ms.json', type=str, nargs='?')

    pre_group_mat2json = parser_pre.add_argument_group(TARGET_MAT2JSON)
    pre_group_mat2json.add_argument('--index-mat', type=str, nargs='?', help='mat format index file path')
    pre_group_mat2json.add_argument('--name', type=str, nargs='?', help='field name in mat format index file.'
                                                                        'this option is also used as a filename'
                                                                        'for output')
    pre_group_mat2json.add_argument('--dataset-json', type=str, nargs='?', help='json format dataset file path.'
                                                                                'output dataset related to index,'
                                                                                'if this option appears')
    pre_group_mat2json.add_argument('--output-dir', type=str, nargs='?', help='all output will in this dir.')
    pre_group_mat2json.add_argument('--index-offset', type=int, nargs='?', default=-1)
    pre_group_mat2json.add_argument('--stdout', action='store_true', default=False)

    # train
    parser_train = subparsers.add_parser(
        'train', help='train', parents=[common_parser])
    train_targets = [TARGET_MPII_SINGLE, TARGET_COCO_MULTI]
    parser_train.add_argument('--device_target', type=str, nargs='?', default='GPU',
                              help="device_target:GPU,CPU,or Ascend")
    parser_train.add_argument('target', metavar='TARGET', choices=train_targets, default=TARGET_MPII_SINGLE,
                              help='option choices: %s' % join_targets(train_targets), nargs='?')

    parser_train.add_argument('--data_url', required=False,
                              default=None, help='Location of data.')
    parser_train.add_argument('--train_url', required=False,
                              default=None, help='Location of training outputs.')
    parser_train.add_argument('--device_id', required=False, default=1,
                              type=int, help='Location of training outputs.')
    parser_train.add_argument('--run_distribute', type=ast.literal_eval, required=False,
                              default=False, help='Location of training outputs.')
    parser_train.add_argument('--is_model_arts', type=ast.literal_eval,
                              default=False, help='Location of training outputs.')

    # eval
    parser_test = subparsers.add_parser(
        'eval', help='eval', parents=[common_parser])
    test_targets = [TARGET_MPII_SINGLE, TARGET_COCO_MULTI]
    test_group_single = parser_test.add_argument_group(TARGET_MPII_SINGLE)
    test_group_single.add_argument('target', metavar='TARGET', choices=test_targets,
                                   help='option choices: %s' % join_targets(test_targets), nargs='?')
    test_group_single.add_argument('--visual', default=False, action='store_true',
                                   help='visualize result')
    test_group_single.add_argument('--cache', default=False, action='store_true',
                                   help='cache score map')
    test_group_single.add_argument('--device_target', type=str, nargs='?',
                                   default='GPU', help="device_target: GPU, CPU, or Ascend")
    test_group_single.add_argument('--accuracy', default=False, action='store_true',
                                   help='only calculate accuracy')
    test_group_single.add_argument('--output', type=str, nargs='?', help="path to save prediction result")
    test_group_single.add_argument('--prediction', type=str, nargs='?', help='prediction path for accuracy.'
                                                                             'or use yaml config,'
                                                                             'single:output multi:gt_segm_output')
    test_group_multi = parser_test.add_argument_group(TARGET_COCO_MULTI)
    test_group_multi.add_argument('--dev', default=False, action='store_true',
                                  help='development mode')
    test_group_multi.add_argument('--graph', default=False, action='store_true',
                                  help='eval graph')
    test_group_multi.add_argument('--score-maps-cached', default=False, action='store_true',
                                  help='use cached score map in yaml config cached_scoremaps')
    test_group_multi.add_argument('--range-num', type=int, nargs='?',
                                  help='range number. split dataset to this number')
    test_group_multi.add_argument('--range-index', type=int, nargs='?',
                                  help='range index. start 0. only eval this range index')
    parsers = {
        'preprocess': parser_pre,
        'train': parser_train,
        'eval': parser_test
    }
    return parsers
