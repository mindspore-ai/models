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
"""
convert the train ckpt to test ckpt
"""

import argparse
from src.model_utils.util import train2test


def main():
    """
    Returns: test ckpt
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, help='Path to option YMAL file.')
    parser.add_argument('--test_path', type=str, help='Path to option YMAL file.')
    args = parser.parse_args()
    train2test(args.ckpt_path, args.test_path)


if __name__ == '__main__':
    main()
