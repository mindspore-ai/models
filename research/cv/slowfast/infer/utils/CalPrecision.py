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
import os
from src.utils import logging
from src.utils.ava_eval_helper import evaluate_ava_from_files
from help.config import Config


def parse_args():
    """set and check parameters."""
    parser = argparse.ArgumentParser(description="SLOWFAST for AVA Dataset")
    parser.add_argument("--data_dir", type=str, default="../data/input",
                        help="Dataset contain frames and ava_annotations")
    args_opt = parser.parse_args()
    return args_opt


def main():
    args = parse_args()
    config = Config()
    config.AVA.FRAME_LIST_DIR = args.data_dir + config.AVA.ANN_DIR
    config.AVA.ANNOTATION_DIR = args.data_dir + config.AVA.ANN_DIR
    config.AVA.FRAME_DIR = args.data_dir + config.AVA.FRA_DIR
    # setup logger

    logger = logging.get_logger(__name__)
    logging.setup_logging()
    logger.info(config)
    excluded_keys = os.path.join(
        config.AVA.ANNOTATION_DIR, config.AVA.EXCLUSION_FILE)
    labelmap = os.path.join(config.AVA.ANNOTATION_DIR,
                            config.AVA.LABEL_MAP_FILE)
    gt_filename = os.path.join(
        config.AVA.ANNOTATION_DIR, config.AVA.GROUNDTRUTH_FILE)
    detc_filename = "detections_latest_mxbase.csv"
    evaluate_ava_from_files(
        labelmap, gt_filename, detc_filename, excluded_keys)


if __name__ == "__main__":
    main()
