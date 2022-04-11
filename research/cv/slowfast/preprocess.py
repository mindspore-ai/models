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
##############preprocess#################
"""
import os
from src.utils import logging
from src.datasets.build import build_dataset
from src.utils.parser import load_config, parse_args
from src.config.defaults import assert_and_infer_cfg

def load():
    """Entrance method."""
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)
    # setup logger
    logger = logging.get_logger(__name__)
    logging.setup_logging()
    logger.info(cfg)
    # build dataset
    dataset = build_dataset(cfg, "test")
    data_path = cfg.TEST.SAVE_BINS_RESULTS_PATH
    # create dirs to store bins.
    slowpath_path = os.path.join(data_path, "slowpath")
    fastpath_path = os.path.join(data_path, "fastpath")
    boxes_path = os.path.join(data_path, "boxes")
    labels_path = os.path.join(data_path, "labels")
    ori_boxes_path = os.path.join(data_path, "ori_boxes")
    metadata_path = os.path.join(data_path, "metadata")
    mask_path = os.path.join(data_path, "mask")
    os.makedirs(slowpath_path)
    os.makedirs(fastpath_path)
    os.makedirs(boxes_path)
    os.makedirs(labels_path)
    os.makedirs(ori_boxes_path)
    os.makedirs(metadata_path)
    os.makedirs(mask_path)
    # start eval
    iterator = dataset.create_tuple_iterator(output_numpy=True)
    for cur_iter, (slowpath, fastpath, boxes, labels, ori_boxes, metadata, mask) in enumerate(iterator):
        file_name = "ava_val_" + str(cur_iter) + ".bin"
        slowpath_file_path = os.path.join(slowpath_path, file_name)
        fastpath_file_path = os.path.join(fastpath_path, file_name)
        boxes_file_path = os.path.join(boxes_path, file_name)
        labels_file_path = os.path.join(labels_path, file_name)
        ori_boxes_file_path = os.path.join(ori_boxes_path, file_name)
        metadata_file_path = os.path.join(metadata_path, file_name)
        mask_file_path = os.path.join(mask_path, file_name)
        # save to file.
        slowpath.tofile(slowpath_file_path)
        fastpath.tofile(fastpath_file_path)
        boxes.tofile(boxes_file_path)
        labels.tofile(labels_file_path)
        ori_boxes.tofile(ori_boxes_file_path)
        metadata.tofile(metadata_file_path)
        mask.tofile(mask_file_path)
        print("=save cur_iter: {}=".format(cur_iter))
    print("Export bin files finished!")

if __name__ == '__main__':
    load()
