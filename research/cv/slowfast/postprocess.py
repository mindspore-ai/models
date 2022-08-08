# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License")
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
##############postprocess#################
"""
import os
import numpy as np

from src.utils.meters import AVAMeter
from src.config.defaults import assert_and_infer_cfg
from src.utils.parser import load_config, parse_args
from src.utils import logging

def run_eval():
    """Entrance method."""
    # general
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)
    # setup logger
    logger = logging.get_logger(__name__)
    logging.setup_logging()
    logger.info(cfg)
    src_bin_dir = args.src_bin_dir
    result_dir = args.result_dir
    logger.info("=================src_bin_dir=%s=================", src_bin_dir)
    # Get bin dir for every params.
    size = len(os.listdir(result_dir))
    test_meter = AVAMeter(size, cfg, mode="test")
    test_meter.iter_tic()
    mask_dir = os.path.join(src_bin_dir, "mask")
    metadata_dir = os.path.join(src_bin_dir, "metadata")
    ori_boxes_dir = os.path.join(src_bin_dir, "ori_boxes")
    for i in range(size):
        test_meter.data_toc()
        label = "ava_val_{}.bin".format(i)
        logger.info("=================label=%s=================", label)
        # get predict bin.
        result_file = os.path.join(result_dir, "ava_val_{}_0.bin".format(i))
        preds = np.fromfile(result_file, dtype=np.float32).reshape(224, 80)
        test_meter.iter_toc()
        # process output.
        # mask's file name same as label's file name.
        mask = np.fromfile(os.path.join(mask_dir, label), dtype=np.int32).reshape(cfg.TEST.BATCH_SIZE, 28)
        logger.info("=================preds.shape=%s, mask.shape=%s===============", preds.shape, mask.shape)
        preds = preds.reshape(mask.shape + (cfg.MODEL.NUM_CLASSES,))
        mask = mask.astype(bool)
        preds = preds[mask]
        padded_idx = np.tile(np.arange(mask.shape[0]).reshape((-1, 1, 1)), (1, mask.shape[1], 1))
        ori_boxes_file = os.path.join(ori_boxes_dir, label)
        ori_boxes = np.fromfile(ori_boxes_file, dtype=np.float32).reshape(cfg.TEST.BATCH_SIZE, 28, 4)
        ori_boxes = np.concatenate((padded_idx, ori_boxes), axis=2)[mask]
        metadata = np.fromfile(os.path.join(metadata_dir, label), dtype=np.int32).reshape(cfg.TEST.BATCH_SIZE, 28, 2)
        metadata = metadata[mask]
        # update stats
        test_meter.update_stats(preds, ori_boxes, metadata)
        test_meter.log_iter_stats(None, i)
        test_meter.iter_tic()
    test_meter.finalize_metrics()

if __name__ == "__main__":
    run_eval()
