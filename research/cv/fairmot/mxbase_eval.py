# Copyright 2021 Huawei Technologies Co., Ltd
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
"""post process."""
import os
import os.path as osp
import logging
import cv2
import motmetrics as mm
import numpy as np

from src.opts import Opts
from src.tracking_utils import visualization as vis
from src.tracking_utils.log import logger
from src.tracking_utils.utils import mkdir_if_missing
from src.tracking_utils.evaluation import Evaluator
from src.tracking_utils.io import read_results, unzip_objs
import src.utils.jde as datasets



def main(data_root, seqs=('MOT17-01-SDP',), save_dir=None):
    logger.setLevel(logging.INFO)
    data_type = 'mot'
    # run tracking
    accs = []
    timer_avgs, timer_calls = [], []
    for sequence in seqs:
        output_dir = os.path.join(save_dir, sequence)
        mkdir_if_missing(output_dir)
        logger.info('start seq: %s', sequence)
        result_filename = osp.join(data_root, 'result_Files', '{}.txt'.format(sequence))
        logger.info('Evaluate seq: %s', sequence)
        evaluator = Evaluator(osp.join(data_root, 'train'), sequence, data_type)
        accs.append(evaluator.eval_file(result_filename))
        result_frame_dict = read_results(result_filename, 'mot', is_gt=False)
        dataloader = datasets.LoadImages(osp.join(os.path.join(data_root, 'train'), sequence, 'img1'),
                                         (1088, 608))
        for i, (_, _, img0) in enumerate(dataloader):
            frame_id = i+1
            trk_objs = result_frame_dict.get(frame_id, [])
            trk_tlwhs, trk_ids = unzip_objs(trk_objs)[:2]
            online_im = vis.plot_tracking(img0, trk_tlwhs, trk_ids, frame_id=frame_id)
            cv2.imwrite(os.path.join(output_dir, '{:05d}.jpg'.format(frame_id)), online_im)
    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    Evaluator.save_summary(summary, os.path.join(data_root, 'summary.xlsx'))


if __name__ == '__main__':
    opts = Opts().init()
    seqs_str = '''  MOT20-01
                    MOT20-02
                    MOT20-03
                    MOT20-05'''
    seq = [seq.strip() for seq in seqs_str.split()]
    save_path = os.path.join(opts.data_dir, 'result')
    main(data_root=opts.data_dir,
         seqs=seq,
         save_dir=save_path)
