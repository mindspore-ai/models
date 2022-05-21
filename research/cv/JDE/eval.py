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
"""Tracker evaluation script."""
import logging
import os
import os.path as osp

import cv2
import motmetrics as mm
import numpy as np
from mindspore import Model
from mindspore import Tensor
from mindspore import context
from mindspore import dtype as mstype
from mindspore.train.serialization import load_checkpoint

from cfg.config import config as default_config
from src import visualization as vis
from src.darknet import DarkNet, ResidualBlock
from src.dataset import LoadImages
from src.evaluation import Evaluator
from src.log import logger
from src.model import JDEeval
from src.model import YOLOv3
from src.timer import Timer
from src.utils import mkdir_if_missing
from tracker.multitracker import JDETracker

_MOT16_VALIDATION_FOLDERS = (
    'MOT16-02',
    'MOT16-04',
    'MOT16-05',
    'MOT16-09',
    'MOT16-10',
    'MOT16-11',
    'MOT16-13',
)

_MOT16_DIR_FOR_TEST = 'MOT16/train'


def write_results(filename, results, data_type):
    """
    Format for evaluation results.
    """
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('Save results to %s', filename)


def eval_seq(
        opt,
        dataloader,
        data_type,
        result_filename,
        net,
        save_dir=None,
        frame_rate=30,
):
    """
    Processes the video sequence given and provides the output
    of tracking result (write the results in video file).

    It uses JDE model for getting information about the online targets present.

    Args:
        opt (Any): Contains information passed as commandline arguments.
        dataloader (Any): Fetching the image sequence and associated data.
        data_type (str): Type of dataset corresponding(similar) to the given video.
        result_filename (str): The name(path) of the file for storing results.
        net (nn.Cell): Model.
        save_dir (str): Path to output results.
        frame_rate (int): Frame-rate of the given video.

    Returns:
        frame_id (int): Sequence number of the last sequence.
        average_time (int): Average time for frame.
        calls (int): Num of timer calls.
    """
    if save_dir:
        mkdir_if_missing(save_dir)
    tracker = JDETracker(opt, net=net, frame_rate=frame_rate)
    timer = Timer()
    results = []
    frame_id = 0
    timer.tic()
    timer.toc()
    timer.calls -= 1

    for img, img0 in dataloader:
        if frame_id % 20 == 0:
            log_info = f'Processing frame {frame_id} ({(1. / max(1e-5, timer.average_time)):.2f} fps)'
            logger.info('%s', log_info)

        # except initialization step at time calculation
        if frame_id != 0:
            timer.tic()

        im_blob = Tensor(np.expand_dims(img, 0), mstype.float32)
        online_targets = tracker.update(im_blob, img0)
        online_tlwhs = []
        online_ids = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)

        if frame_id != 0:
            timer.toc()
        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))
        if save_dir is not None:
            online_im = vis.plot_tracking(
                img0,
                online_tlwhs,
                online_ids,
                frame_id=frame_id,
                fps=1. / timer.average_time,
            )

            cv2.imwrite(os.path.join(save_dir, f'{frame_id:05}.jpg'), online_im)
        frame_id += 1
    # save results
    write_results(result_filename, results, data_type)

    return frame_id, timer.average_time, timer.calls - 1


def main(
        opt,
        data_root,
        seqs,
        exp_name,
        save_videos=False,
):
    logger.setLevel(logging.INFO)
    result_root = os.path.join(data_root, '..', 'results', exp_name)
    mkdir_if_missing(result_root)
    data_type = 'mot'

    darknet53 = DarkNet(
        ResidualBlock,
        opt.backbone_layers,
        opt.backbone_input_shape,
        opt.backbone_shape,
        detect=True,
    )
    model = YOLOv3(
        backbone=darknet53,
        backbone_shape=opt.backbone_shape,
        out_channel=opt.out_channel,
    )

    model = JDEeval(model, opt)
    load_checkpoint(opt.ckpt_url, model)
    model = Model(model)

    # Run tracking
    n_frame = 0
    timer_avgs, timer_calls, accs = [], [], []

    for seq in seqs:
        output_dir = os.path.join(data_root, '..', 'outputs', exp_name, seq) if save_videos else None

        logger.info('start seq: %s', seq)

        dataloader = LoadImages(osp.join(data_root, seq, 'img1'), opt.anchor_scales, opt.img_size)

        result_filename = os.path.join(result_root, f'{seq}.txt')

        with open(os.path.join(data_root, seq, 'seqinfo.ini')) as f:
            meta_info = f.read()

        frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])

        nf, ta, tc = eval_seq(
            opt,
            dataloader,
            data_type,
            result_filename,
            net=model,
            save_dir=output_dir,
            frame_rate=frame_rate,
        )

        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

        # eval
        logger.info('Evaluate seq: %s', seq)
        evaluator = Evaluator(data_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))
        if save_videos:
            output_video_path = osp.join(output_dir, f'{seq}.mp4')
            cmd_str = f'ffmpeg -f image2 -i {output_dir}/%05d.jpg -c:v copy {output_video_path}'
            os.system(cmd_str)

    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)

    log_info = f'Time elapsed: {all_time:.2f} seconds, FPS: {(1.0 / avg_time):.2f}'
    logger.info('%s', log_info)

    # Get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )

    print(strsummary)
    Evaluator.save_summary(summary, os.path.join(result_root, f'summary_{exp_name}.xlsx'))


if __name__ == '__main__':
    config = default_config

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    context.set_context(device_id=config.device_id)

    data_root_path = os.path.join(config.dataset_root, _MOT16_DIR_FOR_TEST)

    if not os.path.isdir(data_root_path):
        raise NotADirectoryError(
            f'Cannot find "{_MOT16_DIR_FOR_TEST}" subdirectory '
            f'in the specified dataset root "{config.dataset_root}"'
        )

    main(
        config,
        data_root=data_root_path,
        seqs=_MOT16_VALIDATION_FOLDERS,
        exp_name=config.ckpt_url.split('/')[-2],
        save_videos=config.save_videos,
    )
