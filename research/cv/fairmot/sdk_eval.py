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
from src.config import Opts
from src.tracking_utils import visualization as vis
from src.tracker.multitracker_sdk import JDETracker
from src.tracking_utils.log import logger
from src.tracking_utils.utils import mkdir_if_missing
from src.tracking_utils.evaluation import Evaluator
from src.tracking_utils.timer import Timer
import src.utils.jde as datasets
import cv2
import motmetrics as mm
import numpy as np


def get_eval_result(img_path, result_path):
    """read bin file"""
    tempfilename = os.path.split(img_path)[1]
    filename, _ = os.path.splitext(tempfilename)
    id_feature_result_file = os.path.join(result_path, filename + "_0.bin")
    dets_result_file = os.path.join(result_path, filename + "_1.bin")
    id_feature = np.fromfile(id_feature_result_file, dtype=np.float32).reshape(500, 128)
    dets = np.fromfile(dets_result_file, dtype=np.float32).reshape(1, 500, 6)
    return [id_feature, dets]


def write_results(filename, results, data_type):
    """write eval results."""
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
    logger.info('save results to %s', filename)


def eval_seq(opt, dataloader, data_type, result_filename, save_dir=None, show_image=True, frame_rate=30,
             result_path=None):
    """evaluation sequence."""
    if save_dir:
        mkdir_if_missing(save_dir)
    tracker = JDETracker(opt, frame_rate=frame_rate)
    timer = Timer()
    results = []
    frame_id = 0
    for path, img, img0 in dataloader:
        result = get_eval_result(path, result_path)
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        # run tracking
        timer.tic()
        blob = np.expand_dims(img, 0)
        height, width = img0.shape[0], img0.shape[1]
        inp_height, inp_width = [blob.shape[2], blob.shape[3]]
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
        meta = {'c': c, 's': s, 'out_height': inp_height // opt.down_ratio,
                'out_width': inp_width // opt.down_ratio}
        online_targets = tracker.update(result[0], result[1], meta)
        online_tlwhs = []
        online_ids = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
        timer.toc()
        results.append((frame_id + 1, online_tlwhs, online_ids))
        if show_image or save_dir is not None:
            online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                          fps=1. / timer.average_time)
        if show_image:
            cv2.imshow('online_im', online_im)
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
        frame_id += 1
    write_results(result_filename, results, data_type)
    return frame_id, timer.average_time, timer.calls


def main(opt, data_root, result_path, seqs=('MOT17-01-SDP',), save_images=True, save_videos=False, show_image=False):
    logger.setLevel(logging.INFO)
    result_root = os.path.join(data_root, '..', 'results')
    mkdir_if_missing(result_root)
    data_type = 'mot'
    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    for sequence in seqs:
        output_dir = os.path.join(data_root, '..', 'outputs', sequence) \
            if save_images or save_videos else None
        logger.info('start seq: %s', sequence)
        dataloader = datasets.LoadImages(osp.join(data_root, sequence, 'img1'), (1088, 608))
        result_filename = osp.join(result_root, '{}.txt'.format(sequence))
        meta_info = open(osp.join(data_root, sequence, 'seqinfo.ini')).read()
        frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
        nf, ta, tc = eval_seq(opt, dataloader, data_type, result_filename,
                              save_dir=output_dir, show_image=show_image, frame_rate=frame_rate,
                              result_path=osp.join(result_path, sequence))
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)
        logger.info('Evaluate seq: %s', sequence)
        evaluator = Evaluator(data_root, sequence, data_type)
        accs.append(evaluator.eval_file(result_filename))
        if save_videos:
            print(output_dir)
            output_video_path = osp.join(output_dir, '{}.mp4'.format(sequence))
            cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(output_dir, output_video_path)
            os.system(cmd_str)
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
    Evaluator.save_summary(summary, os.path.join(result_root, 'summary.xlsx'))


if __name__ == '__main__':
    opts = Opts().get_config()
    seqs_str = '''  MOT20-01
                    MOT20-02
                    MOT20-03
                    MOT20-05'''

    data_roots = os.path.join(opts.data_dir, 'train')
    seq = [seq.strip() for seq in seqs_str.split()]
    result_ = os.path.join(opts.data_dir, '../infer_result')
    main(opts,
         data_root=data_roots,
         result_path=result_,
         seqs=seq,
         show_image=False,
         save_images=True,
         save_videos=False)
