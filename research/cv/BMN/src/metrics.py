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

import shutil
import json
import logging
import os
from multiprocessing import Pool
import numpy as np
import pandas as pd
from tqdm import tqdm
import mindspore.ops as ops
from mindspore.nn import Metric
from src.utils import dump_metric_result
from src.postprocessing import BMN_post_processing
logger = logging.getLogger(__name__)

class BMNMetric(Metric):
    def __init__(self, cfg, subset='train_val'):
        self.tscale = cfg.model.temporal_scale
        self.subset = subset  # 'train', 'validation', 'train_val'
        self.anno_file = cfg.data.video_annotations
        self.postpr_config = cfg
        self.get_dataset_dict()
        self.output_path = cfg.eval.output_path
        self.pbar_update = tqdm(total=len(self.video_list),\
                                postfix="\n")
        self.pbar_update.set_description("Collecting BMN metrics")
        self.clear()
        self.unstack = ops.Unstack(axis=0) # loss unpack
        self.reduce_mean = ops.ReduceMean()
        self.cast = ops.Cast()
        self.sample_counter = 0
        self.results_pairs = []
        self.threads = cfg.eval.threads
        self.env_setup()


    def env_setup(self):
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)
        os.makedirs(self.output_path, exist_ok=True)

    def get_dataset_dict(self):
        annos = json.load(open(self.anno_file))
        self.video_dict = {}
        for video_name in annos.keys():
            video_subset = annos[video_name]["subset"]
            if self.subset == "train_val":
                if "train" in video_subset or "validation" in video_subset:
                    self.video_dict[video_name] = annos[video_name]
            else:
                if self.subset in video_subset:
                    self.video_dict[video_name] = annos[video_name]
        self.video_list = list(self.video_dict.keys())
        self.video_list.sort()

    def clear(self):
        logger.info('Resetting %s metrics...', self.subset)
        self.sample_counter = 0
        self.pbar_update.reset()
        self.results_pairs = []

    def save_results(self, props, fid):
        batch_size = len(props)
        for i in range(batch_size):
            video_name = self.video_list[fid[i]]
            col_name = ["xmin", "xmax", "xmin_score", "xmax_score", "clr_score", "reg_socre", "score"]
            new_df = pd.DataFrame(props[i], columns=col_name)
            self.results_pairs.append((new_df, os.path.join(self.output_path, video_name + ".csv")))

    def update(self, *fetch_list):
        cur_batch_size = fetch_list[0].shape[0]
        start_scores, end_scores, clr_confidence, reg_confidence = fetch_list
        fetch_list = start_scores.asnumpy(),\
                     end_scores.asnumpy(),\
                     clr_confidence.asnumpy(),\
                     reg_confidence.asnumpy()

        batch_new_props = []
        for s_score, e_score, clr_conf, reg_conf in zip(*fetch_list):
            new_props = []
            for idx in range(self.tscale):
                for jdx in range(self.tscale):
                    start_index = idx
                    end_index = jdx + 1
                    if start_index < end_index < self.tscale:
                        xmin = start_index / self.tscale
                        xmax = end_index / self.tscale
                        xmin_score = s_score[start_index]
                        xmax_score = e_score[end_index]
                        clr_score = clr_conf[idx, jdx]
                        reg_score = reg_conf[idx, jdx]
                        score = xmin_score * xmax_score * clr_score * reg_score
                        new_props.append([xmin, xmax, xmin_score, xmax_score, clr_score, reg_score, score])
            new_props = np.stack(new_props)
            batch_new_props.append(new_props)

        fid = np.array([self.sample_counter + x for x in range(cur_batch_size)])
        self.save_results(batch_new_props, fid)

        self.pbar_update.update(cur_batch_size)
        self.sample_counter += cur_batch_size

    def eval(self):
        logger.info("Dumping results...")
        self.dump_results()
        logger.info("start generate proposals of %s subset", (self.subset))
        logger.warning("NOT DONE WAIT FOR THE NEXT LOGGER MESSAGE")
        BMN_post_processing(self.postpr_config)
        logger.info("finish generate proposals of %s subset", (self.subset))

    def dump_results(self):
        with Pool(self.threads) as p:
            p.map(dump_metric_result, self.results_pairs)
