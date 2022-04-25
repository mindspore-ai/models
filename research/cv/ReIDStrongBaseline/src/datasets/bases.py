# encoding: utf-8
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
""" Base dataset loader """

import numpy as np


class BaseDataset:
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        """ Get number of persons, images and cams from image data """
        pids, cams = [], []
        for _, pid, camid in data:
            pids += [pid]
            cams += [camid]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams

    def get_videodata_info(self, data, return_tracklet_stats=False):
        """ Get number of persons, images and cams [and tracklet_stats if flag] from video data """
        pids, cams, tracklet_stats = [], [], []
        for img_paths, pid, camid in data:
            pids += [pid]
            cams += [camid]
            tracklet_stats += [len(img_paths)]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_tracklets = len(data)
        if return_tracklet_stats:
            return num_pids, num_tracklets, num_cams, tracklet_stats
        return num_pids, num_tracklets, num_cams

    def print_dataset_statistics(self, **kwargs):
        """ print dataset statistic in """
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        """ Print train, query and gallery subset statistics in console """
        num_train_pids, num_train_imgs, num_train_cams = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")


class BaseVideoDataset(BaseDataset):
    """
    Base class of video reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        """ Print train, query and gallery subset statistics in console """
        num_train_pids, num_train_tracklets, num_train_cams, train_tracklet_stats = \
            self.get_videodata_info(train, return_tracklet_stats=True)

        num_query_pids, num_query_tracklets, num_query_cams, query_tracklet_stats = \
            self.get_videodata_info(query, return_tracklet_stats=True)

        num_gallery_pids, num_gallery_tracklets, num_gallery_cams, gallery_tracklet_stats = \
            self.get_videodata_info(gallery, return_tracklet_stats=True)

        tracklet_stats = train_tracklet_stats + query_tracklet_stats + gallery_tracklet_stats
        min_num = np.min(tracklet_stats)
        max_num = np.max(tracklet_stats)
        avg_num = np.mean(tracklet_stats)

        print("Dataset statistics:")
        print("  -------------------------------------------")
        print("  subset   | # ids | # tracklets | # cameras")
        print("  -------------------------------------------")
        print("  train    | {:5d} | {:11d} | {:9d}".format(num_train_pids, num_train_tracklets, num_train_cams))
        print("  query    | {:5d} | {:11d} | {:9d}".format(num_query_pids, num_query_tracklets, num_query_cams))
        print("  gallery  | {:5d} | {:11d} | {:9d}".format(num_gallery_pids, num_gallery_tracklets, num_gallery_cams))
        print("  -------------------------------------------")
        print("  number of images per tracklet: {} ~ {}, average {:.2f}".format(min_num, max_num, avg_num))
        print("  -------------------------------------------")
