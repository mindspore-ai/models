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
"""data_manager"""
import csv
import os.path as osp

class DatasetManager():
    """DatasetManager"""
    def __init__(self, dataset_dir, root='data', verbose=True):
        super(DatasetManager, self).__init__()
        self.dataset_dir = dataset_dir
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_img_dir = osp.join(self.dataset_dir, 'image_train')
        self.query_img_dir = osp.join(self.dataset_dir, 'image_query')
        self.gallery_img_dir = osp.join(self.dataset_dir, 'image_test')
        self.train_heatmap_dir = osp.join(self.dataset_dir, 'heatmap_train')
        self.query_heatmap_dir = osp.join(self.dataset_dir, 'heatmap_query')
        self.gallery_heatmap_dir = osp.join(self.dataset_dir, 'heatmap_test')
        self.train_segment_dir = osp.join(self.dataset_dir, 'segment_train')
        self.query_segment_dir = osp.join(self.dataset_dir, 'segment_query')
        self.gallery_segment_dir = osp.join(self.dataset_dir, 'segment_test')
        self.train_label = osp.join(self.dataset_dir, 'label_train.csv')
        self.query_label = osp.join(self.dataset_dir, 'label_query.csv')
        self.gallery_label = osp.join(self.dataset_dir, 'label_test.csv')
        # self._check_before_run()

        train, num_train_vids, num_train_vcolors,\
            num_train_vtypes, num_train_imgs, vcolor2label_train, \
                vtype2label_train = self._process_dir(self.train_img_dir, self.train_heatmap_dir, \
                    self.train_segment_dir, self.train_label, relabel=True)
        query, num_query_vids, num_query_vcolors, \
            num_query_vtypes, num_query_imgs, _, _ = self._process_dir(self.query_img_dir, \
                self.query_heatmap_dir, self.query_segment_dir, self.query_label, relabel=False)
        gallery, num_gallery_vids, num_gallery_vcolors, \
            num_gallery_vtypes, num_gallery_imgs, _, _ = self._process_dir(self.gallery_img_dir, \
                self.gallery_heatmap_dir, self.gallery_segment_dir, self.gallery_label, relabel=False)

        num_total_vids = num_train_vids + num_query_vids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        if verbose:
            print("=> Dataset loaded")
            print("Dataset statistics:")
            print("  ------------------------------")
            print("  subset   | # ids | # images")
            print("  ------------------------------")
            print("  train    | {:5d} | {:8d}".format(num_train_vids, num_train_imgs))
            print("  query    | {:5d} | {:8d}".format(num_query_vids, num_query_imgs))
            print("  gallery  | {:5d} | {:8d}".format(num_gallery_vids, num_gallery_imgs))
            print("  ------------------------------")
            print("  total    | {:5d} | {:8d}".format(num_total_vids, num_total_imgs))
            print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_vids = num_train_vids
        self.num_query_vids = num_query_vids
        self.num_gallery_vids = num_gallery_vids

        self.num_train_vcolors = num_train_vcolors
        self.num_query_vcolors = num_query_vcolors
        self.num_gallery_vcolors = num_gallery_vcolors

        self.num_train_vtypes = num_train_vtypes
        self.num_query_vtypes = num_query_vtypes
        self.num_gallery_vtypes = num_gallery_vtypes

        self.vcolor2label = vcolor2label_train
        self.vtype2label = vtype2label_train

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_img_dir):
            raise RuntimeError("'{}' is not available".format(self.train_img_dir))
        if not osp.exists(self.query_img_dir):
            raise RuntimeError("'{}' is not available".format(self.query_img_dir))
        if not osp.exists(self.gallery_img_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_img_dir))
        if not osp.exists(self.train_heatmap_dir):
            raise RuntimeError("'{}' is not available".format(self.train_heatmap_dir))
        if not osp.exists(self.query_heatmap_dir):
            raise RuntimeError("'{}' is not available".format(self.query_heatmap_dir))
        if not osp.exists(self.gallery_heatmap_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_heatmap_dir))
        if not osp.exists(self.train_segment_dir):
            raise RuntimeError("'{}' is not available".format(self.train_segment_dir))
        if not osp.exists(self.query_segment_dir):
            raise RuntimeError("'{}' is not available".format(self.query_segment_dir))
        if not osp.exists(self.gallery_segment_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_segment_dir))
        if not osp.exists(self.train_label):
            raise RuntimeError("'{}' is not available".format(self.train_label))
        if not osp.exists(self.query_label):
            raise RuntimeError("'{}' is not available".format(self.query_label))
        if not osp.exists(self.gallery_label):
            raise RuntimeError("'{}' is not available".format(self.gallery_label))

    def _process_dir(self, dir_img_path, dir_heatmap_path, dir_segment_path, label_path, relabel=False):
        """process_dir"""
        dataset = []
        vid_container = set()
        vcolor_container = set()
        vtype_container = set()
        vcolor2label = {}
        vtype2label = {}
        with open(label_path) as label_file:
            reader = csv.reader(label_file, delimiter=',')
            for row in reader:
                vid = int(row[1])
                vid_container.add(vid)
                vcolor = int(row[2])
                vcolor_container.add(vcolor)
                vtype = int(row[3])
                vtype_container.add(vtype)
                vkeypt = []
                for k in range(36):
                    vkeypt.extend([float(row[4+3*k]), float(row[5+3*k]), float(row[6+3*k])])
                # synthetic data do not have camera ID
                camid = -1
                camidx = row[0].find('c')
                if camidx >= 0:
                    camid = int(row[0][camidx+1:camidx+4])
                dataset.append([osp.join(dir_img_path, row[0]), vid, camid, vcolor, vtype, vkeypt,
                                osp.join(dir_heatmap_path, row[0][:-4]),
                                osp.join(dir_segment_path, row[0][:-4])])

        if relabel:
            vid2label = {vid: label for label, vid in enumerate(vid_container)}
            vcolor2label = {vcolor: label for label, vcolor in enumerate(vcolor_container)}
            vtype2label = {vtype: label for label, vtype in enumerate(vtype_container)}
            for v in dataset:
                v[1] = vid2label[v[1]]
                v[3] = vcolor2label[v[3]]
                v[4] = vtype2label[v[4]]

        num_vids = len(vid_container)
        num_vcolors = len(vcolor_container)
        num_vtypes = len(vtype_container)
        num_imgs = len(dataset)

        return dataset, num_vids, num_vcolors, num_vtypes, num_imgs, vcolor2label, vtype2label
