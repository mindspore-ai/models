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
""" CUHK03 dataset processing """

import os
import os.path as osp

import h5py
import imageio
from scipy.io import loadmat


class CUHK03:
    """ CUHK03 dataset manager

    root: dir path
    """
    def __init__(self, root, subset_name):
        self.root = root
        self.images_dir = osp.join(root, "detected_images")
        split_new_det_mat_path = osp.join(root, "cuhk03_new_protocol_config_detected.mat")
        self.split_dict = loadmat(split_new_det_mat_path)

        self.subset_name = subset_name
        self.data = []
        self.num_ids = 0
        self.relabel = subset_name == 'train'

        self.prepare_detected_images()
        self.load()

    @staticmethod
    def _deref(mat, ref):
        """ Extract ref image data from mat """
        return mat[ref][:].T

    def _process_images(self, mat, img_refs, campid, pid, save_dir):
        """ Save mat images as png """
        img_paths = []  # Note: some persons only have images for one view
        for imgid, img_ref in enumerate(img_refs):
            img = self._deref(mat, img_ref)
            if img.size == 0 or img.ndim < 3:
                continue  # skip empty cell
            # images are saved with the following format, index-1 (ensure uniqueness)
            # campid: index of camera pair (1-5) --full name: camera pair id
            # pid: index of person in 'campid'-th camera pair
            # viewid: index of view, {1, 2}
            # imgid: index of image, (1-10)
            viewid = 1 if imgid < 5 else 2
            img_name = '{:01d}_{:03d}_{:01d}_{:02d}.png'.format(
                campid + 1, pid + 1, viewid, imgid + 1
            )
            img_path = osp.join(save_dir, img_name)
            if not osp.isfile(img_path):
                imageio.imwrite(img_path, img)
            img_paths.append(img_path)
        return img_paths

    def prepare_detected_images(self):
        """ Load mat data and save as png images """
        detected_images_dir = osp.join(self.root, "detected_images")
        raw_mat_path = osp.join(self.root, "cuhk-03.mat")
        if osp.exists(detected_images_dir):
            print("detected_images_dir has been prepared!")
            return

        os.makedirs(detected_images_dir)
        print(
            'Extract image data from "{}" and save as png'.format(
                raw_mat_path
            )
        )
        mat = h5py.File(raw_mat_path, 'r')

        print('Processing detected images ...')
        for campid, camp_ref in enumerate(mat["detected"][0]):
            camp = self._deref(mat, camp_ref)
            num_pids = camp.shape[0]
            for pid in range(num_pids):
                img_paths = self._process_images(
                    mat, camp[pid, :], campid, pid, detected_images_dir
                )
                assert not img_paths, \
                    'campid{}-pid{} has no images'.format(campid, pid)
            print(
                '- done camera pair {} with {} identities'.format(
                    campid + 1, num_pids
                )
            )

    def preprocess(self, filelist, pids, idxs, relabel):
        """ Get image names with info and number of unique pid """
        ret = []
        unique_pids = set()
        pid2label = dict()
        if relabel:
            tmp_pids = set(pids[idxs])
            pid2label = {pid: label for label, pid in enumerate(tmp_pids)}
        for fid, idx in enumerate(idxs):
            img_name = filelist[idx][0]
            fpath = osp.join(self.images_dir, img_name)
            camid = int(img_name.split('_')[2]) - 1
            pid = pids[idx]
            if relabel:
                pid = pid2label[pid]
            ret.append((fpath, fid, int(pid), camid))
            unique_pids.add(pid)
        return ret, len(unique_pids)

    def load(self):
        """ Load dataset data """
        pids = self.split_dict['labels'].flatten()
        filelist = self.split_dict['filelist'].flatten()
        idxs = self.split_dict[self.subset_name + '_idx'].flatten() - 1

        self.data, self.num_ids = self.preprocess(filelist, pids, idxs, self.relabel)

        print(self.__class__.__name__, "dataset loaded")
        print(" # subset | # ids | # images")
        print("  ---------------------------")
        print("  {:8} | {:5d} | {:8d}".format(self.subset_name, self.num_ids, len(self.data)))
