# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License Version 2.0(the "License");
# you may not use this file except in compliance with the License.
# you may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0#
#
# Unless required by applicable law or agreed to in writing software
# distributed under the License is distributed on an "AS IS" BASIS
# WITHOUT WARRANT IES OR CONITTONS OF ANY KIND， either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ====================================================================================

"""Define datasets: Market1501, Dukemtmc-reid, CUHK03, MSMT17."""

import re
import glob
import warnings
import os
import os.path as osp
import json
import errno
from PIL import Image
import numpy as np


def read_image(path):
    """Reads image from path using ``PIL.Image``.

    Args:
        path (str): path to an image.

    Returns:
        PIL image
    """
    got_img = False
    if not osp.exists(path):
        raise IOError('"{}" does not exist'.format(path))
    while not got_img:
        try:
            img = Image.open(path).convert('RGB')
            got_img = True
        except IOError:
            print(
                'IOError incurred when reading "{}". Will redo. Don\'t worry. Just chill.'
                .format(path)
            )
    return img


def read_json(fpath):
    """Reads json file from a path."""
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    """Writes to a json file."""
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def mkdir_if_missing(dirname):
    """Creates dirname if it is missing."""
    if not osp.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class Dataset():
    """An abstract class representing a Dataset.

    This is the base class for four datasets.

    Args:
        train (list): contains tuples of (img_path(s), pid, camid).
        query (list): contains tuples of (img_path(s), pid, camid).
        gallery (list): contains tuples of (img_path(s), pid, camid).
        transform: transform function.
        k_tfm (int): number of times to apply augmentation to an image
            independently. If k_tfm > 1, the transform function will be
            applied k_tfm times to an image. This variable will only be
            useful for training and is currently valid for image datasets only.
        mode (str): 'train', 'query' or 'gallery'.
        combineall (bool): combines train, query and gallery in a
            dataset for training.
        verbose (bool): show information.
    """

    # junk_pids contains useless person IDs, e.g. background,
    # false detections, distractors. These IDs will be ignored

    _junk_pids = []

    # Some datasets are only used for training, like CUHK-SYSU
    # In this case, "combineall=True" is not used for them
    _train_only = False

    def __init__(
            self,
            train,
            query,
            gallery,
            mode='train',
            verbose=True,
            **kwargs
    ):
        # extend 3-tuple (img_path(s), pid, camid) to
        # 4-tuple (img_path(s), pid, camid, dsetid) by
        # adding a dataset indicator "dsetid"
        if len(train[0]) == 3:
            train = [(*items, 0) for items in train]
        if len(query[0]) == 3:
            query = [(*items, 0) for items in query]
        if len(gallery[0]) == 3:
            gallery = [(*items, 0) for items in gallery]

        self.train = train
        self.query = query
        self.gallery = gallery
        self.mode = mode
        self.verbose = verbose

        self.num_train_pids = self.get_num_pids(self.train)
        self.num_train_cams = self.get_num_cams(self.train)
        self.num_datasets = self.get_num_datasets(self.train)


        if self.mode == 'train':
            self.data = self.train
        elif self.mode == 'query':
            self.data = self.query
        elif self.mode == 'gallery':
            self.data = self.gallery
        else:
            raise ValueError(
                'Invalid mode. Got {}, but expected to be '
                'one of [train | query | gallery]'.format(self.mode)
            )

        if self.verbose:
            self.show_summary()

    def __getitem__(self, index):
        img_path, pid, camid, _ = self.data[index]
        img = read_image(img_path)
        pid = np.array(pid).astype(np.int32)
        if self.mode == 'train':
            return img, pid

        return img, pid, camid


    def __len__(self):
        return len(self.data)

    def get_num_pids(self, data):
        """Returns the number of training person identities.

        Each tuple in data contains (img_path(s), pid, camid, dsetid).
        """
        pids = set()
        for items in data:
            pid = items[1]
            pids.add(pid)
        return len(pids)

    def get_num_cams(self, data):
        """Returns the number of training cameras.

        Each tuple in data contains (img_path(s), pid, camid, dsetid).
        """
        cams = set()
        for items in data:
            camid = items[2]
            cams.add(camid)
        return len(cams)

    def get_num_datasets(self, data):
        """Returns the number of datasets included.

        Each tuple in data contains (img_path(s), pid, camid, dsetid).
        """
        dsets = set()
        for items in data:
            dsetid = items[3]
            dsets.add(dsetid)
        return len(dsets)

    def show_summary(self):
        '''show summary of a dataset'''
        num_train_pids = self.get_num_pids(self.train)
        num_train_cams = self.get_num_cams(self.train)

        num_query_pids = self.get_num_pids(self.query)
        num_query_cams = self.get_num_cams(self.query)

        num_gallery_pids = self.get_num_pids(self.gallery)
        num_gallery_cams = self.get_num_cams(self.gallery)

        print('=> Loaded {}'.format(self.__class__.__name__))
        print('  ----------------------------------------')
        print('  subset   | # ids | # images | # cameras')
        print('  ----------------------------------------')
        print(
            '  train    | {:5d} | {:8d} | {:9d}'.format(
                num_train_pids, len(self.train), num_train_cams
            )
        )
        print(
            '  query    | {:5d} | {:8d} | {:9d}'.format(
                num_query_pids, len(self.query), num_query_cams
            )
        )
        print(
            '  gallery  | {:5d} | {:8d} | {:9d}'.format(
                num_gallery_pids, len(self.gallery), num_gallery_cams
            )
        )
        print('  ----------------------------------------')

    def check_before_run(self, required_files):
        """Checks if required files exist before going deeper.

        Args:
            required_files (str or list): string file name(s).
        """
        if isinstance(required_files, str):
            required_files = [required_files]

        for fpath in required_files:
            if not osp.exists(fpath):
                raise RuntimeError('"{}" is not found'.format(fpath))

    def __repr__(self):
        num_train_pids = self.get_num_pids(self.train)
        num_train_cams = self.get_num_cams(self.train)

        num_query_pids = self.get_num_pids(self.query)
        num_query_cams = self.get_num_cams(self.query)

        num_gallery_pids = self.get_num_pids(self.gallery)
        num_gallery_cams = self.get_num_cams(self.gallery)

        msg = '  ----------------------------------------\n' \
              '  subset   | # ids | # items | # cameras\n' \
              '  ----------------------------------------\n' \
              '  train    | {:5d} | {:7d} | {:9d}\n' \
              '  query    | {:5d} | {:7d} | {:9d}\n' \
              '  gallery  | {:5d} | {:7d} | {:9d}\n' \
              '  ----------------------------------------\n' \
              '  items: images/tracklets for image/video dataset\n'.format(
                  num_train_pids, len(self.train), num_train_cams,
                  num_query_pids, len(self.query), num_query_cams,
                  num_gallery_pids, len(self.gallery), num_gallery_cams
              )

        return msg


class Market1501(Dataset):
    """Market1501.
    Dataset statistics:
        - identities: 1501 (+1 for background).
        - images: 12936 (train) + 3368 (query) + 15913 (gallery).
    """
    _junk_pids = [0, -1]
    dataset_dir = 'market1501'
    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, 'Market-1501-v15.09.15')
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn(
                'The current data structure is deprecated. Please '
                'put data folders such as "bounding_box_train" under '
                '"Market-1501-v15.09.15".'
            )

        self.train_dir = osp.join(self.data_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'bounding_box_test')

        required_files = [
            self.data_dir, self.train_dir, self.query_dir, self.gallery_dir
        ]
        self.check_before_run(required_files)
        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)

        super(Market1501, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, relabel=False):
        '''get images and labels from directory.'''
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue # junk images are just ignored
            assert 0 <= pid <= 1501 # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1 # index starts from 0
            if relabel:
                pid = pid2label[pid]
            data.append((img_path, pid, camid))

        return data


TRAIN_DIR_KEY = 'train_dir'
TEST_DIR_KEY = 'test_dir'
VERSION_DICT = {
    'MSMT17_V1': {
        TRAIN_DIR_KEY: 'train',
        TEST_DIR_KEY: 'test',
    },
    'MSMT17_V2': {
        TRAIN_DIR_KEY: 'mask_train_v2',
        TEST_DIR_KEY: 'mask_test_v2',
    }
}
class MSMT17(Dataset):
    """MSMT17.

    Dataset statistics:
        - identities: 4101.
        - images: 32621 (train) + 11659 (query) + 82161 (gallery).
        - cameras: 15.
    """
    dataset_dir = 'msmt17'
    dataset_url = None

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        has_main_dir = False
        for main_dir in VERSION_DICT:
            if osp.exists(osp.join(self.dataset_dir, main_dir)):
                train_dir = VERSION_DICT[main_dir][TRAIN_DIR_KEY]
                test_dir = VERSION_DICT[main_dir][TEST_DIR_KEY]
                has_main_dir = True
                break
        assert has_main_dir, 'Dataset folder not found'

        self.train_dir = osp.join(self.dataset_dir, main_dir, train_dir)
        self.test_dir = osp.join(self.dataset_dir, main_dir, test_dir)
        self.list_train_path = osp.join(
            self.dataset_dir, main_dir, 'list_train.txt'
        )
        self.list_val_path = osp.join(
            self.dataset_dir, main_dir, 'list_val.txt'
        )
        self.list_query_path = osp.join(
            self.dataset_dir, main_dir, 'list_query.txt'
        )
        self.list_gallery_path = osp.join(
            self.dataset_dir, main_dir, 'list_gallery.txt'
        )

        required_files = [self.dataset_dir, self.train_dir, self.test_dir]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, self.list_train_path)
        dummy_var = self.process_dir(self.train_dir, self.list_val_path)
        query = self.process_dir(self.test_dir, self.list_query_path)
        gallery = self.process_dir(self.test_dir, self.list_gallery_path)

        super(MSMT17, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, list_path):
        '''get images and labels from directory.'''
        with open(list_path, 'r') as txt:
            lines = txt.readlines()

        data = []

        for _, img_info in enumerate(lines):
            img_path, pid = img_info.split(' ')
            pid = int(pid)  # no need to relabel
            camid = int(img_path.split('_')[2]) - 1  # index starts from 0
            img_path = osp.join(dir_path, img_path)
            data.append((img_path, pid, camid))

        return data


class DukeMTMCreID(Dataset):
    """DukeMTMC-reID.

    Dataset statistics:
        - identities: 1404 (train + query).
        - images:16522 (train) + 2228 (query) + 17661 (gallery).
        - cameras: 8.
    """
    dataset_dir = 'dukemtmc-reid'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.train_dir = osp.join(
            self.dataset_dir, 'DukeMTMC-reID/bounding_box_train'
        )
        self.query_dir = osp.join(self.dataset_dir, 'DukeMTMC-reID/query')
        self.gallery_dir = osp.join(
            self.dataset_dir, 'DukeMTMC-reID/bounding_box_test'
        )

        required_files = [
            self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)

        super(DukeMTMCreID, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, relabel=False):
        '''get images and labels from directory.'''
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            assert 1 <= camid <= 8
            camid -= 1  # index starts from 0
            if relabel:
                pid = pid2label[pid]
            data.append((img_path, pid, camid))

        return data


class CUHK03(Dataset):
    """CUHK03.

    Dataset statistics:
        - identities: 1360.
        - images: 13164.
        - cameras: 6.
        - splits: 20 (classic).
    """
    dataset_dir = 'cuhk03'

    def __init__(
            self,
            root='',
            split_id=0,
            cuhk03_labeled=False,
            cuhk03_classic_split=False,
            **kwargs
    ):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        self.data_dir = osp.join(self.dataset_dir, 'cuhk03_release')
        self.raw_mat_path = osp.join(self.data_dir, 'cuhk-03.mat')

        self.imgs_detected_dir = osp.join(self.dataset_dir, 'images_detected')
        self.imgs_labeled_dir = osp.join(self.dataset_dir, 'images_labeled')

        self.split_classic_det_json_path = osp.join(
            self.dataset_dir, 'splits_classic_detected.json'
        )
        self.split_classic_lab_json_path = osp.join(
            self.dataset_dir, 'splits_classic_labeled.json'
        )

        self.split_new_det_json_path = osp.join(
            self.dataset_dir, 'splits_new_detected.json'
        )
        self.split_new_lab_json_path = osp.join(
            self.dataset_dir, 'splits_new_labeled.json'
        )

        self.split_new_det_mat_path = osp.join(
            self.dataset_dir, 'cuhk03_new_protocol_config_detected.mat'
        )
        self.split_new_lab_mat_path = osp.join(
            self.dataset_dir, 'cuhk03_new_protocol_config_labeled.mat'
        )

        required_files = [
            self.dataset_dir, self.data_dir, self.raw_mat_path,
            self.split_new_det_mat_path, self.split_new_lab_mat_path
        ]
        self.check_before_run(required_files)

        self.preprocess_split()

        if cuhk03_labeled:
            split_path = self.split_classic_lab_json_path if cuhk03_classic_split else self.split_new_lab_json_path
        else:
            split_path = self.split_classic_det_json_path if cuhk03_classic_split else self.split_new_det_json_path

        splits = read_json(split_path)
        assert split_id < len(
            splits
        ), 'Condition split_id ({}) < len(splits) ({}) is false'.format(
            split_id, len(splits)
        )
        split = splits[split_id]

        train = split['train']
        query = split['query']
        gallery = split['gallery']

        super(CUHK03, self).__init__(train, query, gallery, **kwargs)

    def preprocess_split(self):
        '''get images and labels from .mat file.'''
        # This function is a bit complex and ugly, what it does is
        # 1. extract data from cuhk-03.mat and save as png images
        # 2. create 20 classic splits (Li et al. CVPR'14)
        # 3. create new split (Zhong et al. CVPR'17)
        if osp.exists(self.imgs_labeled_dir) \
                and osp.exists(self.imgs_detected_dir) \
                and osp.exists(self.split_classic_det_json_path) \
                and osp.exists(self.split_classic_lab_json_path) \
                and osp.exists(self.split_new_det_json_path) \
                and osp.exists(self.split_new_lab_json_path):
            return

        import h5py
        import imageio
        from scipy.io import loadmat

        mkdir_if_missing(self.imgs_detected_dir)
        mkdir_if_missing(self.imgs_labeled_dir)

        print(
            'Extract image data from "{}" and save as png'.format(
                self.raw_mat_path
            )
        )
        mat = h5py.File(self.raw_mat_path, 'r')

        def _deref(ref):
            return mat[ref][:].T

        def _process_images(img_refs, campid, pid, save_dir):
            img_paths = []  # Note: some persons only have images for one view
            for imgid, img_ref in enumerate(img_refs):
                img = _deref(img_ref)
                if img.size == 0 or img.ndim < 3:
                    continue  # skip empty cell
                # images are saved with the following format, index-1 (ensure uniqueness)
                # campid: index of camera pair (1-5)
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

        def _extract_img(image_type):
            print('Processing {} images ...'.format(image_type))
            meta_data = []
            imgs_dir = self.imgs_detected_dir if image_type == 'detected' else self.imgs_labeled_dir
            for campid, camp_ref in enumerate(mat[image_type][0]):
                camp = _deref(camp_ref)
                num_pids = camp.shape[0]
                for pid in range(num_pids):
                    img_paths = _process_images(
                        camp[pid, :], campid, pid, imgs_dir
                    )
                    assert img_paths, \
                        'campid{}-pid{} has no images'.format(campid, pid)
                    meta_data.append((campid + 1, pid + 1, img_paths))
                print(
                    '- done camera pair {} with {} identities'.format(
                        campid + 1, num_pids
                    )
                )
            return meta_data

        meta_detected = _extract_img('detected')
        meta_labeled = _extract_img('labeled')

        def _extract_classic_split(meta_data, test_split):
            train, test = [], []
            num_train_pids, num_test_pids = 0, 0
            num_train_imgs, num_test_imgs = 0, 0
            for _, (campid, pid, img_paths) in enumerate(meta_data):

                if [campid, pid] in test_split:
                    for img_path in img_paths:
                        camid = int(
                            osp.basename(img_path).split('_')[2]
                        ) - 1  # make it 0-based
                        test.append((img_path, num_test_pids, camid))
                    num_test_pids += 1
                    num_test_imgs += len(img_paths)
                else:
                    for img_path in img_paths:
                        camid = int(
                            osp.basename(img_path).split('_')[2]
                        ) - 1  # make it 0-based
                        train.append((img_path, num_train_pids, camid))
                    num_train_pids += 1
                    num_train_imgs += len(img_paths)
            return train, num_train_pids, num_train_imgs, test, num_test_pids, num_test_imgs

        print('Creating classic splits (# = 20) ...')
        splits_classic_det, splits_classic_lab = [], []
        for split_ref in mat['testsets'][0]:
            test_split = _deref(split_ref).tolist()

            # create split for detected images
            train, num_train_pids, num_train_imgs, test, num_test_pids, num_test_imgs = \
                _extract_classic_split(meta_detected, test_split)
            splits_classic_det.append(
                {
                    'train': train,
                    'query': test,
                    'gallery': test,
                    'num_train_pids': num_train_pids,
                    'num_train_imgs': num_train_imgs,
                    'num_query_pids': num_test_pids,
                    'num_query_imgs': num_test_imgs,
                    'num_gallery_pids': num_test_pids,
                    'num_gallery_imgs': num_test_imgs
                }
            )

            # create split for labeled images
            train, num_train_pids, num_train_imgs, test, num_test_pids, num_test_imgs = \
                _extract_classic_split(meta_labeled, test_split)
            splits_classic_lab.append(
                {
                    'train': train,
                    'query': test,
                    'gallery': test,
                    'num_train_pids': num_train_pids,
                    'num_train_imgs': num_train_imgs,
                    'num_query_pids': num_test_pids,
                    'num_query_imgs': num_test_imgs,
                    'num_gallery_pids': num_test_pids,
                    'num_gallery_imgs': num_test_imgs
                }
            )

        write_json(splits_classic_det, self.split_classic_det_json_path)
        write_json(splits_classic_lab, self.split_classic_lab_json_path)

        def _extract_set(filelist, pids, pid2label, idxs, img_dir, relabel):
            tmp_set = []
            unique_pids = set()
            for idx in idxs:
                img_name = filelist[idx][0]
                camid = int(img_name.split('_')[2]) - 1  # make it 0-based
                pid = pids[idx]
                if relabel:
                    pid = pid2label[pid]
                img_path = osp.join(img_dir, img_name)
                tmp_set.append((img_path, int(pid), camid))
                unique_pids.add(pid)
            return tmp_set, len(unique_pids), len(idxs)

        def _extract_new_split(split_dict, img_dir):
            train_idxs = split_dict['train_idx'].flatten() - 1  # index-0
            pids = split_dict['labels'].flatten()
            train_pids = set(pids[train_idxs])
            pid2label = {pid: label for label, pid in enumerate(train_pids)}
            query_idxs = split_dict['query_idx'].flatten() - 1
            gallery_idxs = split_dict['gallery_idx'].flatten() - 1
            filelist = split_dict['filelist'].flatten()
            train_info = _extract_set(
                filelist, pids, pid2label, train_idxs, img_dir, relabel=True
            )
            query_info = _extract_set(
                filelist, pids, pid2label, query_idxs, img_dir, relabel=False
            )
            gallery_info = _extract_set(
                filelist,
                pids,
                pid2label,
                gallery_idxs,
                img_dir,
                relabel=False
            )
            return train_info, query_info, gallery_info

        print('Creating new split for detected images (767/700) ...')
        train_info, query_info, gallery_info = _extract_new_split(
            loadmat(self.split_new_det_mat_path), self.imgs_detected_dir
        )
        split = [
            {
                'train': train_info[0],
                'query': query_info[0],
                'gallery': gallery_info[0],
                'num_train_pids': train_info[1],
                'num_train_imgs': train_info[2],
                'num_query_pids': query_info[1],
                'num_query_imgs': query_info[2],
                'num_gallery_pids': gallery_info[1],
                'num_gallery_imgs': gallery_info[2]
            }
        ]
        write_json(split, self.split_new_det_json_path)

        print('Creating new split for labeled images (767/700) ...')
        train_info, query_info, gallery_info = _extract_new_split(
            loadmat(self.split_new_lab_mat_path), self.imgs_labeled_dir
        )
        split = [
            {
                'train': train_info[0],
                'query': query_info[0],
                'gallery': gallery_info[0],
                'num_train_pids': train_info[1],
                'num_train_imgs': train_info[2],
                'num_query_pids': query_info[1],
                'num_query_imgs': query_info[2],
                'num_gallery_pids': gallery_info[1],
                'num_gallery_imgs': gallery_info[2]
            }
        ]
        write_json(split, self.split_new_lab_json_path)
