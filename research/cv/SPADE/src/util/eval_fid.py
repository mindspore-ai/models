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
FID calculation.
"""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import os
import pathlib
from src.models.inception import InceptionV3
from PIL import Image
import numpy as np
from tqdm import tqdm
from scipy import linalg
from mindspore import Tensor
import mindspore.dataset as ds
import mindspore.dataset.vision.py_transforms as py_vision
from mindspore.train.serialization import load_checkpoint, load_param_into_net

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=50,
                    help='Batch size to use')
parser.add_argument('--id', type=int, default=0,
                    help='Device to use.')
parser.add_argument('--path', type=list, nargs=2, default=['/run/czp/FID/img', '/run/czp/FID/seg_img'],
                    help=('Paths to the generated images or '
                          'to .npz statistic files'))
parser.add_argument('--dims', type=int, default=2048,
                    # choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                    help=('Dimensionality of Inception features to use. '
                          'By default, uses pool3 features'))
parser.add_argument('--ckpt_dir', type=str, default='/run/czp/FID/inception_pid.ckpt', help='saves results1 here.')

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}


class ImagePathDataset:
    """img dataload."""
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return (img,)


def get_activations(files, model, batch_size=50, dims=2048):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)
    dataset = ImagePathDataset(files, transforms=py_vision.ToTensor())
    dataloader = ds.GeneratorDataset(dataset, ['image'], shuffle=False)

    dataloader = dataloader.batch(batch_size, drop_remainder=False)
    pred_arr = np.empty((len(files), dims))
    start_idx = 0
    with tqdm(total=dataloader.get_dataset_size()) as p_bar:
        for batch in dataloader:
            batch = Tensor(batch[0])
            pred = model(batch).asnumpy()
            pred_arr[start_idx:start_idx + pred.shape[0]] = pred

            start_idx = start_idx + pred.shape[0]
            p_bar.update(1)
    return pred_arr


def calculate_activation_statistics(files, model, batch_size=50, dims=2048):
    """Calculation of the statistics used by the FID."""
    act = get_activations(files, model, batch_size, dims)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def compute_statistics_of_path(path, model, batch_size, dims):
    """Calculation statistics of path."""
    if path.endswith('.npz'):
        with np.load(path) as f:
            m, s = f['mu'][:], f['sigma'][:]
    else:
        path = pathlib.Path(path)
        files = sorted([file for ext in IMAGE_EXTENSIONS
                        for file in path.glob('*.{}'.format(ext))])
        m, s = calculate_activation_statistics(files, model, batch_size,
                                               dims)
    return m, s


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance."""

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def calculate_fid_given_paths(paths, batch_size, dims, ckpt_dir):
    """calculate fid for given paths"""
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)
    model = InceptionV3()
    param_dict = load_checkpoint(ckpt_dir)
    load_param_into_net(model, param_dict)
    model.set_train(False)
    m1, s1 = compute_statistics_of_path(paths[0], model, batch_size,
                                        dims)
    m2, s2 = compute_statistics_of_path(paths[1], model, batch_size,
                                        dims)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value
