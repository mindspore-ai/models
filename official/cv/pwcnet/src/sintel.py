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

import os
from glob import glob
import numpy as np

import mindspore.dataset as de
import mindspore
import mindspore.dataset.vision as V
import mindspore.dataset.transforms as T

import src.common as common
import src.transforms as transforms

VALIDATE_INDICES = [
    199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210,
    211, 212, 213, 214, 215, 216, 217, 340, 341, 342, 343, 344,
    345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356,
    357, 358, 359, 360, 361, 362, 363, 364, 536, 537, 538, 539,
    540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551,
    552, 553, 554, 555, 556, 557, 558, 559, 560, 659, 660, 661,
    662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673,
    674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685,
    686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697,
    967, 968, 969, 970, 971, 972, 973, 974, 975, 976, 977, 978,
    979, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989, 990,
    991]



class Sintel():
    '''Sintel dataset.'''
    def __init__(self,
                 dir_root,
                 augmentations=True,
                 imgtype="final",
                 dstype="train"):

        images_root = os.path.join(dir_root, imgtype)
        flow_root = os.path.join(dir_root, "flow")
        occ_root = os.path.join(dir_root, "occlusions")
        self.dstype = dstype


        all_flo_filenames = sorted(glob(os.path.join(flow_root, "*/*.flo")))
        all_img_filenames = sorted(glob(os.path.join(images_root, "*/*.png")))
        all_occ_filenames = sorted(glob(os.path.join(occ_root, "*/*.png")))

        # Remember base for subtraction at runtime
        self._substract_base = common.cd_dotdot(images_root)
        # Get unique basenames
        substract_full_base = common.cd_dotdot(all_img_filenames[0])
        base_folders = [os.path.dirname(fn.replace(substract_full_base, ""))[1:] for fn in all_img_filenames]
        base_folders = sorted(list(set(base_folders)))

        self._image_list = []
        self._flow_list = []
        self._occ_list = []

        for base_folder in base_folders:
            img_filenames = [x for x in all_img_filenames if base_folder in x]
            flo_filenames = [x for x in all_flo_filenames if base_folder in x]
            occ_filenames = [x for x in all_occ_filenames if base_folder in x]

            for i in range(len(img_filenames) - 1):
                im1 = img_filenames[i]
                im2 = img_filenames[i + 1]
                flo = flo_filenames[i]
                occ = occ_filenames[i]

                self._image_list += [[im1, im2]]
                self._flow_list += [flo]
                self._occ_list += [occ]

                # Sanity check
                im1_base_filename = os.path.splitext(os.path.basename(im1))[0]
                im2_base_filename = os.path.splitext(os.path.basename(im2))[0]
                flo_base_filename = os.path.splitext(os.path.basename(flo))[0]
                occ_base_filename = os.path.splitext(os.path.basename(occ))[0]
                im1_frame, im1_no = im1_base_filename.split("_")
                im2_frame, im2_no = im2_base_filename.split("_")
                assert im1_frame == im2_frame
                assert int(im1_no) == (int(im2_no) - 1)

                flo_frame, flo_no = flo_base_filename.split("_")
                assert im1_frame == flo_frame
                assert int(im1_no) == int(flo_no)

                occ_frame, occ_no = occ_base_filename.split("_")
                assert im1_frame == occ_frame
                assert int(im1_no) == int(occ_no)

        assert len(self._image_list) == len(self._flow_list)
        assert len(self._image_list) == len(self._occ_list)

        # Remove invalid validation indices
        full_num_examples = len(self._image_list)
        validate_indices = [x for x in VALIDATE_INDICES if x in range(full_num_examples)]
        # Construct list of indices for training/validation
        list_of_indices = None
        if dstype == "train":
            list_of_indices = [x for x in range(full_num_examples) if x not in validate_indices]
        elif dstype == "valid":
            list_of_indices = validate_indices
        elif dstype == "full":
            list_of_indices = range(full_num_examples)
        else:
            raise ValueError("dstype '%s' unknown!" % dstype)
        # Save list of actual filenames for inputs and flows
        self._image_list = [self._image_list[i] for i in list_of_indices]
        self._flow_list = [self._flow_list[i] for i in list_of_indices]
        self._occ_list = [self._occ_list[i] for i in list_of_indices]

        assert len(self._image_list) == len(self._flow_list)
        assert len(self._image_list) == len(self._occ_list)
        # photometric_augmentations
        if augmentations:
            self._photometric_transform = transforms.ConcatTransformSplitChainer([
                V.ToPIL(),
                V.RandomColorAdjust(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                V.ToTensor(),
                transforms.RandomGamma(min_gamma=0.7, max_gamma=1.5, clip_image=True)
                ])

        else:
            self._photometric_transform = transforms.ConcatTransformSplitChainer([
                V.ToPIL(),
                V.ToTensor(),
                ])

        self._size = len(self._image_list)

    def __getitem__(self, index):
        index = index % self._size

        im1_filename = self._image_list[index][0]
        im2_filename = self._image_list[index][1]
        flo_filename = self._flow_list[index]

        # read float32 images and flow
        im1_np0 = common.read_image_as_byte(im1_filename)
        im2_np0 = common.read_image_as_byte(im2_filename)
        flo_np0 = common.read_flo_as_float32(flo_filename)

        if self.dstype == "train":
            y0 = np.random.randint(0, im1_np0.shape[0] - 384)
            x0 = np.random.randint(0, im1_np0.shape[1] - 512)
            im1_np0 = im1_np0[y0 : y0 + 384, x0 :x0 + 512]
            im2_np0 = im2_np0[y0 : y0 + 384, x0 :x0 + 512]
            flo_np0 = flo_np0[y0 : y0 + 384, x0 :x0 + 512]


        im1, im2 = self._photometric_transform(im1_np0, im2_np0)
        flo = common.numpy2tensor(flo_np0)

        return im1, im2, flo

    def __len__(self):
        return self._size


def SintelTraining(dir_root, augmentations, imgtype, dstype, batchsize, num_parallel_workers, local_rank, world_size):
    '''SintelTraining dataset'''
    dir_root = os.path.join(dir_root, "training")
    dataset = Sintel(dir_root, augmentations, imgtype, dstype)
    dataset_len = len(dataset)
    num_parallel_workers = num_parallel_workers
    de_dataset = de.GeneratorDataset(dataset, ["im1", "im2", "flo"], num_parallel_workers=num_parallel_workers,
                                     shuffle=True, num_shards=world_size, shard_id=local_rank)

    # apply map operations on images
    de_dataset = de_dataset.map(input_columns="im1", operations=T.TypeCast(mindspore.float32))
    de_dataset = de_dataset.map(input_columns="im2", operations=T.TypeCast(mindspore.float32))
    de_dataset = de_dataset.map(input_columns="flo", operations=T.TypeCast(mindspore.float32))
    de_dataset = de_dataset.batch(batchsize, drop_remainder=True)
    return de_dataset, dataset_len
