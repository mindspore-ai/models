# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
"""dataset"""
import os
import math
import random
from src.autoaugment import ImageNetPolicy

from mindspore.mindrecord import FileWriter
import mindspore.dataset as ds
from mindspore.dataset.vision import Inter
import mindspore.common.dtype as mstype
import mindspore.dataset.vision as C
import mindspore.dataset.transforms as C2

class Dataset:
    """Dataset"""
    def __init__(self, root, is_training=True, first_training=True, is_testing=True):
        self.root = os.path.expanduser(root)
        self.train = is_training
        self.first_training = first_training
        self.image_paths = []
        self.image_labels = []
        self.image_cameras = []
        if self.train:
            if self.first_training:
                file = []
                file.append(self.root + 'VeCc/' + 'name_train.txt')  # 0000_x.jpg 136708 true  0-4445 + label
                file.append(self.root + 'VeRi/' + 'name_train.txt')  # 0000_x_x_x.jpg 37746 true  4446-5020 + label
                file.append(self.root + 'VeId/' + 'name_train.txt')  # 0000000 00000 221567 true  5021-31348 + label
                file.append(self.root + 'VeCf/' + 'name_train.txt')  # 0000_x_x.jpg 52717 true  31348-31788 + label
                for idx, txt_file in enumerate(file):
                    with open(txt_file, 'r') as f:
                        for line in f.readlines():
                            if idx != 2:
                                fname, vid = line.split()
                            else:
                                fname, _, vid = line.split()
                            self.image_labels.append(int(vid))
                            if idx == 0:
                                self.image_paths.append(self.root + 'VeCc/' + 'image_train/' + fname)
                            elif idx == 1:
                                self.image_paths.append(self.root + 'VeRi/' + 'image_train/' + fname)
                            elif idx == 2:
                                self.image_paths.append(self.root + 'VeId/' + 'image_train/' + fname)
                            else:
                                self.image_paths.append(self.root + 'VeCf/' + 'image_train/' + fname)
            else:
                txt_file = self.root + 'VeRi/' + 'name_train_second.txt'
                with open(txt_file, 'r') as f:
                    for line in f.readlines():
                        fname, vid = line.split()
                        self.image_paths.append(self.root + 'VeRi/' + 'image_train/' + fname)
                        self.image_labels.append(int(vid))
        else:
            if is_testing:
                txt_file = self.root + 'VeRi/' + 'name_test.txt'
                with open(txt_file, 'r') as f:
                    for line in f.readlines():
                        fname, vid, cid = line.split()
                        self.image_paths.append(self.root + 'VeRi/' + 'image_test/' + fname)
                        self.image_labels.append(int(vid))
                        self.image_cameras.append(int(cid))
            else:
                txt_file = self.root + 'VeRi/' + 'name_query.txt'
                with open(txt_file, 'r') as f:
                    for line in f.readlines():
                        fname, vid, cid = line.split()
                        self.image_paths.append(self.root + 'VeRi/' + 'image_query/' + fname)
                        self.image_labels.append(int(vid))
                        self.image_cameras.append(int(cid))

    def __getdata__(self):
        img_paths = self.image_paths
        img_labels = self.image_labels
        img_cameras = self.image_cameras
        return img_paths, img_labels, img_cameras

    def __len__(self):
        return len(self.image_paths)

def data_to_mindrecord(data_path, is_training, first_training, is_testing, mindrecord_file, file_num=1):
    """data_to_mindrecord"""
    writer = FileWriter(mindrecord_file, file_num)
    data = Dataset(data_path, is_training, first_training, is_testing)
    image_paths, image_labels, image_cameras = data.__getdata__()

    if is_training:
        vehiclenet_json = {
            "image": {"type": "bytes"},
            "label": {"type": "int32"}
        }
        writer.add_schema(vehiclenet_json, "vehiclenet_json")
    else:
        veri_json = {
            "image": {"type": "bytes"},
            "label": {"type": "int32"},
            "camera": {"type": "int32"}
        }
        writer.add_schema(veri_json, "veri_json")

    image_files_num = len(image_paths)
    for ind, image_name in enumerate(image_paths):
        with open(image_name, 'rb') as f:
            image = f.read()
        label = image_labels[ind]

        if is_training:
            row = {"image": image, "label": label}
        else:
            camera = image_cameras[ind]
            row = {"image": image, "label": label, "camera": camera}

        if (ind + 1) % 10000 == 0:
            print("writing {}/{} into mindrecord".format(ind + 1, image_files_num))
        writer.write_raw_data([row])
    writer.commit()

def create_vehiclenet_dataset(mindrecord_file, batch_size=1, device_num=1, is_training=True,
                              use_aug=True, erasing_p=0.5, color_jitter=False, rank_id=0):
    """create_vehiclenet_dataset"""
    if is_training:
        dataset = ds.MindDataset(mindrecord_file, columns_list=["image", "label"],
                                 num_shards=device_num, shard_id=rank_id,
                                 num_parallel_workers=8, shuffle=True)
    else:
        dataset = ds.MindDataset(mindrecord_file, columns_list=["image", "label", "camera"],
                                 num_shards=device_num, shard_id=rank_id,
                                 num_parallel_workers=8, shuffle=True)

    decode = C.Decode()
    dataset = dataset.map(operations=decode, input_columns=["image"])
    transforms_list = []
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    train_inputsize = 384
    test_inputsize = 384

    if is_training:
        if use_aug:
            py_to_pil_op = C.ToPIL()
            autoaugment_op = ImageNetPolicy()
            to_tensor_op = C.ToTensor()
            transforms_list += [py_to_pil_op, autoaugment_op, to_tensor_op]

        resized_op = C.Resize([train_inputsize, train_inputsize], interpolation=Inter.BICUBIC)
        pad_op = C.Pad(15)
        random_resized_crop_op = C.RandomResizedCrop(train_inputsize, scale=(0.75, 1.0),
                                                     ratio=(0.75, 1.3333), interpolation=Inter.BICUBIC)
        random_horizontal_flip_op = C.RandomHorizontalFlip()
        normlize_op = C.Normalize(mean=mean, std=std)
        to_tensor_op = C.HWC2CHW()
        transforms_list = [resized_op, pad_op, random_resized_crop_op,
                           random_horizontal_flip_op, normlize_op, to_tensor_op]

        if erasing_p > 0:
            random_erasing = RandomErasing(probability=erasing_p, mean=[0.0, 0.0, 0.0])
            transforms_list += [random_erasing]

        if color_jitter:
            color_jitter_op = C.RandomColorAdjust(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)
            transforms_list += [color_jitter_op]
    else:
        resized_op = C.Resize([test_inputsize, test_inputsize], interpolation=Inter.BICUBIC)
        normlize_op = C.Normalize(mean=mean, std=std)
        to_tensor_op = C.HWC2CHW()
        transforms_list = [resized_op, normlize_op, to_tensor_op]

    type_cast_op = C2.TypeCast(mstype.int32)

    dataset = dataset.map(operations=transforms_list, input_columns=['image'])
    dataset = dataset.map(operations=type_cast_op, input_columns=['label'])
    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset

#pylint: disable-msg=W0102
class RandomErasing:
    """RandomErasing"""
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img

        for _ in range(100):
            area = img.shape[1] * img.shape[2]
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.shape[2] and h < img.shape[1]:
                x1 = random.randint(0, img.shape[1] - h)
                y1 = random.randint(0, img.shape[2] - w)
                if img.shape[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img
