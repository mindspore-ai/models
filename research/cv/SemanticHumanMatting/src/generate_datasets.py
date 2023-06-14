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

"""Generate datasets"""
import os
import random
import argparse
import shutil
from distutils.dir_util import copy_tree
import yaml

import cv2
import numpy as np


class GenerateData:
    """
    Generate train, eval, test datasets
    """

    def __init__(self, cfg):
        self.cfg = cfg

        self.list_path_clip = list()  # absolute path list of images in [clip] after copying
        self.list_path_matting = list()  # absolute path list of images in [matting] after copying

        self.list_path_debug = list()  # absolute path list of images in [matting] in debug datasets
        self.split_proportion = list()

    def run(self):
        """
        Entry to generate datasets
        """
        self.remove_error_files()
        self.split_datasets_and_save()
        self.generate_mask()
        self.generate_txt(self.cfg["path_save"], self.list_path_matting)
        self.generate_trimap()
        self.generate_alpha()
        self.generate_debug()
        self.generate_mean_std()

    @staticmethod
    def list_file_by_recursive(dirname, match):
        """
        Recursive enumeration of absolute path of image file
        """
        filelist = []
        if os.path.isfile(dirname):
            if dirname.split(".")[-1] in match:
                filelist.append(dirname)
        else:
            for root, _, files in os.walk(dirname):
                for name in files:
                    if name.split(".")[-1] in match:
                        filelist.append(os.path.join(root, name))
        return filelist

    def remove_error_files(self):
        """
        Remove error file
        """
        for error_file in self.cfg["list_error_files"]:
            if os.path.exists(error_file):
                os.remove(error_file)

    @staticmethod
    def safe_makedirs(path_dir):
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)

    def split_datasets_and_save(self):
        """
        Split and save datasets
        """
        path_clip = os.path.join(self.cfg["path_mt_human"], "clip_img")
        path_matting = os.path.join(self.cfg["path_mt_human"], "matting")

        list_clip = self.list_file_by_recursive(path_clip, match=["png", "jpg"])
        list_matting = self.list_file_by_recursive(path_matting, match=["png", "jpg"])

        list_clip = sorted(list_clip)
        list_matting = sorted(list_matting)

        # datasets split p (proportion)： [0, x1, x2, total images]
        s = 0
        sums = len(list_clip)
        p = [0]
        for x in self.cfg["proportion"].split(":")[:-1]:
            s += sums * int(x) // 10
            p.append(s)
        p.append(sums)

        self.split_proportion = p

        list_clip_matting = list(zip(list_clip, list_matting))
        for idx, one in enumerate(["train", "eval", "test"]):
            list_stage = list_clip_matting[p[idx] : p[idx + 1]]
            if self.cfg["copy_pic"]:
                print("Copying source files to {} dir...".format(one))
            for item in list_stage:
                one_clip = item[0].replace(self.cfg["path_mt_human"], os.path.join(self.cfg["path_save"], one))
                one_matting = item[1].replace(self.cfg["path_mt_human"], os.path.join(self.cfg["path_save"], one))

                self.list_path_clip.append(one_clip)
                self.list_path_matting.append(one_matting)

                if self.cfg["copy_pic"]:
                    save_dir_clip = os.path.split(one_clip)[0]
                    save_dir_matting = os.path.split(one_matting)[0]

                    self.safe_makedirs(save_dir_clip)
                    self.safe_makedirs(save_dir_matting)

                    shutil.copyfile(item[0], one_clip)
                    shutil.copyfile(item[1], one_matting)

    def generate_mask(self):
        """
        Generate mask images from matting
        """
        if self.cfg["generate_mask"]:
            print("Generate mask ...")
            for one in self.list_path_matting:
                in_image = cv2.imread(one, cv2.IMREAD_UNCHANGED)
                alpha = in_image[:, :, 3]

                save_file = one.replace("/matting/", "/mask/")
                save_dir = os.path.split(save_file)[0]
                self.safe_makedirs(save_dir)

                cv2.imwrite(save_file, alpha)

    def generate_txt(self, path_save, file_list):
        """
        Generate [train, eval, test] .txt file
        """
        flag = False
        if path_save == self.cfg["path_save"] and self.cfg["generate_txt"]:
            print("Generate datasets txt ...")
            flag = True
        elif path_save == self.cfg["path_debug"] and self.cfg["generate_debug"]:
            print("Generate datasets_debug txt ...")
            flag = True

        if flag:
            dict_file = dict()
            for fl in ["train", "eval", "test"]:
                dict_file[fl] = open(os.path.join(path_save, fl, "{}.txt".format(fl)), "w")

            for one in file_list:
                path_split = one.split("/matting/")
                stage = path_split[0].split("/")[-1]
                content = path_split[-1]

                dict_file[stage].write(content + "\n")

            for fl in ["train", "eval", "test"]:
                dict_file[fl].close()

    @staticmethod
    def erode_dilate_fixed(msk, struct="ELLIPSE", size=(10, 10)):
        """
        Perform erode and dilate on a image, kernel size is fixed
        """
        if struct == "RECT":
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, size)
        elif struct == "CORSS":
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, size)
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)

        msk = msk / 255

        dilated = cv2.dilate(msk, kernel, iterations=1) * 255
        eroded = cv2.erode(msk, kernel, iterations=1) * 255

        res = dilated.copy()
        res[((dilated == 255) & (eroded == 0))] = 128

        return res

    @staticmethod
    def erode_dilate_random(alpha, struct="ELLIPSE"):
        """
        Perform erode and dilate on a image, kernel size is random
        """
        k_size = random.choice(range(3, 10))
        iterations = np.random.randint(1, 2)

        if struct == "RECT":
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
        elif struct == "CORSS":
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (k_size, k_size))
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))

        dilated = cv2.dilate(alpha, kernel, iterations)
        eroded = cv2.erode(alpha, kernel, iterations)
        trimap = np.zeros(alpha.shape)
        trimap.fill(128)
        trimap[eroded >= 255] = 255
        trimap[dilated <= 0] = 0

        return trimap

    def generate_trimap(self):
        """
        Generate trimap from mask
        """
        if self.cfg["generate_trimap"]:
            print("Generate trimap ...")
            for one in self.list_path_matting:
                mask_file = one.replace("/matting/", "/mask/")

                save_file = one.replace("/matting/", "/trimap/")
                save_dir = os.path.split(save_file)[0]
                self.safe_makedirs(save_dir)

                msk = cv2.imread(mask_file, 0)  # 0：read gray image

                if self.cfg["fixed_ksize"]:
                    trimap = self.erode_dilate_fixed(msk, size=(self.cfg["ksize"], self.cfg["ksize"]))
                else:
                    trimap = self.erode_dilate_random(msk)

                cv2.imwrite(save_file, trimap)

    def generate_alpha(self):
        """
        Generate alpha from mask
        """
        if self.cfg["generate_alpha"]:
            print("Generate alpha ...")
            dir_names = ["train", "eval", "test"]
            for dirs in dir_names:
                path_mask = os.path.join(self.cfg["path_save"], dirs, "mask")
                path_alpha = os.path.join(self.cfg["path_save"], dirs, "alpha")
                print("Copying {} mask into alpha...".format(dirs))
                copy_tree(path_mask, path_alpha)

    @staticmethod
    def safe_modify_file_name(file_name):
        if not os.path.exists(file_name):
            if "jpg" in file_name:
                return file_name.replace("jpg", "png")
            return file_name.replace("png", "jpg")

        return file_name

    def generate_debug(self):
        """
        Generate debug datasets

        Extract a certain number from the divided data set as the debug datasets:
            splited datasets：  [0, x1, x2, sum]
            extract num：  [70， 20， 10]
            extraction rules： 0:70, x1:x1+20, x2:x2+10
        """
        if self.cfg["generate_debug"]:
            print("Generate datasets_debug ...")
            list_data_type = ["alpha", "clip_img", "mask", "matting", "trimap"]

            p = list()
            for x in self.cfg["proportion"].split(":"):
                p.append(self.cfg["debug_pic_nums"] * int(x) // 10)

            for i, spl_n in enumerate(self.split_proportion[:-1]):
                spl_start = spl_n
                spl_end = spl_n + p[i]

                list_file_src = self.list_path_matting[spl_start:spl_end]
                for file_path in list_file_src:
                    split_src = file_path.split("/matting/")

                    for dt in list_data_type:
                        file_src = os.path.join(split_src[0], dt, split_src[-1])
                        if dt == "clip_img":
                            file_src = file_src.replace("matting_", "clip_")

                        file_src = self.safe_modify_file_name(file_src)

                        file_dst = file_src.replace(self.cfg["path_save"], self.cfg["path_debug"])
                        save_dir = os.path.split(file_dst)[0]
                        self.safe_makedirs(save_dir)

                        if dt == "matting":
                            self.list_path_debug.append(file_dst)

                        shutil.copyfile(file_src, file_dst)
            self.generate_txt(self.cfg["path_debug"], self.list_path_debug)

    def generate_mean_std(self):
        """
        Generate train and eval datasets mean/std

        example output:
            Total images: 30983
            mean_clip:    [0.39999, 0.43269, 0.49550]  [101, 110, 126]
            std_clip:     [0.24667, 0.24772, 0.26308]  [62, 63, 67]
            mean_trimap:  [0.48881, 0.48881, 0.48881]  [124, 124, 124]
            std_trimap:   [0.44594, 0.44594, 0.44594]  [113, 113, 113]
        """
        if self.cfg["generate_mean_std"]:
            print("Generate train and eval datasets mean/std ...")

            n = 0
            mean_clip = [0] * 3
            std_clip = [0] * 3
            mean_trimap = [0] * 3
            std_trimap = [0] * 3
            for one in self.list_path_clip:
                if "train" in one or "eval" in one:
                    n += 1

                    src_clip = cv2.imread(one)
                    mean_clip[0] += np.mean(src_clip[:, :, 0])
                    mean_clip[1] += np.mean(src_clip[:, :, 1])
                    mean_clip[2] += np.mean(src_clip[:, :, 2])
                    std_clip[0] += np.std(src_clip[:, :, 0])
                    std_clip[1] += np.std(src_clip[:, :, 1])
                    std_clip[2] += np.std(src_clip[:, :, 2])

                    file_trimap = one.replace("/clip_img/", "/trimap/").replace("/clip_", "/matting_")
                    file_trimap = self.safe_modify_file_name(file_trimap)
                    src_trimap = cv2.imread(file_trimap)
                    mean_trimap[0] += np.mean(src_trimap[:, :, 0])
                    mean_trimap[1] += np.mean(src_trimap[:, :, 1])
                    mean_trimap[2] += np.mean(src_trimap[:, :, 2])
                    std_trimap[0] += np.std(src_trimap[:, :, 0])
                    std_trimap[1] += np.std(src_trimap[:, :, 1])
                    std_trimap[2] += np.std(src_trimap[:, :, 2])

            print("Total images: {}".format(str(n)))
            print(
                "mean_clip: ", [float("{:.5f}".format(x / n / 255)) for x in mean_clip], [int(x / n) for x in mean_clip]
            )
            print("std_clip: ", [float("{:.5f}".format(x / n / 255)) for x in std_clip], [int(x / n) for x in std_clip])
            print(
                "mean_trimap: ",
                [float("{:.5f}".format(x / n / 255)) for x in mean_trimap],
                [int(x / n) for x in mean_trimap],
            )
            print(
                "std_trimap: ",
                [float("{:.5f}".format(x / n / 255)) for x in std_trimap],
                [int(x / n) for x in std_trimap],
            )


def get_args():
    parser = argparse.ArgumentParser(description="semantic human matting !")
    parser.add_argument("--yaml_path", type=str, default="../config.yaml", help="yaml config file path")
    args = parser.parse_args()
    print(args)
    return args


def get_config_from_yaml(args):
    fd = open(args.yaml_path, "r")
    content = fd.read()
    fd.close()

    cfg = yaml.load(content, Loader=yaml.FullLoader)
    cfg = cfg["generate_data"]

    return cfg


if __name__ == "__main__":
    config = get_config_from_yaml(get_args())
    gd = GenerateData(config)
    gd.run()
