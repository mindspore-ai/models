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

import contextlib
import argparse
import os

import numpy as np
from third_party import arithmetic_coding_base
from src.tools.utils import bin_loader, chamfer_distance
from src.tools.octree_base import Octree, deserialize_depth_first


def softmax(x):
    if x.ndim == 1:
        res = np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))
    elif x.ndim == 2:
        res = np.zeros_like(x)
        for i in range(len(x)):
            part_x = x[i]
            res[i] = np.exp(part_x - np.max(part_x)) / np.sum(np.exp(part_x - np.max(part_x)))
    return res


def def_arguments():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dataset", "-t", type=str, default="../data/infernece_dataset/")
    parser.add_argument("--ori_dataset", type=str, default="../../KITTI/object/training/velodyne")
    parser.add_argument(
        "--pipeline", "-m", type=str, default="../data/config/octsqueeze.pipeline", help="route of model"
    )
    parser.add_argument("--eval", action="store_true", default=True)
    parser.add_argument("--save_result", action="store_true", default=False)
    parser.add_argument("--compression", "-c", type=str, default="./compression/")
    return parser.parse_args()


def run_octsqueeze(dataset_path, precision_oct, if_eval=True):
    frames = []
    for filename in sorted(os.listdir(dataset_path + "/" + str(precision_oct))):
        frames.append(dataset_path + "/" + str(precision_oct) + "/{}/input.txt".format(filename))
    bpip_CD = np.empty([len(frames), 2])
    time_sum = 0
    frame_num = len(frames)

    for frame_idx, frame in enumerate(frames):
        frame_n = frame_idx + 7000
        frame_name = frame.split("/")[-2]
        max_range = (2**13 - 1) * 0.01

        outputs = np.fromfile("results/{}/{:06d}_results.bin".format(precision_oct, frame_n), dtype=np.float32).reshape(
            -1, 256
        )
        outputs = softmax(outputs)

        if if_eval:
            occupancy_stream = np.loadtxt(frame.replace("input", "gt"), dtype=int, delimiter=" ")
            # Write compressed file
            f_frame = open(os.path.join(args.compression, frame_name + ".bin"), "wb")
            with contextlib.closing(arithmetic_coding_base.BitOutputStream(f_frame)) as bitout:
                enc = arithmetic_coding_base.ArithmeticEncoder(32, bitout)

                for node_idx in range(outputs.shape[0]):
                    gt_occupancy = int(occupancy_stream[node_idx])
                    distribution = outputs[node_idx]
                    distribution_value = np.floor(distribution * 10000)
                    label = np.ones(257)
                    label[:256] += distribution_value
                    frequencies = arithmetic_coding_base.SimpleFrequencyTable(label.astype(int))
                    enc.write(frequencies, gt_occupancy)

                enc.finish()
            f_frame.close()
            file_size = os.path.getsize(os.path.join(args.compression, frame_name + ".bin")) * 8

            # occupancy stream in the compressed binary file
            occupancy_stream = occupancy_stream.astype(np.int)
            recon_tree = Octree(max_range=max_range, precision=precision_oct)
            recon_tree, recon_points = deserialize_depth_first(iter(occupancy_stream), recon_tree.max_depth, recon_tree)
            recon_points = np.array(recon_points).astype(np.float32)

            print("Precision {} - Frame {} done!".format(precision_oct, frame))
            # Calculate bpip, CD, PSNR
            orignal_points = bin_loader(os.path.join(args.ori_dataset, frame_name + ".bin"))
            orignal_points = np.array(orignal_points)[:, :3]
            CD = chamfer_distance(recon_points, orignal_points)

            compressed_size = file_size
            bpip = compressed_size / orignal_points.shape[0]

            bpip_CD[frame_idx] = [bpip, CD]
            print("bpip: {}; CD: {}".format(bpip, CD))
    if if_eval:
        print(
            "Precision: {} - Totally time cost: {} - Average cost: {}".format(
                precision_oct, time_sum, time_sum / frame_num
            )
        )
    return bpip_CD


if __name__ == "__main__":
    args = def_arguments()
    if not os.path.exists(args.compression):
        os.makedirs(args.compression)
    # Evaluate test data at four bitrate whose max point-to-point error should less then [0.01 0.02, 0.04, 0.08]
    precision_list = [0.01, 0.02, 0.04, 0.08]
    bpip_CD_all = np.empty([len(precision_list), 2])
    for idx, precision in enumerate(precision_list):
        bpip_CD_temp = run_octsqueeze(args.test_dataset, precision, True)
        bpip_CD_all[idx] = np.mean(bpip_CD_temp, axis=0)
    if args.eval:
        print("bpip and chamfer distance at different rates:")
        np.savetxt("acc.txt", bpip_CD_all)
        print(bpip_CD_all)
