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
"""Feature extraction program from KITTI dataset"""

import os
import argparse
from array import array

import numpy as np

from src.tools.utils import bin_loader, node_data2int
from src.tools.octree_base import Octree
from src.dataset import normalization


def feature_extraction(octree):
    """Extract feature from octree structure"""

    def traverse_extraction(node):
        if node and node.attribute != "leaf":
            node_cur = node
            node_info = [node_data2int(node.data)]
            if node.depth < 4:
                for _ in range(node.depth):
                    node_info += [
                        node_cur.position[0],
                        node_cur.position[1],
                        node_cur.position[2],
                        node_cur.octant,
                        node_cur.depth,
                        node_data2int(node_cur.parent.data),
                    ]
                    node_cur = node_cur.parent
                for _ in range(4 - node.depth):
                    node_info += [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            else:
                for _ in range(4):
                    node_info += [
                        node_cur.position[0],
                        node_cur.position[1],
                        node_cur.position[2],
                        node_cur.octant,
                        node_cur.depth,
                        node_data2int(node_cur.parent.data),
                    ]
                    node_cur = node_cur.parent
            feature_branch.append(node_info)
            for branch_id in range(8):
                traverse_extraction(node.branches[branch_id])

    feature_branch = []
    traverse_extraction(octree)

    return feature_branch


def feature_arrange(nodes, points_num):
    nodes_num = len(nodes)
    node_info = {
        "gt": np.empty(nodes_num),
        "cur_node": np.empty([nodes_num, 6]),
        "parent1": np.empty([nodes_num, 6]),
        "parent2": np.empty([nodes_num, 6]),
        "parent3": np.empty([nodes_num, 6]),
        "points_num": points_num,
    }

    for i in range(nodes_num):
        node_index = i
        feature = nodes[node_index, :].copy()
        node_info["gt"][i] = feature[0]
        node_info["cur_node"][i, :] = feature[1:7]
        node_info["parent1"][i, :] = feature[7:13]
        node_info["parent2"][i, :] = feature[13:19]
        node_info["parent3"][i, :] = feature[19:25]

    return node_info


def post_processing_310(node_info):
    """normalize and rearrange data for 310 inference"""

    gt = node_info["gt"].astype(np.int32)
    points_num = node_info["points_num"]
    cur_node = normalization(node_info["cur_node"].astype(np.float32))
    parent1 = normalization(node_info["parent1"].astype(np.float32))
    parent2 = normalization(node_info["parent2"].astype(np.float32))
    parent3 = normalization(node_info["parent3"].astype(np.float32))
    feature = np.concatenate([cur_node, parent1, parent2, parent3], axis=1)

    return gt, points_num, feature


def write_data(data_dir, points_num, inp, gt):
    """Write data for 310 inference"""

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    # points_num
    fpath = os.path.join(data_dir, "points_num.bin")
    fp = open(fpath, "wb")
    var = array("i", [points_num])
    var.tofile(fp)
    fp.close()

    # feature
    fpath = os.path.join(data_dir, "input.bin")
    fp = open(fpath, "wb")
    inp.flatten().tofile(fp, sep="\n", format="%s")
    fp.close()

    # gt
    fpath = os.path.join(data_dir, "gt.bin")
    fp = open(fpath, "wb")
    var = array("i", gt)
    var.tofile(fp)
    fp.close()


def KITTI_feature(input_route, output_route, min_file, max_file):
    if not os.path.exists(output_route):
        os.makedirs(output_route)
    if args.mode == "train":
        # For training data, precision is set at 1cm
        precision = 0.01
        for i in range(min_file, max_file + 1, 1):
            data_route = input_route + str(i).zfill(6) + ".bin"
            pcd_example = bin_loader(data_route)
            points_num = pcd_example.shape[0]

            # Fix max range
            max_range = (2**13 - 1) * 0.01

            tree = Octree(max_range=max_range, precision=precision)
            for j in range(pcd_example.shape[0]):
                tree.insert_node(tree, tree.size, None, pcd_example[j], 0)

            feature_branch = feature_extraction(tree)
            feature_branch = np.array(feature_branch)
            node_info = feature_arrange(feature_branch, points_num)

            np.save(output_route + str(i).zfill(6) + ".npy", node_info)
            print("Feature extraction from {} is done".format(output_route + str(i).zfill(6) + ".bin"))
    elif args.mode == "inference":
        # For inference, 4 bitrates, aka 4 precsion: 1cm, 2cm, 4cm, 8cm, are calculated
        precision_oct = [0.01, 0.02, 0.04, 0.08]
        for precision in precision_oct:
            output_route_precision = os.path.join(output_route, str(precision))
            if not os.path.exists(output_route_precision):
                os.makedirs(output_route_precision)
            for i in range(min_file, max_file + 1, 1):
                data_route = input_route + str(i).zfill(6) + ".bin"
                pcd_example = bin_loader(data_route)
                points_num = pcd_example.shape[0]

                # Fix max range
                max_range = (2**13 - 1) * 0.01

                tree = Octree(max_range=max_range, precision=precision)
                for j in range(pcd_example.shape[0]):
                    tree.insert_node(tree, tree.size, None, pcd_example[j], 0)

                feature_branch = feature_extraction(tree)
                feature_branch = np.array(feature_branch)
                node_info = feature_arrange(feature_branch, points_num)

                gt, points_num, feature = post_processing_310(node_info)
                data_dir = os.path.join(output_route_precision, str(i).zfill(6))
                write_data(data_dir, points_num, feature, gt)

                print("Precision: {} ;Feature extraction from {} is done".format(precision, data_dir + ".bin"))


if __name__ == "__main__":
    # Generate data according to given parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_route", "-r", type=str, default="/home/OctSqueeze/KITTI_sub/")
    parser.add_argument("--output_route", "-o", type=str, default="/home/OctSqueeze/feature/inference_sub/")
    parser.add_argument("--min_file", "-i", type=int, default=7000, help="min file ID")
    parser.add_argument("--max_file", "-a", type=int, default=7099, help="max file ID")
    parser.add_argument(
        "--mode", "-m", type=str, default="inference", choices=["train", "inference"], help="train or inference"
    )
    args = parser.parse_args()

    KITTI_feature(
        input_route=args.input_route, output_route=args.output_route, min_file=args.min_file, max_file=args.max_file
    )
