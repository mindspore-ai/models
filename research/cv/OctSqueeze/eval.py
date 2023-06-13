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

import contextlib
import argparse
import time
import os

import numpy as np
import pandas as pd
import mindspore
from mindspore import Model, context, load_checkpoint, load_param_into_net, ops, Tensor

from third_party import arithmetic_coding_base

from process_data import feature_extraction, feature_arrange
from src.dataset import normalization
from src.tools.utils import bin_loader, write_ply, chamfer_distance
from src.tools.octree_base import Octree, deserialize_depth_first
import src.network as network


def def_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dataset", "-t", type=str, default="/home/OctSqueeze/test_dataset/")
    parser.add_argument("--compression", "-c", type=str, default="/home/OctSqueeze/experiment/compression/")
    parser.add_argument("--recon", "-r", type=str, default="/home/OctSqueeze/experiment/recon/")
    parser.add_argument(
        "--model", "-m", type=str, default="/home/OctSqueeze/checkpoint/octsqueeze.ckpt", help="route of model"
    )
    parser.add_argument(
        "--device_target",
        type=str,
        default="Ascend",
        choices=["Ascend", "GPU", "CPU"],
        help="device where the code will be implemented",
    )
    parser.add_argument("--device_id", type=int, default=0)
    return parser.parse_args()


def compression_decompression_simulation(dataset_path, precision_oct):
    # read test data names
    frames = []
    for filename in os.listdir(dataset_path):
        if filename.endswith(".bin"):
            frames.append("{}".format(filename))

    ## Create networks
    net = network.OctSqueezeNet()
    param_dict = load_checkpoint(args.model)
    load_param_into_net(net, param_dict)

    bpip_CD = np.empty([len(frames), 2])
    time_sum = 0
    frame_num = len(frames)

    metrics = {}
    predict_net = Model(net, metrics=metrics)

    for frame_idx, frame in enumerate(frames):
        data_route = os.path.join(dataset_path, frame)
        pcd_example = bin_loader(data_route)
        points_num = pcd_example.shape[0]

        max_range = (2**13 - 1) * 0.01
        tree = Octree(max_range=max_range, precision=precision_oct)
        for j in range(pcd_example.shape[0]):
            tree.insert_node(tree, tree.size, None, pcd_example[j], 0)

        feature_branch = feature_extraction(tree)
        feature_branch = np.array(feature_branch)
        nodes = feature_arrange(feature_branch, points_num)

        start = time.time()
        cur_node = normalization(nodes["cur_node"].astype(np.float32))
        parent1 = normalization(nodes["parent1"].astype(np.float32))
        parent2 = normalization(nodes["parent2"].astype(np.float32))
        parent3 = normalization(nodes["parent3"].astype(np.float32))
        feature = Tensor(np.concatenate([cur_node, parent1, parent2, parent3], axis=1))

        output = predict_net.predict(feature)
        output.set_dtype(mindspore.float32)
        softmax = ops.Softmax()
        output = softmax(output)

        output = output.asnumpy()
        end = time.time()
        time_cost = end - start
        time_sum = time_sum + time_cost

        # Write compressed file
        f_frame = open(os.path.join(args.compression, frame), "wb")
        with contextlib.closing(arithmetic_coding_base.BitOutputStream(f_frame)) as bitout:
            enc = arithmetic_coding_base.ArithmeticEncoder(32, bitout)

            for node_idx in range(output.shape[0]):
                gt_occupancy = int(nodes["gt"][node_idx])
                distribution = output[node_idx]
                distribution_value = np.floor(distribution * 10000)
                label = np.ones(257)
                label[:256] += distribution_value
                frequencies = arithmetic_coding_base.SimpleFrequencyTable(label.astype(int))
                enc.write(frequencies, gt_occupancy)

            enc.finish()
        f_frame.close()
        file_size = os.path.getsize(os.path.join(args.compression, frame)) * 8

        # occupancy stream in the compressed binary file
        occupancy_stream = nodes["gt"].astype(np.int)
        recon_tree = Octree(max_range=max_range, precision=precision_oct)
        recon_tree, recon_points = deserialize_depth_first(iter(occupancy_stream), recon_tree.max_depth, recon_tree)
        recon_points = np.array(recon_points).astype(np.float32)

        df = pd.DataFrame(recon_points[:, :3], columns=["x", "y", "z"])
        recon_out = os.path.join(args.recon, frame) + ".ply"
        write_ply(recon_out, df)
        print("Precision {} - Frame {} done!".format(precision_oct, frame))

        # Calculate bpip, CD, PSNR
        orignal_points = bin_loader(os.path.join(args.test_dataset, frame))
        orignal_points = np.array(orignal_points)[:, :3]
        CD = chamfer_distance(recon_points, orignal_points)

        # compressed_size = entropy
        compressed_size = file_size
        bpip = compressed_size / orignal_points.shape[0]

        bpip_CD[frame_idx] = [bpip, CD]
        print("bpip: {}; CD: {}".format(bpip, CD))
    print(
        "Precision: {} - Totally time cost: {} - Average cost: {}".format(precision_oct, time_sum, time_sum / frame_num)
    )

    return bpip_CD


if __name__ == "__main__":
    # Evaluate test data at four bitrate whose max point-to-point error should less then [0.01 0.02, 0.04, 0.08]
    args = def_arguments()
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=args.device_id)
    precision_list = [0.01, 0.02, 0.04, 0.08]
    bpip_CD_all = np.empty([len(precision_list), 2])
    if not os.path.exists(args.compression):
        os.makedirs(args.compression)
    if not os.path.exists(args.recon):
        os.makedirs(args.recon)
    for idx, precision in enumerate(precision_list):
        bpip_CD_temp = compression_decompression_simulation(args.test_dataset, precision)
        bpip_CD_all[idx] = np.mean(bpip_CD_temp, axis=0)

    print("bpip and chamfer distance at different rates:")
    print(bpip_CD_all)
