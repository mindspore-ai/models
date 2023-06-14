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

import argparse
import time
import os
import numpy as np
from StreamManagerApi import StreamManagerApi, StringVector
from StreamManagerApi import MxDataInput, InProtobufVector, MxProtobufIn
import MxpiDataType_pb2 as MxpiDataType
from third_party import arithmetic_coding_base
from src.tools.octree_base import Octree, deserialize_depth_first
from src.tools.utils import bin_loader, chamfer_distance


def def_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dataset", "-t", type=str, default="../data/infernece_dataset/")
    parser.add_argument("--ori_dataset", type=str, default="../../KITTI/object/training/velodyne")
    parser.add_argument(
        "--pipeline", "-m", type=str, default="../data/config/octsqueeze.pipeline", help="route of model"
    )
    parser.add_argument("--eval", action="store_true", default=False)
    parser.add_argument("--save_result", action="store_true", default=True)
    parser.add_argument("--compression", "-c", type=str, default="./compression/")
    return parser.parse_args()


def softmax(x):
    if x.ndim == 1:
        res = np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))
    elif x.ndim == 2:
        res = np.zeros_like(x)
        for i in range(len(x)):
            part_x = x[i]
            res[i] = np.exp(part_x - np.max(part_x)) / np.sum(np.exp(part_x - np.max(part_x)))
    return res


def create_protobufVec(data, key):
    """
    create_protobufVec
    """
    data_input = MxDataInput()
    data_input.data = data.tobytes()
    tensorPackageList = MxpiDataType.MxpiTensorPackageList()
    tensorPackage = tensorPackageList.tensorPackageVec.add()
    tensorVec = tensorPackage.tensorVec.add()
    tensorVec.deviceId = 0
    tensorVec.memType = 0
    for t in data.shape:
        tensorVec.tensorShape.append(t)
    tensorVec.dataStr = data_input.data
    tensorVec.tensorDataSize = len(data.tobytes())
    protobufVec = InProtobufVector()
    protobuf = MxProtobufIn()
    protobuf.key = key
    protobuf.type = b"MxTools.MxpiTensorPackageList"
    protobuf.protobuf = tensorPackageList.SerializeToString()
    protobufVec.push_back(protobuf)

    return protobufVec


def infer(stream_manager1, stream_name, input_data):
    """
    infer
    """
    stream_manager1.SendProtobuf(stream_name, b"appsrc0", create_protobufVec(input_data, b"appsrc0"))
    keyVec = StringVector()
    keyVec.push_back(b"mxpi_tensorinfer0")
    infer_result = stream_manager1.GetProtobuf(stream_name, 0, keyVec)
    if infer_result.size() == 0:
        print("inferResult is null")
        exit()
    if infer_result[0].errorCode != 0:
        print("GetProtobuf error. errorCode=%d" % (infer_result[0].errorCode))
        exit()
    # get infer result

    result = MxpiDataType.MxpiTensorPackageList()
    result.ParseFromString(infer_result[0].messageBuf)
    # convert the inference result to Numpy array
    score = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32).reshape(-1, 256)

    return score


def run_octsqueeze(dataset_path, precision_oct, is_eval=True):
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

        input_data = np.loadtxt(frame, dtype=np.float32, delimiter=" ").reshape(1, 1, -1, 24)
        length = input_data.shape[2]

        outputs = []
        steps = length // 1000 + 1
        for idx in range(steps):
            if idx < steps - 1:
                input_ = input_data[:, :, idx * 1000 : idx * 1000 + 1000]
            else:
                input_ = np.zeros((1, 1, 1000, 24)).astype(np.float32)
                input_[:, :, : length - idx * 1000] = input_data[:, :, idx * 1000 :]
            start = time.time()
            output = infer(stream_manager, b"octsqueeze", input_)
            end = time.time()
            time_cost = end - start
            time_sum = time_sum + time_cost
            outputs.append(output)
        outputs = np.concatenate(outputs, axis=0)[:length]
        if args.save_result:
            outputs.tofile("results/{}/{:06d}_results.bin".format(precision_oct, frame_n))
        outputs = softmax(outputs)

        if is_eval:
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

            # compressed_size = entropy
            compressed_size = file_size
            bpip = compressed_size / orignal_points.shape[0]

            bpip_CD[frame_idx] = [bpip, CD]
            print("bpip: {}; CD: {}".format(bpip, CD))
    if is_eval:
        print(
            "Precision: {} - Totally time cost: {} - Average cost: {}".format(
                precision_oct, time_sum, time_sum / frame_num
            )
        )
    return bpip_CD


if __name__ == "__main__":
    args = def_arguments()
    stream_manager = StreamManagerApi()
    ret = stream_manager.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open(args.pipeline, "rb") as f:
        pipeline = f.read()
    ret = stream_manager.CreateMultipleStreams(pipeline)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()
    if not os.path.exists("results"):
        os.makedirs("results")
    if not os.path.exists("results/0.01"):
        os.makedirs("results/0.01")
    if not os.path.exists("results/0.02"):
        os.makedirs("results/0.02")
    if not os.path.exists("results/0.04"):
        os.makedirs("results/0.04")
    if not os.path.exists("results/0.08"):
        os.makedirs("results/0.08")

    if args.eval:
        if not os.path.exists(args.compression):
            os.makedirs(args.compression)
    # Evaluate test data at four bitrate whose max point-to-point error should less then [0.01 0.02, 0.04, 0.08]
    precision_list = [0.01, 0.02, 0.04, 0.08]
    bpip_CD_all = np.empty([len(precision_list), 2])
    for index, precision in enumerate(precision_list):
        bpip_CD_temp = run_octsqueeze(args.test_dataset, precision, args.eval)
        bpip_CD_all[index] = np.mean(bpip_CD_temp, axis=0)
    if args.eval:
        print("bpip and chamfer distance at different rates:")
        np.savetxt("acc.txt", bpip_CD_all)
        print(bpip_CD_all)
