"""
推理文件
"""

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
from help.config import Config
from src.datasets.ava_dataset import Ava
from src.utils.meters import AVAMeter
from src.utils import logging
import numpy as np
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, InProtobufVector, MxProtobufIn, StringVector


def parse_args():
    """set and check parameters."""
    parser = argparse.ArgumentParser(description="SLOWFAST for AVA Dataset")
    parser.add_argument("--pipeline", type=str,
                        default="../data/config/slowfast.pipeline", help="SDK infer pipeline")
    parser.add_argument("--data_dir", type=str, default="../data/input",
                        help="Dataset contain frames and ava_annotations")
    args_opt = parser.parse_args()
    return args_opt


def send_tensor_input(stream_name_, plugin_id, input_data, input_shape, stream_manager):
    """" send tensor data to om """
    tensor_list = MxpiDataType.MxpiTensorPackageList()
    tensor_pkg = tensor_list.tensorPackageVec.add()
    # init tensor vector
    tensor_vec = tensor_pkg.tensorVec.add()
    tensor_vec.deviceId = 0
    tensor_vec.memType = 0
    tensor_vec.tensorShape.extend(input_shape)
    tensor_vec.tensorDataType = 0
    tensor_vec.dataStr = input_data
    tensor_vec.tensorDataSize = len(input_data)

    return send_protobuf(stream_name_, plugin_id, tensor_list, stream_manager)


def send_protobuf(stream_name_, plugin_id, pkg_list, stream_manager):
    """" send data buffer to om """
    protobuf = MxProtobufIn()
    protobuf.key = "appsrc{}".format(plugin_id).encode('utf-8')
    protobuf.type = b"MxTools.MxpiTensorPackageList"
    protobuf.protobuf = pkg_list.SerializeToString()
    protobuf_vec = InProtobufVector()
    protobuf_vec.push_back(protobuf)
    err_code = stream_manager.SendProtobuf(
        stream_name_, plugin_id, protobuf_vec)
    if err_code != 0:
        logging.error(
            "Failed to send data to stream, stream_name(%s), plugin_id(%s), element_name(%s), "
            "err_code(%s).", stream_name_, plugin_id,
            "appsrc{}".format(plugin_id).encode('utf-8'), err_code)
        return False
    return True


def get_result(stream_name_, stream_manager, out_plugin_id=0):
    """get_result"""
    key_vec = StringVector()
    key_vec.push_back(b'mxpi_tensorinfer0')
    infer_result = stream_manager.GetProtobuf(
        stream_name_, out_plugin_id, key_vec)
    if infer_result.size() == 0:
        print("inferResult is null")
        return None
    result_ = MxpiDataType.MxpiTensorPackageList()
    result_.ParseFromString(infer_result[0].messageBuf)
    return np.frombuffer(result_.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32)


def pack_input(input_):
    for ii in range(len(input_)):
        input_[ii] = input_[ii][np.newaxis, :]
    return np.concatenate(input_, axis=0)


def main():
    args = parse_args()
    config = Config()
    config.AVA.FRAME_LIST_DIR = args.data_dir + config.AVA.ANN_DIR
    config.AVA.ANNOTATION_DIR = args.data_dir + config.AVA.ANN_DIR
    config.AVA.FRAME_DIR = args.data_dir + config.AVA.FRA_DIR
    stream_name = config.sdk_pipeline_name
    # init stream manager
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        return

    # create streams by pipeline config file
    with open(args.pipeline, "rb") as f:
        pipeline_str = f.read()
    ret = streamManagerApi.CreateMultipleStreams(pipeline_str)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        return

    # setup logger
    logger = logging.get_logger(__name__)
    logging.setup_logging()
    logger.info(config)
    dataset = Ava(config, "test")
    meter = AVAMeter(len(dataset)/config.BATCH_SIZE, config, mode="test")
    meter.iter_tic()
    count = 0
    cur_iter = 0
    slowpaths = []
    fastpaths = []
    boxess = []
    ori_boxess = []
    metadatas = []
    masks = []
    for slowpath, fastpath, boxes, _, ori_boxes, metadata, mask in dataset:
        count += 1
        slowpaths.append(slowpath.astype(np.float32))
        fastpaths.append(fastpath.astype(np.float32))
        boxess.append(boxes.astype(np.float32))
        ori_boxess.append(ori_boxes.astype(np.float32))
        metadatas.append(metadata.astype(np.float32))
        masks.append(mask.astype(np.float32))
        if count == config.BATCH_SIZE:

            slowpath_ = pack_input(slowpaths)
            fastpath_ = pack_input(fastpaths)
            boxes_ = pack_input(boxess)
            ori_boxes_ = pack_input(ori_boxess)
            metadata_ = pack_input(metadatas)
            mask_ = pack_input(masks)
            res0 = send_tensor_input(
                stream_name, 0, slowpath_.tobytes(), slowpath_.shape, streamManagerApi)
            res1 = send_tensor_input(
                stream_name, 1, fastpath_.tobytes(), fastpath_.shape, streamManagerApi)
            res2 = send_tensor_input(
                stream_name, 2, boxes_.tobytes(), boxes_.shape, streamManagerApi)

            if not res0 or not res1 or not res2:
                return
            preds = get_result(stream_name, streamManagerApi)
            preds1 = preds.reshape(mask_.shape + (config.MODEL.NUM_CLASSES,))
            mask_ = np.array(mask_).astype(bool)
            preds = preds1[mask_]
            padded_idx = np.tile(np.arange(mask_.shape[0]).reshape(
                (-1, 1, 1)), (1, mask_.shape[1], 1))
            ori_boxes_ = np.concatenate(
                (padded_idx, np.array(ori_boxes_)), axis=2)[mask_]
            metadata_ = np.array(metadata_)[mask_]
            meter.update_stats(preds, ori_boxes_, metadata_)
            meter.log_iter_stats(None, cur_iter)
            meter.iter_tic()
            count = 0
            slowpaths = []
            fastpaths = []
            boxess = []
            ori_boxess = []
            metadatas = []
            masks = []
            cur_iter += 1

    meter.finalize_metrics()
    streamManagerApi.DestroyAllStreams()


if __name__ == "__main__":
    main()
