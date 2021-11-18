'''
The scripts to execute sdk infer
'''
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

import argparse
import os
import glob
import time
import math
import PIL.Image as Image

import MxpiDataType_pb2 as MxpiDataType
import numpy as np
from StreamManagerApi import StreamManagerApi, InProtobufVector, \
    MxProtobufIn, StringVector


def parse_args():
    """set and check parameters."""
    parser = argparse.ArgumentParser(description="BRDNet process")
    parser.add_argument("--pipeline", type=str, default=None, help="SDK infer pipeline")
    parser.add_argument("--clean_image_path", type=str, default=None, help="root path of image without noise")
    parser.add_argument('--image_width', default=500, type=int, help='resized image width')
    parser.add_argument('--image_height', default=500, type=int, help='resized image height')
    parser.add_argument('--channel', default=3, type=int
                        , help='image channel, 3 for color, 1 for gray')
    parser.add_argument('--sigma', type=int, default=15, help='level of noise')
    args_opt = parser.parse_args()
    return args_opt

def calculate_psnr(image1, image2):
    image1 = np.float64(image1)
    image2 = np.float64(image2)
    diff = image1 - image2
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff**2.))
    return 20*math.log10(1.0/rmse)

def send_source_data(appsrc_id, tensor, stream_name, stream_manager):
    """
    Construct the input of the stream,
    send inputs data to a specified stream based on streamName.

    Returns:
        bool: send data success or not
    """
    tensor_package_list = MxpiDataType.MxpiTensorPackageList()
    tensor_package = tensor_package_list.tensorPackageVec.add()
    array_bytes = tensor.tobytes()
    tensor_vec = tensor_package.tensorVec.add()
    tensor_vec.deviceId = 0
    tensor_vec.memType = 0
    for i in tensor.shape:
        tensor_vec.tensorShape.append(i)
    tensor_vec.dataStr = array_bytes
    tensor_vec.tensorDataSize = len(array_bytes)
    key = "appsrc{}".format(appsrc_id).encode('utf-8')
    protobuf_vec = InProtobufVector()
    protobuf = MxProtobufIn()
    protobuf.key = key
    protobuf.type = b'MxTools.MxpiTensorPackageList'
    protobuf.protobuf = tensor_package_list.SerializeToString()
    protobuf_vec.push_back(protobuf)

    ret = stream_manager.SendProtobuf(stream_name, appsrc_id, protobuf_vec)
    if ret < 0:
        print("Failed to send data to stream.")
        return False
    return True


def main():
    """
    read pipeline and do infer
    """

    args = parse_args()

    # init stream manager
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        return

    # create streams by pipeline config file
    with open(os.path.realpath(args.pipeline), 'rb') as f:
        pipeline_str = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipeline_str)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        return

    stream_name = b'brdnet'
    infer_total_time = 0
    psnr = []   #after denoise
    image_list = glob.glob(os.path.join(args.clean_image_path, '*'))
    if not os.path.exists("./outputs"):
        os.makedirs("./outputs")
    with open("./outputs/denoise_results.txt", 'w') as f:
        for image in sorted(image_list):
            print("Denosing image:", image)# read image
            if args.channel == 3:
                img_clean = np.array(Image.open(image).resize((args.image_width, args.image_height), \
                                     Image.ANTIALIAS), dtype='float32') / 255.0
            else:
                img_clean = np.expand_dims(np.array(Image.open(image).resize((args.image_width, \
                            args.image_height), Image.ANTIALIAS).convert('L'), dtype='float32') / 255.0, axis=2)
            np.random.seed(0) #obtain the same random data when it is in the test phase
            img_test = img_clean + np.random.normal(0, args.sigma/255.0, img_clean.shape).astype(np.float32)#HWC
            noise_image = np.expand_dims(img_test.transpose((2, 0, 1)), 0)#NCHW

            if not send_source_data(0, noise_image, stream_name, stream_manager_api):
                return
            # Obtain the inference result by specifying streamName and uniqueId.
            key_vec = StringVector()
            key_vec.push_back(b'modelInfer')
            start_time = time.time()
            infer_result = stream_manager_api.GetProtobuf(stream_name, 0, key_vec)
            infer_total_time += time.time() - start_time
            if infer_result.size() == 0:
                print("inferResult is null")
                return
            if infer_result[0].errorCode != 0:
                print("GetProtobuf error. errorCode=%d" % (infer_result[0].errorCode))
                return
            result = MxpiDataType.MxpiTensorPackageList()
            result.ParseFromString(infer_result[0].messageBuf)
            res = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype='<f4')
            y_predict = res.reshape(args.channel, args.image_height, args.image_width)
            img_out = y_predict.transpose((1, 2, 0))#HWC
            img_out = np.clip(img_out, 0, 1)
            psnr_denoised = calculate_psnr(img_clean, img_out)
            psnr.append(psnr_denoised)
            print(image, ": psnr_denoised: ", " ", psnr_denoised)
            print(image, ": psnr_denoised: ", " ", psnr_denoised, file=f)
            filename = image.split('/')[-1].split('.')[0]    # get the name of image file
            img_out.tofile(os.path.join("./outputs", filename+'_denoise.bin'))
        psnr_avg = sum(psnr)/len(psnr)
        print("Average PSNR:", psnr_avg)
        print("Average PSNR:", psnr_avg, file=f)
    print("Testing finished....")
    print("=======================================")
    print("The total time of inference is {} s".format(infer_total_time))
    print("=======================================")

    # destroy streams
    stream_manager_api.DestroyAllStreams()

if __name__ == '__main__':
    main()
