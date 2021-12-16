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
import time
import numbers

import cv2
import numpy as np

import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, InProtobufVector, \
    MxProtobufIn, StringVector

def parse_args():
    """set and check parameters."""
    parser = argparse.ArgumentParser(description="R(2+1)D process")
    parser.add_argument("--pipeline", type=str, default="../data/config/r2p1d.pipeline", help="SDK infer pipeline")
    parser.add_argument("--dataset_root_path", type=str, default="../dataset", help="root path of images")
    args_opt = parser.parse_args()
    return args_opt

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


def get_resize_sizes(im_h, im_w, size):
    if im_w < im_h:
        ow = size
        oh = int(size * im_h / im_w)
    else:
        oh = size
        ow = int(size * im_w / im_h)
    return oh, ow

def resize_clip(clip, size, interpolation='bilinear'):
    '''
    resize the clip
    '''
    assert isinstance(clip[0], np.ndarray)
    if isinstance(size, numbers.Number):
        im_h, im_w, _ = clip[0].shape
        # Min spatial dim already matches minimal size
        if (im_w <= im_h and im_w == size) or (im_h <= im_w
                                               and im_h == size):
            return clip
        new_h, new_w = get_resize_sizes(im_h, im_w, size)
        size = (new_w, new_h)
    else:
        size = size[0], size[1]
    if interpolation == 'bilinear':
        np_inter = cv2.INTER_LINEAR
    else:
        np_inter = cv2.INTER_NEAREST
    scaled = [
        cv2.resize(img, size, interpolation=np_inter) for img in clip
    ]
    return scaled


def normalize(buffer, mean, std):
    for i in range(3):
        buffer[i] = (buffer[i] - mean[i]) / std[i]
    return buffer

def center_crop(clip, size):
    """
    center_crop
    """
    assert isinstance(clip[0], np.ndarray)
    if isinstance(size, numbers.Number):
        size = (size, size)
    h, w = size
    im_h, im_w, _ = clip[0].shape

    if w > im_w or h > im_h:
        error_msg = (
            'Initial image size should be larger then '
            'cropped size but got cropped sizes : ({w}, {h}) while '
            'initial image is ({im_w}, {im_h})'.format(
                im_w=im_w, im_h=im_h, w=w, h=h))
        raise ValueError(error_msg)

    x1 = int(round((im_w - w) / 2.))
    y1 = int(round((im_h - h) / 2.))
    cropped = [img[y1:y1 + h, x1:x1 + w, :] for img in clip]

    return cropped


def loadvideo_decord(sample, sample_rate_scale=1):
    '''
    loadvideo_decord
    '''
    frames = sorted([os.path.join(sample, img) for img in os.listdir(sample)])
    frame_count = len(frames)
    frame_list = np.empty((frame_count, 128, 171, 3), np.dtype('float32'))
    for i, frame_name in enumerate(frames):
        frame = np.array(cv2.imread(frame_name)).astype(np.float64)
        frame_list[i] = frame

    # handle temporal segments
    frame_sample_rate = 2
    converted_len = 16 * frame_sample_rate
    seg_len = frame_count

    all_index = []
    if seg_len <= converted_len:
        index = np.linspace(0, seg_len, num=seg_len // frame_sample_rate)
        index = np.concatenate((index, np.ones(16 - seg_len // frame_sample_rate) * seg_len))
        index = np.clip(index, 0, seg_len - 1).astype(np.int64)
    else:
        end_idx = np.random.randint(converted_len, seg_len)
        str_idx = end_idx - converted_len
        index = np.linspace(str_idx, end_idx, num=16)
        index = np.clip(index, str_idx, end_idx - 1).astype(np.int64)
    all_index.extend(list(index))

    all_index = all_index[::int(sample_rate_scale)]
    buffer = frame_list[all_index]
    return buffer


ucf101_label_names = ['ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling', 'BalanceBeam', \
                      'BandMarching', 'BaseballPitch', 'Basketball', 'BasketballDunk', 'BenchPress', \
                      'Biking', 'Billiards', 'BlowDryHair', 'BlowingCandles', 'BodyWeightSquats', \
                      'Bowling', 'BoxingPunchingBag', 'BoxingSpeedBag', 'BreastStroke', 'BrushingTeeth', \
                      'CleanAndJerk', 'CliffDiving', 'CricketBowling', 'CricketShot', 'CuttingInKitchen', \
                      'Diving', 'Drumming', 'Fencing', 'FieldHockeyPenalty', 'FloorGymnastics', \
                      'FrisbeeCatch', 'FrontCrawl', 'GolfSwing', 'Haircut', 'HammerThrow', \
                      'Hammering', 'HandstandPushups', 'HandstandWalking', 'HeadMassage', 'HighJump', \
                      'HorseRace', 'HorseRiding', 'HulaHoop', 'IceDancing', 'JavelinThrow', \
                      'JugglingBalls', 'JumpRope', 'JumpingJack', 'Kayaking', 'Knitting', \
                      'LongJump', 'Lunges', 'MilitaryParade', 'Mixing', 'MoppingFloor', \
                      'Nunchucks', 'ParallelBars', 'PizzaTossing', 'PlayingCello', 'PlayingDaf', \
                      'PlayingDhol', 'PlayingFlute', 'PlayingGuitar', 'PlayingPiano', 'PlayingSitar', \
                      'PlayingTabla', 'PlayingViolin', 'PoleVault', 'PommelHorse', 'PullUps', \
                      'Punch', 'PushUps', 'Rafting', 'RockClimbingIndoor', 'RopeClimbing', \
                      'Rowing', 'SalsaSpin', 'ShavingBeard', 'Shotput', 'SkateBoarding', \
                      'Skiing', 'Skijet', 'SkyDiving', 'SoccerJuggling', 'SoccerPenalty', \
                      'StillRings', 'SumoWrestling', 'Surfing', 'Swing', 'TableTennisShot', \
                      'TaiChi', 'TennisSwing', 'ThrowDiscus', 'TrampolineJumping', 'Typing', \
                      'UnevenBars', 'VolleyballSpiking', 'WalkingWithDog', 'WallPushups', 'WritingOnBoard', \
                      'YoYo']

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

    stream_name = b'r2p1d'
    infer_total_time = 0

    all_file_num = 0
    predict_num = 0
    for index, category in enumerate(sorted(os.listdir(os.path.join(args.dataset_root_path, "val")))):
        print("Category: ", category)
        for video in sorted(os.listdir(os.path.join(args.dataset_root_path, "val", category))):
            buffer = loadvideo_decord(os.path.join(args.dataset_root_path, "val", category, video))
            buffer = resize_clip(buffer, 128, "bilinear")
            buffer = center_crop(buffer, size=(112, 112))

            buffer = np.array(buffer).transpose((3, 0, 1, 2)) / 255.0
            buffer = normalize(buffer, mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
            buffer = buffer.astype(np.float32)
            buffer = np.expand_dims(buffer, 0)#NCTHW
            if not send_source_data(0, buffer, stream_name, stream_manager_api):
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

            y_predict = res.reshape(101)
            y_predict = y_predict.reshape(101)
            y_predict = np.exp(y_predict) / np.sum(np.exp(y_predict), axis=0)
            predict = np.argmax(y_predict)
            print("The predicted category of:", video, "--->", ucf101_label_names[predict])
            predict_num += 1 if predict == index else 0
            all_file_num += 1

    print("Accuracy:", predict_num/all_file_num)

    # destroy streams
    stream_manager_api.DestroyAllStreams()

if __name__ == '__main__':
    np.random.seed(1)
    main()
