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
from os.path import join
import math
from scipy.io import loadmat
import numpy as np
import cv2
import transforms3d


def rotateBody(RzBody, pelvisRotVec):
    angle = np.linalg.norm(pelvisRotVec)
    Rpelvis = transforms3d.axangles.axangle2mat(pelvisRotVec / angle, angle)
    globRotMat = np.dot(RzBody, Rpelvis)
    R90 = transforms3d.euler.euler2mat(np.pi / 2, 0, 0)
    globRotAx, globRotAngle = transforms3d.axangles.mat2axangle(np.dot(R90, globRotMat))
    globRotVec = globRotAx * globRotAngle
    return globRotVec

# Extract SURREAL training dataset
def extract_surreal_train(dataset_path, out_path):
    shapes_, poses_ = [], []
    scales_, centers_, parts_, S_ = [], [], [], []
    genders_ = []
    videonames_ = []
    framenums_ = []

    # bbox expansion factor
    scaleFactor = 1.2
    height = 240
    width = 320

    train_path = join(dataset_path, 'cmu', 'train')
    dirs1 = os.listdir(train_path)
    dirs1.sort()
    for dir1 in dirs1:
        path_tmp1 = join(train_path, dir1)
        dirs2 = os.listdir(path_tmp1)
        dirs2.sort()
        for dir2 in dirs2:
            path_tmp2 = join(path_tmp1, dir2)
            info_files = os.listdir(path_tmp2)
            info_files.sort()
            for info_file in info_files:
                if info_file.endswith('_info.mat'):
                    file_path = join(path_tmp2, info_file)
                    info = loadmat(file_path)
                    seq_len = info['gender'].shape[0]
                    videoname = join(dir1, dir2, info_file.replace('_info.mat', '.mp4'))
                    print(videoname)

                    ind = np.arrange(0, seq_len, 10)
                    # read GT data
                    shape = info['shape'][:, ind]
                    pose = info['pose'][:, ind]
                    part24 = info['joints2D'][:, :, ind].transpose(1, 0)
                    joint3d24 = info['joints3D'][:, :, ind].transpose(1, 0)
                    gender = info['gender'][ind, 0]
                    zrot = info['zrot'][ind, 0]

                    # The video of SURREAL is mirrored, and the 2D joints location are consistent with the video.
                    # In order to get the image consistent with the SMPL parameters,
                    # we need to mirror the video and 2D joints.

                    part24[:, 0] = width - 1 - part24[:, 0]     # Mirror the 2D joints

                    bbox = [min(part24[:, 0]), min(part24[:, 1]),
                            max(part24[:, 0]), max(part24[:, 1])]
                    center = [(bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2]
                    scale = scaleFactor * max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200

                    # Some frames in SURREAL contains no human,
                    # so we need to remove the frames where human is outside the image.
                    if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > width or bbox[3] > height:
                        continue

                    # rotate 3D joint to align with camera
                    RBody = np.array([[0, 0, 1],
                                      [0, 1, 0],
                                      [-1, 0, 0]])
                    joint3d24 = np.dot(RBody, joint3d24.T).T

                    # rotate SMPL to align with camera
                    RzBody = np.array(((math.cos(zrot), -math.sin(zrot), 0),
                                       (math.sin(zrot), math.cos(zrot), 0),
                                       (0, 0, 1)))
                    pose[0:3] = rotateBody(RzBody, pose[0:3])
                    aa = pose[:3]
                    per_rdg, _ = cv2.Rodrigues(aa)
                    resrot, _ = cv2.Rodrigues(np.dot(RBody, per_rdg))
                    aa_new = (resrot.T)[0]
                    pose[:3] = aa_new

                    # store data
                    part = np.ones([24, 3])
                    part[:, :-1] = part24
                    S = np.ones([24, 4])
                    S[:, :-1] = joint3d24
                    videonames_.append(videoname)
                    framenums_.append(ind)
                    genders_.append(gender)
                    centers_.append(center)
                    scales_.append(scale)
                    parts_.append(part)
                    shapes_.append(shape)
                    poses_.append(pose)
                    S_.append(S)

    # # store the data struct
    # if not os.path.isdir(out_path):
    os.makedirs(out_path)
    out_file = os.path.join(out_path, 'surreal_train.npz')
    np.savez(out_file,
             gender=genders_,
             videoname=videonames_,
             framenum=framenums_,
             center=centers_,
             scale=scales_,
             pose=poses_,
             shape=shapes_,
             part_smpl=parts_,
             S_smpl=S_)

# Extract the val dataset of SURREAL.
def extract_surreal_eval(dataset_path, out_path):

    eval_names = []
    with open(join(dataset_path, 'namescmu.txt'), 'r') as f:
        for line in f:
            tmp = line.split('val/')[1]
            tmp = tmp.split('\t')[0]
            eval_names.append(tmp)

    # Some val images contain no human body, so we only use the
    # meaningful val images as BodyNet.
    with open(join(dataset_path, 'valid_list.txt'), 'r') as f:
        valid_list = f.readline()
        valid_list = valid_list.split('[')[1].split(']')[0]

    valid_list = valid_list[1:-1].split(',')
    valid_list = [int(ind) - 1 for ind in valid_list]
    valid_eval_names = [eval_names[tmp] for tmp in valid_list]

    shapes_, poses_ = [], []
    scales_, centers_, parts_, S_ = [], [], [], []
    genders_ = []
    videonames_ = []
    framenums_ = []

    # bbox expansion factor
    scaleFactor = 1.2
    width = 320

    val_path = join(dataset_path, 'cmu', 'val')
    for videoname in valid_eval_names:
        info_file = videoname[:-4] + '_info.mat'
        file_path = join(val_path, info_file)
        info = loadmat(file_path)
        seq_len = info['gender'].shape[0]
        print(videoname)

        if seq_len < 2:     # ignore the video with only 1 frame.
            continue

        ind = seq_len // 2  # choose the middle frame

        # read GT data
        shape = info['shape'][:, ind]
        pose = info['pose'][:, ind]
        part24 = info['joints2D'][:, :, ind].transpose(1, 0)
        joint3d24 = info['joints3D'][:, :, ind].transpose(1, 0)
        gender = info['gender'][ind]
        gender = 'f' if gender == 0 else 'm'  # 0: female; 1: male
        zrot = info['zrot'][ind, 0]

        # The video of SURREAL is mirrored, and the 2D joints location are consistent with the video.
        # In order to get the image consistent with the SMPL parameters,
        # we need to mirror the video and 2D joints.

        part24[:, 0] = width - 1 - part24[:, 0]  # Mirror the 2D joints

        bbox = [min(part24[:, 0]), min(part24[:, 1]),
                max(part24[:, 0]), max(part24[:, 1])]
        center = [(bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2]
        scale = scaleFactor * max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200

        # Some frames in SURREAL contains no human,
        # so we need to remove the frames where human is outside the image.
        if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > width or bbox[3] > height:
            continue

        # rotate 3D joint to align with camera
        RBody = np.array([[0, 0, 1],
                          [0, 1, 0],
                          [-1, 0, 0]])
        joint3d24 = np.dot(RBody, joint3d24.T).T

        # rotate SMPL to align with camera
        RzBody = np.array(((math.cos(zrot), -math.sin(zrot), 0),
                           (math.sin(zrot), math.cos(zrot), 0),
                           (0, 0, 1)))
        pose[0:3] = rotateBody(RzBody, pose[0:3])
        aa = pose[:3]
        per_rdg, _ = cv2.Rodrigues(aa)
        resrot, _ = cv2.Rodrigues(np.dot(RBody, per_rdg))
        aa_new = (resrot.T)[0]
        pose[:3] = aa_new

        # store data
        part = np.ones([24, 3])
        part[:, :-1] = part24
        S = np.ones([24, 4])
        S[:, :-1] = joint3d24
        videonames_.append(videoname)
        framenums_.append(ind)
        genders_.append(gender)
        centers_.append(center)
        scales_.append(scale)
        parts_.append(part)
        shapes_.append(shape)
        poses_.append(pose)
        S_.append(S)

    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, 'surreal_val.npz')

    np.savez(out_file,
             gender=genders_,
             videoname=videonames_,
             framenum=framenums_,
             center=centers_,
             scale=scales_,
             part_smpl=parts_,
             pose=poses_,
             shape=shapes_,
             S_smpl=S_)
