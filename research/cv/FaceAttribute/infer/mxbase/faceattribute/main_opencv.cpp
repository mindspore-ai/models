/*
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include "MxBase/Log/Log.h"
#include "FaceAttribute.h"

namespace {
const uint32_t CLASS_NUM_age = 9;
const uint32_t CLASS_NUM_gender = 2;
const uint32_t CLASS_NUM_mask = 2;
}  // namespace


int main(int argc, char* argv[]) {
    if (argc <= 2) {
        LogWarn << "Please input label_txt_path path and modelPath！！！";
        return APP_ERR_OK;
    }

    // Receive path parameters
    std::string label_txt_path = argv[1];
    std::string modelPath = argv[2];

    // Custom parameters can be left alone here, just change the modelpath and you're done.
    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.labelPath = "./resnet18_clsidx_to_labels_age.names";
    initParam.classNum = 9;
    initParam.topk = 1;
    initParam.softmax = false;
    initParam.checkTensor = true;
    initParam.modelPath = modelPath;
    auto resnet50 = std::make_shared<faceattribute>();
    // init
    APP_ERROR ret = resnet50->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "faceAttribute init failed, ret=" << ret << ".";
        return ret;
    }

    // Here define the total number of all attributes to store the total number of test images.
    // Here you need to change the number of attributes yourself.
    int attributeTotalNum[] = {0, 0, 0};
    int true_num[] = {0, 0, 0};

    // Read the contents of a label
    std::ifstream fin;
    std::string s;
    fin.open(label_txt_path);
    while (getline(fin, s)) {
        int pos1;
        int pos2;
        int pos3 = 0;
        for (int i = 0; i < s.size(); i++) {
            if (s[i] == '/') {
                pos1 = i;
            }
            if (s[i] == '.') {
                pos2 = i;
            }
            if (s[i] == ' ') {
                pos3 = i;
                break;
            }
        }
        pos1 = pos1+1;
        pos2 = pos2-1;
        std::string num_string = s.substr(pos1, pos2-pos1+1);
        int num;
        std::stringstream ss;
        ss << num_string;
        ss >> num;

        std::string jpg_path = s.substr(0, pos3);

        // Get three tags
        int pos_sign1 = pos2+6;
        int pos_sign2 = pos_sign1+2;
        int pos_sign3 = pos_sign2+2;
        // Here it's a character, so subtract 48 from 0.
        int sign1 = s[pos_sign1]-48;
        int sign2 = s[pos_sign2]-48;
        int sign3 = s[pos_sign3]-48;

        // Get the ground_truth of the image. Subtract the number of image paths here.
        int ground_truth[3];
        ground_truth[0] = sign1;
        ground_truth[1] = sign2;
        ground_truth[2] = sign3;
        for (int index = 0; index < 3; index++) {
            // If the label is -1 then the total is not counted
            if (ground_truth[index] != -1) {
                attributeTotalNum[index] += 1;
            }
        }

        // Incoming data and ground_truth
        ret = resnet50->Process(jpg_path, ground_truth);
        if (ret != APP_ERR_OK) {
            LogError << "faceattribute process failed, ret=" << ret << ".";
            resnet50->DeInit();
            return ret;
        }

        // Check the element in ground_truth , if the element is 1,
        // it means the predicted value and the label value are the same
        for (int i = 0; i < 3; i++) {
            // +1 for the correct number of each attribute
            if (ground_truth[i] == 1) {
                true_num[i] += 1;
            }
        }
    }
    // close file
    fin.close();
    // close stream
    resnet50->DeInit();
    // Output test results
    int length = 3;
    for (int i = 0; i < length; i++) {
        std::cout << "accucracy is " << static_cast<float>(true_num[i])
        / static_cast<float>(attributeTotalNum[i]) << std::endl;
    }
    return APP_ERR_OK;
}

