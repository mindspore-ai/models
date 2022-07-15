/**
 * Copyright 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "Attgan.h"
#include <string>
#include <typeinfo>

namespace {
    std::vector<std::string> SELECTED_ATTRS {"Bald", "Bangs", "Black_Hair", "Blond_Hair", "Brown_Hair",
    "Bushy_Eyebrows", "Eyeglasses", "Male", "Mouth_Slightly_Open", "Mustache", "No_Beard", "Pale_Skin", "Young"};
    const std::string OM_MODEL_PATH = "../data/model/attgan.om";
}

APP_ERROR ScanImages(const std::string &path, std::vector<std::string> *imgFiles) {
    DIR *dirPtr = opendir(path.c_str());
    if (dirPtr == nullptr) {
        LogError << "opendir failed. dir:" << path;
        return APP_ERR_INTERNAL_ERROR;
    }
    dirent *direntPtr = nullptr;
    while ((direntPtr = readdir(dirPtr)) != nullptr) {
        std::string fileName = direntPtr->d_name;
        if (fileName == "." || fileName == "..") {
            continue;
        }

        imgFiles->emplace_back(path + "/" + fileName);
    }
    closedir(dirPtr);
    return APP_ERR_OK;
}

std::vector<std::string> split(std::string str, char ch) {
    size_t start = 0;
    size_t len = 0;
    std::vector<std::string> ret;
    for (size_t i = 0; i < str.length(); i++) {
        if (str[i] == ch && i+1 < str.length() && str[i+1] == ch) {
            continue;
        }
        if (str[i] == ch) {
            ret.push_back(str.substr(start, len));
            start = i+1;
            len = 0;
        } else {
            len++;
        }
    }
    if (start < str.length())
        ret.push_back(str.substr(start, len));
    return ret;
}

void _set(int att_[1][13], const std::string &att_name) {
    std::string att_text_0[] = {"Bald", "Bangs", "Black_Hair", "Blond_Hair", "Brown_Hair", "Bushy_Eyebrows",
    "Eyeglasses", "Male", "Mouth_Slightly_Open", "Mustache", "No_Beard", "Pale_Skin", "Young"};
    for (int j = 0; j < 13; j++) {
    if (att_text_0[j] == att_name) {
        att_[0][j] = 0;
    }
    }
}

void check_attribute_1(int att_batch[1][13], const std::string &att_name, const int attr_list) {
    if ((att_name == "Straight_Hair" || att_name == "Wavy_Hair") && (att_batch[0][attr_list] != 0)) {
        std::string att_text_2[] = {"Straight_Hair", "Wavy_Hair"};
        for (int j = 0; j < 2; j++) {
            std::string n;
            n = att_text_2[j];
            if (n != att_name) {
                _set(att_batch, n);
            }
        }
    } else if ((att_name == "Mustache" || att_name == "No_Beard") && (att_batch[0][attr_list] != 0)) {
        std::string att_text_3[] = {"Mustache", "No_Beard"};
        for (int j = 0; j < 2; j++) {
            std::string n;
            n = att_text_3[j];
            if (n != att_name) {
                _set(att_batch, n);
            }
        }
    }
}


void check_attribute(int att_batch[1][13], const std::string &att_name, const int attr_list) {
    if ((att_name == "Bald" || att_name == "Receding_Hairline") && (att_batch[0][attr_list] != 0)) {
        _set(att_batch, "Bangs");
    } else if ((att_name == "Bangs") && (att_batch[0][attr_list] != 0)) {
        _set(att_batch, "Bald");
        _set(att_batch, "Receding_Hairline");
    } else if ((att_name == "Black_Hair" || att_name == "Blond_Hair" || att_name == "Brown_Hair" ||
        att_name == "Gray_Hair") && (att_batch[0][attr_list] != 0)) {
        std::string att_text_1[] = {"Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair"};
        for (int j = 0; j < 4; j++) {
            std::string n;
            n = att_text_1[j];
            if (n != att_name) {
                _set(att_batch, n);
            }
        }
    }
}

void labal_process(int tmp_2[1][14][13], int label[13]) {
    int attr_num;
    attr_num = SELECTED_ATTRS.size();
    int tmp[1][13];
    int tmp_0[1][13];
    int tmp_1[1][13];
    for (int i = 0; i < attr_num; i++) {
        tmp[0][i] = label[i];
        tmp_0[0][i] =  tmp[0][i];
        tmp_2[0][0][i] = tmp_0[0][i];
    }
    for (int i = 0; i < attr_num; ++i) {
        int attr_length = i;
        tmp_0[0][i] = 1 - tmp[0][i];
        for (int k = 0; k < 13; k++) {
            tmp_1[0][k] = tmp_0[0][k];
        }
        check_attribute(tmp_1, SELECTED_ATTRS[i], attr_length);
        check_attribute_1(tmp_1, SELECTED_ATTRS[i], attr_length);
        for (int j = 0; j < 13; j++) {
            tmp_2[0][i+1][j] = tmp_1[0][j];
        }
        tmp_0[0][i] = tmp[0][i];
    }
}

int main(int argc, char* argv[]) {
    if (argc <= 1) {
        LogWarn << "Please input image path, such as '../data/test_data/'.";
        return APP_ERR_OK;
    }

    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.modelPath = OM_MODEL_PATH;
    initParam.savePath = "../data/mxbase_result";
    auto attgan = std::make_shared<Attgan>();
    APP_ERROR ret = attgan->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "Attgan init failed, ret=" << ret << ".";
        return ret;
    }

     // Read the contents of a label
    std::string dataPath = argv[1];
    std::string imagePath = dataPath + "/image/";
    std::string labelPath = dataPath + "/list_attr_celeba.txt";

    std::vector<std::string> imagePathList;
    ret = ScanImages(imagePath, &imagePathList);
    if (ret != APP_ERR_OK) {
        LogError << "Attgan init failed, ret=" << ret << ".";
        return ret;
    }
    std::ifstream fin;
    std::string s;
    fin.open(labelPath);
    int i = 0;
    int imgNum;
    std::map<int, std::string> idx2attr;
    std::map<std::string, int> attr2idx;
    auto startTime = std::chrono::high_resolution_clock::now();

    while (getline(fin, s)) {
        i++;
        if (i == 1) {
            imgNum = atoi(s.c_str());
        } else if (i == 2) {
            std::vector<std::string> allAttrNames = split(s, ' ');
            for (size_t j = 0; j < allAttrNames.size(); j++) {
                idx2attr[j] = allAttrNames[j];
                attr2idx[allAttrNames[j]] = j;
                SELECTED_ATTRS[12] = allAttrNames[39];
            }
        } else {
            std::vector<std::string> eachAttr = split(s, ' ');
            // first one is file name
            std::string imgName = eachAttr[0];
            std::vector<float> label;
            std::vector<float> label_last;
            int label_2[13];
            int tmp_last[1][14][13];
            float att_b_[1][14][13];
            for (size_t j = 0; j < SELECTED_ATTRS.size(); j++) {
                  label.push_back(static_cast<int>(ceil(atoi
                  (eachAttr[attr2idx[SELECTED_ATTRS[j]] + 1 ].c_str())+1)) * 0.5);
            }
            for (int k = 0; k < 13; k ++) {
                label_2[k] = label[k];
            }
            labal_process(tmp_last, label_2);
            for (int a = 0; a < 14; a++) {
                for (int j = 0; j < 13; j++) {
                    att_b_[0][a][j] = (static_cast<float>(tmp_last[0][a][j]) * 2.0 - 1.0) * 0.5;
                    if (a > 0 && j == a-1) {
                    att_b_[0][a][a-1] = (att_b_[0][a][a-1] * 1.0) / 0.5;
                    }
                    }
            }
            for (int b = 0; b < 14; b++) {
                for (int j = 0; j < 13; j++) {
                    label_last.push_back(att_b_[0][b][j]);
                }
                ret = attgan->Process(imagePath + imgName, imgName, label_last, b);
                if (ret != APP_ERR_OK) {
                    LogError << "Attgan process failed, ret=" << ret << ".";
                    attgan->DeInit();
                    return ret;
                }
                label_last.clear();
            }
        }
    }
    fin.close();
    auto endTime = std::chrono::high_resolution_clock::now();
    attgan->DeInit();
    double costMilliSecs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    double fps = 1000.0 * imgNum / attgan->GetInferCostMilliSec();
    LogInfo << "[Process Delay] cost: " << costMilliSecs << " ms\tfps: " << fps << " imgs/sec";
    return APP_ERR_OK;
}
