/*
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#include <fstream>
#include <string>
#include <vector>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include "Deepsort.h"
#include "opencv2/opencv.hpp"

#include "./KalmanFilter/tracker.h"

#include "MxBase/Log/Log.h"

const float confThreshold = 0.0;  // Confidence threshold
const float nmsThreshold = 1.0;  // Non-maximum suppression threshold

const int nn_budget = 100;
const float max_cosine_distance = 0.2;

void postprocess(const uint32_t &frame, std::vector<std::vector<float>> &dataset, DETECTIONS &d);
void WriteResult(const std::vector<float> &outputs, const std::string &result_path);
void dp_track(const std::string &result_path);


APP_ERROR GetFileNames(const std::string& path, std::vector<std::string>& filenames) {
    DIR *pDir;
    struct dirent* ptr;
    if (!(pDir = opendir(path.c_str())))
        return APP_ERR_OK;
    while ((ptr = readdir(pDir)) != 0) {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0)
            filenames.push_back(path + "/" + ptr->d_name);
    }
    closedir(pDir);
    return APP_ERR_OK;
}


APP_ERROR ReadDet(const std::string &path, std::vector<std::vector<float>> &dataset) {
    std::ifstream fp(path);
    std::string line;
    const int len_det = 10;
    while (std::getline(fp, line)) {
        std::vector<float> data_line;
        std::string number;
        std::istringstream readstr(line);
        for (int j = 0; j < len_det; j++) {
            std::getline(readstr, number, ',');

            data_line.push_back(atof(number.c_str()));
        }
        dataset.push_back(data_line);
    }
    return APP_ERR_OK;
}


APP_ERROR ReadTXT(std::string pathname, std::vector<std::vector<float>>& res) {
    std::ifstream infile;
    infile.open(pathname.data());
    assert(infile.is_open());
    std::vector<float> suanz;
    std::string s;
    while (std::getline(infile, s)) {
        std::istringstream is(s);
        float d;
        while (!is.eof()) {
            is >> d;
            suanz.push_back(d);
        }
        res.push_back(suanz);
        suanz.clear();
        s.clear();
    }
    infile.close();
    return APP_ERR_OK;
}


APP_ERROR ReadFilesFromPath(const std::string &path, std::vector<std::string> &files) {
    DIR *dir = NULL;
    struct dirent *ptr = NULL;

    if ((dir=opendir(path.c_str())) == NULL) {
        LogError << "Open dir error: " << path;
        return APP_ERR_COMM_OPEN_FAIL;
    }

    while ((ptr=readdir(dir)) != NULL) {
        if (ptr->d_type == 8) {
            files.push_back(ptr->d_name);
        }
    }
    closedir(dir);
    return APP_ERR_OK;
}


int main(int argc, char* argv[]) {
    if (argc <= 2) {
        LogWarn << "Please input image path and det path, such as '../data/data/image ../../../data/det'";
        return APP_ERR_OK;
    }

    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.checkTensor = true;
    initParam.modelPath = "../data/model/deepsort.om";
    auto deepsort = std::make_shared<DEEPSORT>();
    APP_ERROR ret = deepsort->Init(initParam);
    if (ret != APP_ERR_OK) {
        deepsort->DeInit();
        LogError << "deepsort init failed, ret=" << ret << ".";
        return ret;
    }

    std::string imgPath = argv[1];
    std::string detPath = argv[2];

    std::vector<std::string> image_files;
    std::vector<std::string> det_files;

    ret = ReadFilesFromPath(imgPath, image_files);
    if (ret != APP_ERR_OK) {
        deepsort->DeInit();
        LogError << "read image path failed, ret=" << ret << ".";
        return ret;
    }

    ret = GetFileNames(detPath, det_files);
    std::sort(det_files.begin(), det_files.end());
    std::vector<std::vector<float>> det;
    std::vector<uint32_t> length;

    for (uint32_t i = 0; i < det_files.size(); i++) {
        ret = ReadDet(det_files[i] + "/det/det.txt", det);
        if (ret != APP_ERR_OK) {
            LogError << "read det file failed, ret=" << ret << ".";
            deepsort->DeInit();
            return ret;
        }
        length.push_back(det.size());
    }

    if (ret != APP_ERR_OK) {
        deepsort->DeInit();
        LogError << "read det path failed, ret=" << ret << ".";
        return ret;
    }
    // create result directory
    std::string result_path = "../detection/";
    if (access(result_path.c_str(), 0) == -1) {
        int flag = mkdir(result_path.c_str(), S_IRWXO);
        if (flag == 0) {
            std::cout << "Create directory successfully." << std::endl;
        } else {
            std::cout << "Fail to create directory." << std::endl;
            throw std::exception();
        }
    } else {
        std::cout << "This directory already exists." << std::endl;
    }

    int idx = 0;
    for (uint32_t i = 0; i < image_files.size(); i++) {
        std::string image_name = imgPath + "/" + image_files[i].substr(0, 18) + std::to_string(i) + ".bin";
        if (i+1 > length[idx]) {
            ++idx;
        }
        std::string result_name = result_path + det_files[idx].substr(det_files[idx].size()-8, 8) + ".txt";
        ret = deepsort->Process(image_name, det[i], result_name);
        if (ret != APP_ERR_OK) {
            LogError << "deepsort process failed, ret=" << ret << ".";
            deepsort->DeInit();
            return ret;
        }
    }
    deepsort->DeInit();

    dp_track(result_path);
    return APP_ERR_OK;
}

void dp_track(const std::string &result_path) {
    // The part of tracking
    std::string output_path = "../result/";

    // tracker mytracker(max_cosine_distance, nn_budget);

    if (access(output_path.c_str(), 0) == -1) {
        int flag = mkdir(output_path.c_str(), S_IRWXO);
        if (flag == 0) {
            std::cout << "Create directory successfully." << std::endl;
        } else {
            std::cout << "Fail to create directory." << std::endl;
            throw std::exception();
        }
    } else {
        rmdir(output_path.c_str());
        std::cout << "This directory already exists." << std::endl;
        int flag = mkdir(output_path.c_str(), S_IRWXO);
        if (flag == 0) {
            std::cout << "Create directory successfully." << std::endl;
        } else {
            std::cout << "Fail to create directory." << std::endl;
            throw std::exception();
        }
    }
    std::vector<std::string> detection_files;
    auto ret = ReadFilesFromPath(result_path, detection_files);
    if (ret != APP_ERR_OK) {
        LogError << "read detection files failed, ret=" << ret << ".";
        return;
    }

    for (auto sequence : detection_files) {
        std::cout << sequence << std::endl;
        tracker mytracker(max_cosine_distance, nn_budget);
        std::vector<std::vector<float>> detections;
        ret = ReadTXT(result_path + sequence, detections);

        if (ret != APP_ERR_OK) {
            LogError << "read detection file failed, ret=" << ret << ".";
            return;
        }

        uint32_t frame_min = detections[0][0], frame_max = detections[detections.size()-1][0];
        for (uint32_t frame = frame_min; frame <= frame_max; frame++) {
            std::cout << "Processing frame:" << frame << std::endl;
            DETECTIONS detection;
            postprocess(frame, detections, detection);
            mytracker.predict();
            mytracker.update(detection);

            for (Track& track : mytracker.tracks) {
                if (!track.is_confirmed() || track.time_since_update > 1) {
                    continue;
                }
                std::vector<float> res;
                res.push_back(frame);
                res.push_back(track.track_id);
                res.push_back(track.to_tlwh()[0]);
                res.push_back(track.to_tlwh()[1]);
                res.push_back(track.to_tlwh()[2]);
                res.push_back(track.to_tlwh()[3]);
                WriteResult(res, output_path+sequence);
            }
        }
    }
}

void postprocess(const uint32_t &frame, std::vector<std::vector<float>> &dataset, DETECTIONS &d) {
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<FEATURE> features;
    while (dataset.size() > 1 && dataset[0][0] == frame) {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float confidence = dataset[0][6];

        if (confidence > confThreshold) {
            FEATURE feature;
            int left = static_cast<int>(dataset[0][2]);
            int top = static_cast<int>(dataset[0][3]);
            int width = static_cast<int>(dataset[0][4]);
            int height = static_cast<int>(dataset[0][5]);
            if (height < 0) {
                continue;
            }
            confidences.push_back(confidence);

            boxes.push_back(cv::Rect(left, top, width, height));
            int idx = 0;
            for (size_t i = 10; i < dataset[0].size(); ++i) {
                feature[idx++] = dataset[0][i];
            }
            features.push_back(feature);
        }
        dataset.erase(dataset.begin());
    }

    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    std::vector<int> indices;

    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

    for (size_t i = 0; i < indices.size(); ++i) {
        size_t idx = static_cast<size_t>(indices[i]);
        cv::Rect box = boxes[idx];
        FEATURE f = features[i];
        DETECTION_ROW tmpRow;
        tmpRow.tlwh = DETECTBOX(box.x, box.y, box.width, box.height);  // DETECTBOX(x, y, w, h);
        tmpRow.confidence = confidences[idx];
        tmpRow.feature = f;
        d.push_back(tmpRow);
    }
}


void WriteResult(const std::vector<float> &outputs, const std::string &result_path) {
    std::ofstream outfile(result_path, std::ios::app);
    if (outfile.fail()) {
        LogError << "Failed to open result file: ";
    }

    std::string tmp = "";
    tmp += std::to_string(static_cast<int>(outputs[0])) + ",";
    tmp += std::to_string(static_cast<int>(outputs[1])) + ",";
    for (uint32_t i = 2; i < outputs.size(); ++i) {
        auto str = std::to_string(static_cast<int>(outputs[i]*100+0.5)/100.0);
        tmp += str.substr(0, str.find(".") + 3) + ",";
    }
    tmp += "1,-1,-1,-1";

    outfile << tmp << std::endl;
    outfile.close();
}
