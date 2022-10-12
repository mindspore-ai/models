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

#include <iostream>
#include <opencv2/dnn/dnn.hpp>

#include "PSPNet.h"
#include "MxBase/Log/Log.h"

using std::vector;
using std::string;
using std::cout;
using std::endl;
using std::min;


std::vector<std::string> SplitLine(const std::string & str, const char* delim) {
    std::vector<std::string> res;
    if ("" == str) {
        res.push_back(str);
        return res;
    }

    char* p_strs = new char[str.length() + 1];
    char* p_save = NULL;
    strncpy(p_strs, str.c_str(), str.length());

    char* part = strtok_r(p_strs, delim, &p_save);
    while (part) {
        std::string s = part;
        res.push_back(s);
        part = strtok_r(NULL, delim, &p_save);
    }
    return res;
}

std::vector<std::string> GetAllFiles(const std::string & root_path, const std::string & data_path) {
    std::ifstream ifs;
    std::vector<std::string> files;

    ifs.open(data_path, std::ios::in);
    if (!ifs.is_open()) {
        std::cout << "File: " << data_path << " is not exist" << std::endl;
        return files;
    }
    std::string buf;
    while (getline(ifs, buf)) {
        std::vector<std::string> line = SplitLine(buf, " ");
        std::string img_path = line[0];
        std::string msk_path = line[1];

        files.emplace_back(img_path);
        files.emplace_back(img_path);
    }
    ifs.close();
    return files;
}

void SaveResult(const cv::Mat& binImg, const std::string& res_dir, const std::string & file_name) {
    cv::Mat imageGrayC3 = cv::Mat::zeros(binImg.rows, binImg.cols, CV_8UC3);
    std::vector<cv::Mat> planes;
    for (int i = 0; i < 3; i++) {
        planes.push_back(binImg);
    }
    cv::merge(planes, imageGrayC3);
    uchar rgbColorMap[256*3] = {
        0, 0, 0,
        128, 0, 0,
        0, 128, 0,
        128, 128, 0,
        0, 0, 128,
        128, 0, 128,
        0, 128, 128,
        128, 128, 128,
        64, 0, 0,
        192, 0, 0,
        64, 128, 0,
        192, 128, 0,
        64, 0, 128,
        192, 0, 128,
        64, 128, 128,
        192, 128, 128,
        0, 64, 0,
        128, 64, 0,
        0, 192, 0,
        128, 192, 0,
        0, 64, 128,
    };
    cv::Mat lut(1, 256, CV_8UC3, rgbColorMap);

    cv::Mat imageColor;
    cv::LUT(imageGrayC3, lut, imageColor);
    cv::cvtColor(imageColor, imageColor, cv::COLOR_RGB2BGR);

    std::string gray_path = res_dir + "gray";
    std::string color_path = res_dir + "color";

    std::string command = "mkdir -p " + gray_path;
    system(command.c_str());
    command = "mkdir -p " + color_path;
    system(command.c_str());

    std::cout << "save to " << gray_path << std::endl;
    std::cout << "save to " << color_path << std::endl;
    cv::imwrite(color_path + file_name, imageColor);
    cv::imwrite(gray_path + file_name, binImg);
}

void ArgMax(const cv::Mat& Tensor, const cv::Mat& Res) {
    uchar* pDst = Res.data;
    float* tensordata = reinterpret_cast<float * >(Tensor.data);
    int high = Tensor.rows;
    int width = Tensor.cols;
    int classes = Tensor.channels();

    for (int i = 0; i < width; i++) {
        for (int j = 0; j < high; j++) {
            float max = 0;
            uint8_t index = 0;
            for (int k = 0; k < classes; k++) {
                float res = *(tensordata + i*high*classes + j * 21 + k);
                if (res > max) {
                    max = res;
                    index = k;
                }
            }
            uint8_t gray = index;
            *(pDst + i*high + j) = gray;
        }
    }
}

cv::Mat ScaleProcess(cv::Mat image,
                     int classes,
                     int crop_h,
                     int crop_w,
                     int ori_h,
                     int ori_w,
                     float stride_rate,
                     bool flip,
                     PSPNet& pspnet) {
    int ori_h1 = image.rows;
    int ori_w1 = image.cols;

    int pad_h = (crop_h - ori_h1) > 0 ?  (crop_h - ori_h1) : 0;
    int pad_w = (crop_w - ori_w1) > 0 ?  (crop_w - ori_w1) : 0;

    int pad_h_half = static_cast<int>(pad_h / 2);
    int pad_w_half = static_cast<int>(pad_w / 2);
    cv::Scalar mean_value(0.485 * 255, 0.456 * 255, 0.406 * 255);
    vector<double> std_value = {0.229 * 255, 0.224 * 255, 0.225 * 255};

    cv::Mat padded_img;
    padded_img.convertTo(padded_img, CV_32FC3);

    if (pad_h > 0 || pad_w > 0) {
        cv::copyMakeBorder(image,
                           padded_img,
                           pad_h_half,
                           pad_h - pad_h_half,
                           pad_w_half,
                           pad_w - pad_w_half,
                           cv::BORDER_CONSTANT,
                           mean_value);
    } else {
        padded_img = image;
    }

    int new_h = padded_img.rows;
    int new_w = padded_img.cols;

    int stride_h = ceil(static_cast<float>(crop_h * stride_rate));
    int stride_w = ceil(static_cast<float>(crop_w * stride_rate));
    int grid_h = static_cast<int>(ceil(static_cast<float>(new_h - crop_h) / stride_h) + 1);
    int grid_w = static_cast<int>(ceil(static_cast<float>(new_w - crop_w) / stride_w) + 1);

    cv::Mat count_crop = cv::Mat::zeros(new_h, new_w, CV_32FC1);
    cv::Mat prediction = cv::Mat::zeros(new_h, new_w, CV_32FC(classes));

    for (int index_h = 0; index_h < grid_h; index_h++) {
        for (int index_w = 0; index_w < grid_w; index_w++) {
            int start_x = min(index_w * stride_w + crop_w, new_w) - crop_w;
            int start_y = min(index_h * stride_h + crop_h, new_h) - crop_h;

            cv::Mat crop_roi(count_crop, cv::Rect(start_x, start_y, crop_w, crop_h));
            crop_roi += 1;  // area infer count

            cv::Mat prediction_roi(prediction, cv::Rect(start_x, start_y, crop_w, crop_h));

            cv::Mat image_roi = padded_img(cv::Rect(start_x, start_y, crop_w, crop_h)).clone();

            image_roi = image_roi - mean_value;

            std::vector<cv::Mat> rgb_channels(3);
            cv::split(image_roi, rgb_channels);
            for (int i = 0; i < 3; i++) {
                rgb_channels[i].convertTo(rgb_channels[i], CV_32FC1, 1.0 / std_value[i]);
            }
            cv::merge(rgb_channels, image_roi);

            cv::Mat blob = cv::dnn::blobFromImage(image_roi);  // 473 473 3 ---> 3 473 473

            std::vector<MxBase::TensorBase> outputs;
            pspnet.Process(blob, outputs);
            MxBase::TensorBase pred = outputs[0];
            pred.ToHost();
            float* data = reinterpret_cast<float* >(pred.GetBuffer());

            if (flip) {
                cv::Mat flipped_img;
                std::vector<MxBase::TensorBase> flipped_outputs;
                cv::flip(image_roi, flipped_img, 1);
                cv::Mat blob_flip = cv::dnn::blobFromImage(flipped_img);

                pspnet.Process(blob_flip, flipped_outputs);
                MxBase::TensorBase flipped_pred = flipped_outputs[0];
                flipped_pred.ToHost();
                float* flipped_data = reinterpret_cast<float* >(flipped_pred.GetBuffer());
                for (int i = 0; i < crop_h; i++) {
                    for (int j = 0; j < crop_w; j++) {
                        for (int k = 0; k < classes; k ++) {
                            float res = (*(data+k*crop_h*crop_w + i*crop_w + j) +  // data[k][i][j]
                                         *(flipped_data+k*crop_h*crop_w + i*crop_w + 472-j)) / 2;
                            *(data+k*crop_h*crop_w + i*crop_w + j) = res;
                        }
                    }
                }
            }

            for (int i = 0; i < crop_h; i++) {
                for (int j = 0; j < crop_w; j++) {
                    for (int k = 0; k < classes; k ++) {
                        float res = *(data+k*crop_h*crop_w + i*crop_w + j);
                        prediction_roi.ptr<float>(i)[j * classes + k] += res;  // 21 473 473
                    }
                }
            }
        }
    }

    std::vector<cv::Mat> cls_channels(classes);
    cv::split(prediction, cls_channels);
    for (int i = 0; i < classes; i++) {
        cls_channels[i] = cls_channels[i] / count_crop;
    }
    cv::merge(cls_channels, prediction);
    cv::Mat prediction_crop(prediction, cv::Rect(pad_w_half, pad_h_half, ori_w1, ori_h1));

    cv::Mat final_pre;
    cv::resize(prediction_crop, final_pre, cv::Size(ori_w, ori_h), cv::INTER_LINEAR);

    return final_pre;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        LogError << "Please input the om file path and dataset path";
    }

    std::string om_path = argv[1];
    std::string dataset_path = argv[2];

    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.modelPath = om_path;

    PSPNet pspnet;
    APP_ERROR ret = pspnet.Init(initParam);

    if (ret != APP_ERR_OK) {
        LogError << "PSPNet init failed, ret=" << ret << ".";
        return ret;
    }
    cout << "PSPNet Init Done." << endl;

    std::string voc_val_list = dataset_path + "/voc_val_lst.txt";

    int crop_h = 473;
    int crop_w = 473;  // crop image to 473
    int classes = 21;  // number of classes
    float stride_rate = 2.0/3.0;

    cout << "Start to get image" << endl;
    auto all_files = GetAllFiles(dataset_path, voc_val_list);

    if (all_files.empty()) {
        std::cout << "ERROR: no input data." << std::endl;
        return APP_ERR_INVALID_FILE;
    }

    for (int i = 0; i < all_files.size(); i = i + 2) {
        std::string imgPath = all_files[i];
        cout << "Process image : " << imgPath << endl;

        cv::Mat image =  cv::imread(imgPath, cv::IMREAD_COLOR);
        cv::Mat image_RGB;
        cv::cvtColor(image, image_RGB, cv::COLOR_BGR2RGB);

        float ori_h = image.rows;
        float ori_w = image.cols;
        float long_size = 512;  // The longer edge should align to 512

        int new_h = long_size;
        int new_w = long_size;

        if (ori_h > ori_w) {
            new_w = round(long_size / ori_h * ori_w);
        } else {
            new_h = round(long_size / ori_w * ori_h);
        }
        cv::Mat resized_img;
        image_RGB.convertTo(image_RGB, CV_32FC3);
        resized_img.convertTo(resized_img, CV_32FC3);
        cv::resize(image_RGB, resized_img, cv::Size(new_w, new_h), cv::INTER_LINEAR);

        cv::Mat pre = ScaleProcess(resized_img,
                                   classes,
                                   crop_h,
                                   crop_w,
                                   image.rows,
                                   image.cols,
                                   stride_rate, true, pspnet);
        cv::Mat pre_max(pre.rows, pre.cols, CV_8UC1, cv::Scalar(0));
        ArgMax(pre, pre_max);

        size_t pos = imgPath.find_last_of("/");
        std::string file_name(imgPath.begin() + pos, imgPath.end());
        pos = file_name.find_last_of(".");
        file_name.replace(file_name.begin() + pos, imgPath.end(), ".png");

        SaveResult(pre_max, "cpp_res/", file_name);
    }

    pspnet.DeInit();
    return APP_ERR_OK;
}
