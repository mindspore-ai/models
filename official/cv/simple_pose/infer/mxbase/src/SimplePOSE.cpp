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
#include "SimplePOSE.h"
#include <string>
#include <cstdlib>
#include <sstream>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include "acl/acl.h"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"

#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/stringbuffer.h"



namespace {
    const uint32_t MODEL_HEIGHT = 256;
    const uint32_t MODEL_WIDTH = 192;
    const int NPOINTS = 17;
}  // namespace

void PrintTensorShape(const std::vector<MxBase::TensorDesc> &tensorDescVec, const std::string &tensorName) {
    LogInfo << "The shape of " << tensorName << " is as follows:";
    for (size_t i = 0; i < tensorDescVec.size(); ++i) {
        LogInfo << "  Tensor " << i << ":";
        for (size_t j = 0; j < tensorDescVec[i].tensorDims.size(); ++j) {
            LogInfo << "   dim: " << j << ": " << tensorDescVec[i].tensorDims[j];
        }
    }
}

APP_ERROR SimplePOSE::Init(const InitParam &initParam) {
    deviceId_ = initParam.deviceId;
    APP_ERROR ret = MxBase::DeviceManager::GetInstance()->InitDevices();
    if (ret != APP_ERR_OK) {
        LogError << "Init devices failed, ret=" << ret << ".";
        return ret;
    }
    ret = MxBase::TensorContext::GetInstance()->SetContext(initParam.deviceId);
    if (ret != APP_ERR_OK) {
        LogError << "Set context failed, ret=" << ret << ".";
        return ret;
    }
    dvppWrapper_ = std::make_shared<MxBase::DvppWrapper>();
    ret = dvppWrapper_->Init();
    if (ret != APP_ERR_OK) {
        LogError << "DvppWrapper init failed, ret=" << ret << ".";
        return ret;
    }
    model_ = std::make_shared<MxBase::ModelInferenceProcessor>();
    ret = model_->Init(initParam.modelPath, modelDesc_);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInferenceProcessor init failed, ret=" << ret << ".";
        return ret;
    }

    PrintTensorShape(modelDesc_.inputTensors, "Model Input Tensors");
    PrintTensorShape(modelDesc_.outputTensors, "Model Output Tensors");

    post_ = std::make_shared<MxBase::SimplePOSEMindsporePost>();
    return APP_ERR_OK;
}

APP_ERROR SimplePOSE::DeInit() {
    dvppWrapper_->DeInit();
    model_->DeInit();
    post_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}


APP_ERROR SimplePOSE::ReadImage(const std::string& imgPath, cv::Mat *imageMat, ImageShape *imgShape) {
    *imageMat = cv::imread(imgPath, cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION);
    imgShape->width = imageMat->cols;
    imgShape->height = imageMat->rows;
    LogInfo << "the original image width:" << imageMat->cols;
    LogInfo << "the original image height:" << imageMat->rows;
    return APP_ERR_OK;
}

APP_ERROR SimplePOSE::Resize_Affine(const cv::Mat& srcImage, cv::Mat *dstImage,
    ImageShape *imgShape, const float center[], const float scale[]) {
    int new_width, new_height;
    new_height = static_cast<int>(imgShape->height);
    new_width = static_cast<int>(imgShape->width);

    float scale_tem[2] = {};
    scale_tem[0] = scale[0] * 200.0;
    scale_tem[1] = scale[1] * 200.0;
    float src_w = scale_tem[0];
    float dst_w = MODEL_WIDTH;
    float dst_h = MODEL_HEIGHT;
    float src_dir[2] = {};
    float dst_dir[2] = {};
    float sn = sin(0);
    float cs = cos(0);
    src_dir[0] = src_w * 0.5 * sn;
    src_dir[1] = src_w * (-0.5) * cs;
    dst_dir[0] = 0;
    dst_dir[1] = dst_w * (-0.5);

    float src[3][2] = {};
    float dst[3][2] = {};

    src[0][0] = center[0];
    src[0][1] = center[1];
    src[1][0] = center[0] + src_dir[0];
    src[1][1] = center[1] + src_dir[1];
    dst[0][0] = dst_w * 0.5;
    dst[0][1] = dst_h * 0.5;
    dst[1][0] = dst_w * 0.5 + dst_dir[0];
    dst[1][1] = dst_h * 0.5 + dst_dir[1];

    float src_direct[2] = {};
    src_direct[0] = src[0][0] - src[1][0];
    src_direct[1] = src[0][1] - src[1][1];
    src[2][0] = src[1][0] - src_direct[1];
    src[2][1] = src[1][1] + src_direct[0];

    float dst_direct[2] = {};
    dst_direct[0] = dst[0][0] - dst[1][0];
    dst_direct[1] = dst[0][1] - dst[1][1];
    dst[2][0] = dst[1][0] - dst_direct[1];
    dst[2][1] = dst[1][1] + dst_direct[0];
    cv::Point2f srcPoint2f[3], dstPoint2f[3];
    srcPoint2f[0] = cv::Point2f(static_cast<float>(src[0][0]), static_cast<float>(src[0][1]));
    srcPoint2f[1] = cv::Point2f(static_cast<float>(src[1][0]), static_cast<float>(src[1][1]));
    srcPoint2f[2] = cv::Point2f(static_cast<float>(src[2][0]), static_cast<float>(src[2][1]));
    dstPoint2f[0] = cv::Point2f(static_cast<float>(dst[0][0]), static_cast<float>(dst[0][1]));
    dstPoint2f[1] = cv::Point2f(static_cast<float>(dst[1][0]), static_cast<float>(dst[1][1]));
    dstPoint2f[2] = cv::Point2f(static_cast<float>(dst[2][0]), static_cast<float>(dst[2][1]));
    cv::Mat warp_mat(2, 3, CV_32FC1);
    warp_mat = cv::getAffineTransform(srcPoint2f, dstPoint2f);

    cv::Mat src_cv(new_height, new_width, CV_8UC3, srcImage.data);

    cv::Mat warp_dst = cv::Mat::zeros(cv::Size(static_cast<int>(MODEL_WIDTH), static_cast<int>(MODEL_HEIGHT)),
        src_cv.type());

    cv::warpAffine(src_cv, warp_dst, warp_mat, warp_dst.size());

    cv::Mat image_finally(warp_dst.rows, warp_dst.cols, CV_32FC3);

    warp_dst.convertTo(image_finally, CV_32FC3, 1 / 255.0);

    float mean[3] = { 0.485, 0.456, 0.406 };
    float std[3] = { 0.229, 0.224, 0.225 };
    for (int i = 0; i < image_finally.rows; i++) {
        for (int j = 0; j < image_finally.cols; j++) {
            if (warp_dst.channels() == 3) {
                image_finally.at<cv::Vec3f>(i, j)[0]= (image_finally.at<cv::Vec3f>(i, j)[0] - mean[0]) / std[0];
                image_finally.at<cv::Vec3f>(i, j)[1]= (image_finally.at<cv::Vec3f>(i, j)[1] - mean[1]) / std[1];
                image_finally.at<cv::Vec3f>(i, j)[2]= (image_finally.at<cv::Vec3f>(i, j)[2] - mean[2]) / std[2];
            }
        }
    }
    *dstImage = image_finally;
    return APP_ERR_OK;
}

APP_ERROR SimplePOSE::CVMatToTensorBase(const cv::Mat& imageMat, MxBase::TensorBase *tensorBase) {
    uint32_t dataSize = 1;
    for (size_t i = 0; i < modelDesc_.inputTensors.size(); ++i) {
        std::vector<uint32_t> shape = {};
        for (size_t j = 0; j < modelDesc_.inputTensors[i].tensorDims.size(); ++j) {
            shape.push_back((uint32_t)modelDesc_.inputTensors[i].tensorDims[j]);
        }
        for (uint32_t s = 0; s < shape.size(); ++s) {
            dataSize *= shape[s];
        }
    }
    // mat NHWC to NCHW
    size_t  H = 256, W = 192, C = 3;
    float mat_data[dataSize] = {};
    dataSize = dataSize * 4;
    for (size_t c = 0; c < C; c++) {
        for (size_t h = 0; h < H; h++) {
            for (size_t w = 0; w < W; w++) {
                int id = c * (H * W) + h * W + w;
                mat_data[id] = imageMat.at<cv::Vec3f>(h, w)[c];
            }
        }
    }
    MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(mat_data, dataSize, MxBase::MemoryData::MEMORY_HOST_MALLOC);
    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }
    std::vector<uint32_t> shape = { 1, 3, 256, 192 };
    *tensorBase = MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_FLOAT32);
    return APP_ERR_OK;
}

APP_ERROR SimplePOSE::CVMatToTensorBaseFlip(const cv::Mat& imageMat, MxBase::TensorBase* tensorBase) {
    uint32_t dataSize = 1;
    for (size_t i = 0; i < modelDesc_.inputTensors.size(); ++i) {
        std::vector<uint32_t> shape = {};
        for (size_t j = 0; j < modelDesc_.inputTensors[i].tensorDims.size(); ++j) {
            shape.push_back((uint32_t)modelDesc_.inputTensors[i].tensorDims[j]);
        }
        for (uint32_t s = 0; s < shape.size(); ++s) {
            dataSize *= shape[s];
        }
    }
    // mat NHWC to NCHW
    size_t  H = 256, W = 192, C = 3;
    float mat_data[dataSize] = {};
    dataSize = dataSize * 4;
    for (size_t c = 0; c < C; c++) {
        for (size_t h = 0; h < H; h++) {
            for (size_t w = 0; w < W; w++) {
                int id = c * (H * W) + h * W + w;
                mat_data[id] = imageMat.at<cv::Vec3f>(h, w)[c];
            }
        }
    }
    float mat_data_flip[dataSize] = {};
    for (size_t c = 0; c < C; c++) {
        for (size_t h = 0; h < H; h++) {
            for (size_t w = 0; w < W; w++) {
                int id1 = c * (H * W) + h * W + w;
                int id2 = c * (H * W) + h * W + W-w-1;
                mat_data_flip[id1] = mat_data[id2];
            }
        }
    }

    // LogInfo << "first " << mat_data[975] <<"   " << mat_data_flip[975];
    MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(mat_data_flip, dataSize, MxBase::MemoryData::MEMORY_HOST_MALLOC);
    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }
    std::vector<uint32_t> shape = { 1, 3, 256, 192 };
    *tensorBase = MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_FLOAT32);
    return APP_ERR_OK;
}

APP_ERROR SimplePOSE::Inference(const std::vector<MxBase::TensorBase> &inputs,
    std::vector<MxBase::TensorBase> *outputs) {
    auto dtypes = model_->GetOutputDataType();
    for (size_t i = 0; i < modelDesc_.outputTensors.size(); ++i) {
        std::vector<uint32_t> shape = {};
        for (size_t j = 0; j < modelDesc_.outputTensors[i].tensorDims.size(); ++j) {
            shape.push_back((uint32_t)modelDesc_.outputTensors[i].tensorDims[j]);
        }
        MxBase::TensorBase tensor(shape, dtypes[i], MxBase::MemoryData::MemoryType::MEMORY_DEVICE, deviceId_);
        APP_ERROR ret = MxBase::TensorBase::TensorBaseMalloc(tensor);
        if (ret != APP_ERR_OK) {
            LogError << "TensorBaseMalloc failed, ret=" << ret << ".";
            return ret;
        }
        outputs->push_back(tensor);
    }
    MxBase::DynamicInfo dynamicInfo = {};
    dynamicInfo.dynamicType = MxBase::DynamicType::STATIC_BATCH;
    dynamicInfo.batchSize = 1;

    APP_ERROR ret = model_->ModelInference(inputs, *outputs, dynamicInfo);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR SimplePOSE::PostProcess(const std::vector<MxBase::TensorBase> &inputs,
    const std::vector<MxBase::TensorBase>& inputs1,
    std::vector<std::vector<float> >* node_score_list, const float center[], const float scale[]) {
    APP_ERROR ret = post_->selfProcess(center, scale, inputs, inputs1, node_score_list);
    if (ret != APP_ERR_OK) {
        LogError << "Process failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

void drawpicture(const std::string& imgPath, const std::string& resultPath, const float (*preds)[2]) {
    cv::Mat imageMat = cv::imread(imgPath);
    for (int i = 0; i < NPOINTS; i++) {
        cv::Point center;
        center.x = static_cast<int>(preds[i][0]);
        center.y = static_cast<int>(preds[i][1]);
        cv::circle(imageMat, center, 1, cv::Scalar(0, 255, 85), -1);
    }
    std::string fileName = imgPath.substr(imgPath.find_last_of("/") + 1);
    size_t dot = fileName.find_last_of(".");
    std::string resFileName = resultPath + fileName.substr(0, dot) + "_detect_result.jpg";
    cv::imwrite(resFileName, imageMat);
}

void SaveInferResult(const std::string& imgPath, const std::vector<std::vector<float> >& node_score_list,
    const std::string &resultPath) {
    std::string path1 = "mkdir -p " + resultPath;
    system(path1.c_str());
    std::string fileName =
        imgPath.substr(imgPath.find_last_of("/") + 1);
    size_t dot = fileName.find_last_of(".");
    std::string resFileName = resultPath +
        fileName.substr(0, dot) + "_1.txt";
    std::ofstream outfile(resFileName);
    if (node_score_list.empty()) {
        LogWarn << "The predict result is empty.";
        return;
    }
    for (int i = 0; i < node_score_list.size(); i++) {
        float preds[NPOINTS][2] = {};
        float maxvals[NPOINTS] = {};
        int idx = 0;
        for (int j = 0; j < node_score_list[i].size(); j+=3) {
            preds[idx][0] = node_score_list[i][j];
            preds[idx][1] = node_score_list[i][j + 1];
            maxvals[idx] = node_score_list[i][j + 2];
            idx++;
        }
        LogInfo << "infer result:";
        LogInfo << "preds:";
        for (int m = 0; m < NPOINTS; m++) {
            LogInfo << preds[m][0] << "  " << preds[m][1];
        }
        LogInfo << "maxvals:";
        for (int m = 0; m < NPOINTS; m++) {
            LogInfo << maxvals[m];
        }
        drawpicture(imgPath, resultPath, preds);

        std::string result_str = "preds";
        for (int m = 0; m < NPOINTS; m++) {
            result_str += " " + std::to_string(preds[m][0]) + " " + std::to_string(preds[m][1]);
        }
        outfile << result_str << std::endl;
        result_str = "maxvals";
        for (int m = 0; m < NPOINTS; m++) {
            result_str += " " + std::to_string(maxvals[m]);
        }
        outfile << result_str << std::endl;
    }
    outfile.close();
}


APP_ERROR SimplePOSE::Process(const std::string& BBOX_FILE, const std::string &imgPath, const std::string &resultPath) {
    std::ifstream t(BBOX_FILE);
    std::string str((std::istreambuf_iterator<char>(t)),
        std::istreambuf_iterator<char>());

    rapidjson::Document document;
    document.Parse(str.c_str());

    const rapidjson::Value& arr = document;

    for (int i = 0; i < arr.Size(); ++i) {
    // for (int i = 0; i < 1; ++i) {
        const rapidjson::Value& tmp = arr[i];
        rapidjson::Value::ConstMemberIterator iter = tmp.FindMember("category_id");
        if (iter != tmp.MemberEnd()) {
            if (iter->value.GetInt() != 1) {
                continue;
            }
        }
        float x = 0, y = 0, w = 0, h = 0;
        float center[2] = {};
        float scale[2] = {};
        if (tmp.HasMember("bbox")) {
            const rapidjson::Value& childValue = tmp["bbox"];
            x = childValue[0].GetFloat();
            y = childValue[1].GetFloat();
            w = childValue[2].GetFloat();
            h = childValue[3].GetFloat();
        }

        center[0] = x + w * 0.5;
        center[1] = y + h * 0.5;
        // float aspect_ratio = (float)MODEL_WIDTH/ (float)MODEL_HEIGHT;
        float aspect_ratio = static_cast<float>(MODEL_WIDTH / MODEL_HEIGHT);
        if (w > aspect_ratio * h) {
            h = w * 1.0 / aspect_ratio;
        }
        if (w < aspect_ratio * h) {
            w = h * aspect_ratio;
        }
        scale[0] = w * 1.0 / 200;
        scale[1] = h * 1.0 / 200;
        if (center[0] != -1) {
            scale[0] = scale[0] * 1.25;
            scale[1] = scale[1] * 1.25;
        }
        int image_id = 0;
        iter = tmp.FindMember("image_id");
        if (iter != tmp.MemberEnd()) {
            image_id = iter->value.GetInt();
        }

        std::string image_id_str = std::to_string(image_id);
        while (image_id_str.size() < 12) {
            image_id_str = "0" + image_id_str;
        }
        image_id_str = image_id_str + ".jpg";

        std::string img_path = imgPath + image_id_str;
        cv::Mat imageMat;
        ImageShape imageShape{};
        APP_ERROR ret = ReadImage(img_path, &imageMat, &imageShape);
        if (ret != APP_ERR_OK) {
            LogError << "ReadImage failed, ret=" << ret << ".";
            return ret;
        }
        cv::Mat dstImage;

        Resize_Affine(imageMat, &dstImage, &imageShape, center, scale);
        std::vector<MxBase::TensorBase> inputs = {};
        std::vector<MxBase::TensorBase> outputs = {};
        std::vector<MxBase::TensorBase> inputs1 = {};
        std::vector<MxBase::TensorBase> outputs1 = {};
        MxBase::TensorBase tensorBase, tensorflip;
        ret = CVMatToTensorBase(dstImage, &tensorBase);
        if (ret != APP_ERR_OK) {
            LogError << "CVMatToTensorBase failed, ret=" << ret << ".";
            return ret;
        }
        ret = CVMatToTensorBaseFlip(dstImage, &tensorflip);
        if (ret != APP_ERR_OK) {
            LogError << "CVMatToTensorBase failed, ret=" << ret << ".";
            return ret;
        }
        inputs.push_back(tensorBase);
        inputs1.push_back(tensorflip);

        ret = Inference(inputs, &outputs);
        std::vector<std::vector<float> > node_score_list = {};

        if (ret != APP_ERR_OK) {
            LogError << "Inference failed, ret=" << ret << ".";
            return ret;
        }
        ret = Inference(inputs1, &outputs1);
        if (ret != APP_ERR_OK) {
            LogError << "Inference failed, ret=" << ret << ".";
            return ret;
        }
        LogInfo << "Inference success, ret=" << ret << ".";
        ret = PostProcess(outputs, outputs1, &node_score_list, center, scale);
        if (ret != APP_ERR_OK) {
            LogError << "PostProcess failed, ret=" << ret << ".";
            return ret;
        }
        SaveInferResult(img_path, node_score_list, resultPath);
    }
    return APP_ERR_OK;
}
