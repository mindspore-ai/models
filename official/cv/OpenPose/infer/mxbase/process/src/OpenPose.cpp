/*
* Copyright (c) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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

#include <memory>
#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include "acl/acl.h"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"
#include "OpenPose.h"

using  namespace MxBase;
namespace {
    const uint32_t YUV_BYTE_NU = 3;
    const uint32_t YUV_BYTE_DE = 2;
    const uint32_t MODEL_HEIGHT = 560;
    const uint32_t MODEL_WIDTH = 560;
    const int NPOINTS = 18;
}  // namespace

void PrintTensorShape(const std::vector<TensorDesc> &tensorDescVec, const std::string &tensorName) {
    LogInfo << "The shape of " << tensorName << " is as follows:";
    for (size_t i = 0; i < tensorDescVec.size(); ++i) {
        LogInfo << "  Tensor " << i << ":";
        for (size_t j = 0; j < tensorDescVec[i].tensorDims.size(); ++j) {
            LogInfo << "   dim: " << j << ": " << tensorDescVec[i].tensorDims[j];
        }
    }
}

APP_ERROR OpenPose::Init(const InitParam &initParam) {
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
    post_ = std::make_shared<MxBase::OpenPoseMindsporePost>();
    return APP_ERR_OK;
}

APP_ERROR OpenPose::DeInit() {
    dvppWrapper_->DeInit();
    model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

APP_ERROR OpenPose::ReadImage(const std::string &imgPath, MxBase::DvppDataInfo *output, ImageShape *imgShape) {
    APP_ERROR ret = dvppWrapper_->DvppJpegDecode(imgPath, *output);
    if (ret != APP_ERR_OK) {
        LogError << "DvppWrapper DvppJpegDecode failed, ret=" << ret << ".";
        return ret;
    }
    imgShape->width = output->width;
    imgShape->height = output->height;
    return APP_ERR_OK;
}

APP_ERROR OpenPose::Resize(const MxBase::DvppDataInfo &input, MxBase::TensorBase *outputTensor) {
    MxBase::CropRoiConfig cropRoi = {0, input.width, input.height, 0};
    float ratio =
      std::min(static_cast<float>(MODEL_WIDTH) / input.width, static_cast<float>(MODEL_HEIGHT) / input.height);
    MxBase::CropRoiConfig pasteRoi = {0, 0, 0, 0};
    pasteRoi.x1 = input.width * ratio;
    pasteRoi.y1 = input.height * ratio;

    MxBase::MemoryData memoryData(MODEL_WIDTH * MODEL_HEIGHT * YUV_BYTE_NU / YUV_BYTE_DE,
                                  MemoryData::MemoryType::MEMORY_DVPP, deviceId_);
    APP_ERROR ret = MxBase::MemoryHelper::MxbsMalloc(memoryData);
    if (ret != APP_ERR_OK) {
        LogError << "Fail to allocate dvpp memory.";
        MemoryHelper::MxbsFree(memoryData);
        return APP_ERR_COMM_INVALID_PARAM;
    }

    ret = MxBase::MemoryHelper::MxbsMemset(memoryData, 0, memoryData.size);
    if (ret != APP_ERR_OK) {
        LogError << "Fail to set 0.";
        MemoryHelper::MxbsFree(memoryData);
        return APP_ERR_COMM_INVALID_PARAM;
    }

    MxBase::DvppDataInfo output = {};
    output.dataSize = memoryData.size;
    output.width = MODEL_WIDTH;
    output.height = MODEL_HEIGHT;
    output.widthStride = MODEL_WIDTH;
    output.heightStride = MODEL_HEIGHT;
    output.format = input.format;
    output.data = static_cast<uint8_t *>(memoryData.ptrData);

    ret = dvppWrapper_->VpcCropAndPaste(input, output, pasteRoi, cropRoi);
    if (ret != APP_ERR_OK) {
        LogError << "VpcCropAndPaste failed, ret=" << ret << ".";
        return ret;
    }

    std::vector<uint32_t> shape = {output.heightStride * YUV_BYTE_NU / YUV_BYTE_DE, output.widthStride};
    *outputTensor = TensorBase(memoryData, false, shape, TENSOR_DTYPE_UINT8);
    return APP_ERR_OK;
}

APP_ERROR OpenPose::Inference(const std::vector<MxBase::TensorBase> &inputs,
    std::vector<MxBase::TensorBase> *outputs) {
    auto dtypes = model_->GetOutputDataType();
    for (size_t i = 0; i < modelDesc_.outputTensors.size(); ++i) {
        std::vector<uint32_t> shape = {};
        for (size_t j = 0; j < modelDesc_.outputTensors[i].tensorDims.size(); ++j) {
            shape.push_back((uint32_t)modelDesc_.outputTensors[i].tensorDims[j]);
        }
        TensorBase tensor(shape, dtypes[i], MemoryData::MemoryType::MEMORY_DEVICE, deviceId_);
        APP_ERROR ret = TensorBase::TensorBaseMalloc(tensor);
        if (ret != APP_ERR_OK) {
            LogError << "TensorBaseMalloc failed, ret=" << ret << ".";
            return ret;
        }
        outputs->push_back(tensor);
    }
    DynamicInfo dynamicInfo = {};
    dynamicInfo.dynamicType = DynamicType::STATIC_BATCH;
    dynamicInfo.batchSize = 1;

    APP_ERROR ret = model_->ModelInference(inputs, *outputs, dynamicInfo);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }

    return APP_ERR_OK;
}

APP_ERROR OpenPose::PostProcess(const std::vector<MxBase::TensorBase> &inputs,
    const std::vector<int> &vision_infos,
    std::vector<std::vector<PartPair> > *person_list) {
    APP_ERROR ret = post_->selfProcess(inputs, vision_infos, person_list);
    if (ret != APP_ERR_OK) {
        LogError << "Process failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR OpenPose::GetInferResults(const std::string &imgPath, const std::vector<std::vector<PartPair> > &person_list,
                            const std::string &resultPath) {
    std::string fileName = imgPath.substr(imgPath.find_last_of("/") + 1);
    size_t dot = fileName.find_last_of(".");
    std::string resFileName = resultPath + fileName.substr(0, dot) + "_1.txt";
    std::ofstream outfile(resFileName);
    if (outfile.fail()) {
        LogError << "Failed to open result file: ";
        return APP_ERR_COMM_FAILURE;
    }
    std::vector<std::vector<float> > coco_keypoints;
    std::vector<float> scores;
    float coor_bias = 0.5;
    float float_equal_zero_bias = 0.000001;
    for (int k = 0; k < person_list.size(); k++) {
        float person_score = post_->PersonScore(person_list[k]);
        // Ignore person with score 0
        if (fabs(person_score - 0) < float_equal_zero_bias) {
            continue;
        }
        person_score = person_score - 1;
        std::vector<float> keypoints(17*3, 0.0);
        int to_coco_map[] = {0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3};
        std::set<int> seen_idx = {1};
        for (int j = 0; j < person_list[k].size(); j++) {
            PartPair skele = person_list[k][j];
            int part_idx1 = skele.partIdx1;
            // two end points of a skeleton
            int part_idx2 = skele.partIdx2;
            if (seen_idx.count(part_idx1) == 0) {
                float center_x = skele.coord1[0] + coor_bias;
                float center_y = skele.coord1[1] + coor_bias;
                keypoints[to_coco_map[part_idx1] * 3 + 0] = center_x;
                keypoints[to_coco_map[part_idx1] * 3 + 1] = center_y;
                keypoints[to_coco_map[part_idx1] * 3 + 2] = 1;
                seen_idx.insert(part_idx1);
            }
            if (seen_idx.count(part_idx2) == 0) {
                float center_x = skele.coord2[0] + coor_bias;
                float center_y = skele.coord2[1] + coor_bias;
                keypoints[to_coco_map[part_idx2] * 3 + 0] = center_x;
                keypoints[to_coco_map[part_idx2] * 3 + 1] = center_y;
                keypoints[to_coco_map[part_idx2] * 3 + 2] = 1;
                seen_idx.insert(part_idx2);
            }
        }
        coco_keypoints.push_back(keypoints);
        scores.push_back(person_score);
        std::string resultStr;
        resultStr += "[";
        std::cout << "keypoints: [";
        int i = 0;
        for (i = 0; i < keypoints.size()-1; i++) {
            resultStr += std::to_string(keypoints[i]) + ",";
        }
        resultStr += "]";
        outfile << resultStr << std::endl;
        resultStr = "person_score: ";
        resultStr += std::to_string(person_score);
        outfile << resultStr << std::endl;
    }
    outfile.close();
    return APP_ERR_OK;
}

void OpenPose::DrawPoseBbox(const std::string &imgPath, const std::vector<std::vector<PartPair> > &person_list,
                            const std::string &resultPath) {
    std::vector<std::vector<int> > COCO_PAIRS_RENDER = {std::vector<int>{1, 2}, std::vector<int>{1, 5},
        std::vector<int>{2, 3}, std::vector<int>{3, 4}, std::vector<int>{5, 6}, std::vector<int>{6, 7},
        std::vector<int>{1, 8}, std::vector<int>{8, 9}, std::vector<int>{9, 10}, std::vector<int>{1, 11},
        std::vector<int>{11, 12}, std::vector<int>{12, 13}, std::vector<int>{1, 0}, std::vector<int>{0, 14},
        std::vector<int>{14, 16}, std::vector<int>{0, 15}, std::vector<int>{15, 17}};  // = 19

    std::vector<cv::Scalar> COCO_COLORS = {cv::Scalar(255, 0, 0), cv::Scalar(255, 85, 0), cv::Scalar(255, 170, 0),
                cv::Scalar(255, 255, 0), cv::Scalar(170, 255, 0), cv::Scalar(85, 255, 0), cv::Scalar(0, 255, 0),
                cv::Scalar(0, 255, 85), cv::Scalar(0, 255, 170), cv::Scalar(0, 255, 255), cv::Scalar(0, 170, 255),
                cv::Scalar(0, 85, 255), cv::Scalar(0, 0, 255), cv::Scalar(85, 0, 255), cv::Scalar(170, 0, 255),
                cv::Scalar(255, 0, 255), cv::Scalar(255, 0, 170), cv::Scalar(255, 0, 85)};

    cv::Mat imageMat = cv::imread(imgPath, cv::IMREAD_COLOR);
    for (int k = 0; k < person_list.size(); k++) {
        std::map<int, cv::Point> centers;
        std::set<int> seen_idx;
        for (int j = 0; j < person_list[k].size(); j++) {
            PartPair skele = person_list[k][j];
            // two end points of a skeleton
            int part_idx1 = skele.partIdx1;
            int part_idx2 = skele.partIdx2;
            if (seen_idx.count(part_idx1) == 0) {
                cv::Point center;
                center.x = static_cast<int>(skele.coord1[0]);
                center.y = static_cast<int>(skele.coord1[1]);
                centers[part_idx1] = center;
                cv::circle(imageMat, center, 3, COCO_COLORS[part_idx1], -1, cv::LINE_AA);
                seen_idx.insert(part_idx1);
            }
            if (seen_idx.count(part_idx2) == 0) {
                cv::Point center;
                center.x = static_cast<int>(skele.coord2[0]);
                center.y = static_cast<int>(skele.coord2[1]);
                centers[part_idx2] = center;
                cv::circle(imageMat, center, 3, COCO_COLORS[part_idx2], -1, cv::LINE_AA);
                seen_idx.insert(part_idx2);
            }
        }
        for (int i = 0; i < COCO_PAIRS_RENDER.size(); i++) {
            std::vector<int> pair = COCO_PAIRS_RENDER[i];
            if ((seen_idx.count(pair[0]) != 0) && (seen_idx.count(pair[1]) != 0))
                cv::line(imageMat, centers[pair[0]], centers[pair[1]], COCO_COLORS[i], 2, cv::LINE_AA);
        }
    }
    std::string fileName = imgPath.substr(imgPath.find_last_of("/") + 1);
    size_t dot = fileName.find_last_of(".");
    std::string resFileName = resultPath + fileName.substr(0, dot) + "_detect_result.jpg";
    cv::imwrite(resFileName, imageMat);
}

APP_ERROR OpenPose::Process(const std::string &imgPath, const std::string &resultPath) {
    ImageShape imageShape{};
    MxBase::DvppDataInfo dvppData = {};

    APP_ERROR ret = ReadImage(imgPath, &dvppData, &imageShape);
    if (ret != APP_ERR_OK) {
        LogError << "ReadImage failed, ret=" << ret << ".";
        return ret;
    }
    TensorBase resizeImage;
    ret = Resize(dvppData, &resizeImage);
    if (ret != APP_ERR_OK) {
        LogError << "Resize failed, ret=" << ret << ".";
        return ret;
    }

    std::vector<int> vision_infos;
    vision_infos.push_back(imageShape.height);
    vision_infos.push_back(imageShape.width);
    vision_infos.push_back(MODEL_HEIGHT);
    vision_infos.push_back(MODEL_WIDTH);

    std::vector<MxBase::TensorBase> inputs = {};
    std::vector<MxBase::TensorBase> outputs = {};

    inputs.push_back(resizeImage);

    ret = Inference(inputs, &outputs);
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }
    LogInfo << "Inference success, ret=" << ret << ".";
    std::vector<std::vector<PartPair> > person_list {};

    ret = PostProcess(outputs, vision_infos, &person_list);
    if (ret != APP_ERR_OK) {
        LogError << "PostProcess failed, ret=" << ret << ".";
        return ret;
    }
    // Get keypoints and person_score info.
    ret = GetInferResults(imgPath, person_list, resultPath);
    if (ret != APP_ERR_OK) {
        LogError << "Save infer results into file failed. ret = " << ret << ".";
        return ret;
    }
    // Visualize the postprocess results.
    DrawPoseBbox(imgPath, person_list, resultPath);
    return APP_ERR_OK;
}
