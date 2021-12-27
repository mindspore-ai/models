/*
 * Copyright (c) 2021.Huawei Technologies Co., Ltd. All rights reserved.
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

#include "MxCenterfacePostProcessor.h"

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

APP_ERROR MxCenterfacePostProcessor::Init(const std::string &configPath,
                                          const std::string &labelPath) {
    APP_ERROR ret = APP_ERR_OK;

    std::map<std::string, std::shared_ptr<void>> postConfig;
    if (!configPath.empty())
        postConfig["postProcessConfigPath"] =
            std::make_shared<std::string>(configPath);
    if (!labelPath.empty())
        postConfig["labelPath"] = std::make_shared<std::string>(labelPath);

    ret = Init(postConfig);
    if (ret == APP_ERR_OK) {  // Init for this class derived information
        ret = ReadConfigParams();
    }
    return ret;
}

APP_ERROR MxCenterfacePostProcessor::Init(
    const std::map<std::string, std::shared_ptr<void>> &postConfig) {
    APP_ERROR ret = LoadConfigDataAndLabelMap(postConfig);
    if (ret != APP_ERR_OK) {
        LogError << "LoadConfigDataAndLabelMap failed. ret=" << ret;
        return ret;
    }
    ReadConfigParams();
    LogDebug << "End to Init centerface postprocessor";
    return APP_ERR_OK;
}

APP_ERROR MxCenterfacePostProcessor::Process(
    const std::vector<MxBase::TensorBase> &tensors,
    std::vector<std::vector<MxBase::ObjectInfo>> &objectInfos,
    const std::vector<MxBase::ResizedImageInfo> &resizedImageInfos,
    const std::map<std::string, std::shared_ptr<void>> &configParamMap) {
    LogDebug << "Start to Process CenterfacePostProcess ...";
    APP_ERROR ret = APP_ERR_OK;
    auto outputs = tensors;
    ret = CheckAndMoveTensors(outputs);
    if (ret != APP_ERR_OK) {
        LogError << "CheckAndMoveTensors failed:" << ret;
        return ret;
    }

    auto shape = outputs[0].GetShape();
    size_t batch_size = shape[0];
    std::vector<void *> featLayerData;
    MxBase::ResizedImageInfo resizeImgInfo;

    for (size_t i = 0; i < batch_size; i++) {
        std::vector<MxBase::ObjectInfo> objInfo;
        featLayerData.reserve(tensors.size());
        std::transform(tensors.begin(), tensors.end(), featLayerData.begin(),
                       [batch_size, i](MxBase::TensorBase tensor) -> void * {
                           return reinterpret_cast<void *>(
                               reinterpret_cast<char *>(tensor.GetBuffer()) +
                               tensor.GetSize() / batch_size * i);
                       });
        resizeImgInfo = resizedImageInfos[i];
        this->Process(featLayerData, objInfo, resizeImgInfo);
        objectInfos.push_back(objInfo);
    }
    return APP_ERR_OK;
}

APP_ERROR MxCenterfacePostProcessor::Process(
    std::vector<void *> &featLayerData,
    std::vector<MxBase::ObjectInfo> &objInfos,
    const MxBase::ResizedImageInfo &resizeInfo) {
    ImageInfo imgInfo;
    imgInfo.imgWidth = resizeInfo.widthOriginal;
    imgInfo.imgHeight = resizeInfo.heightOriginal;
    imgInfo.modelWidth = resizeInfo.widthResize;
    imgInfo.modelHeight = resizeInfo.heightResize;

    ObjectDetectionOutput(featLayerData, objInfos, imgInfo);
    return APP_ERR_OK;
}

void MxCenterfacePostProcessor::calculateScaleCoord(const ImageInfo &imgInfo,
                                                    float &scaleX,
                                                    float &scaleY,
                                                    float &offsetX,
                                                    float &offsetY) {
    offsetX = 0.0;
    offsetY = 0.0;
    if (IsRawResize()) {  // resize to left-top corner and padding black
        scaleX = static_cast<float>(imgInfo.modelWidth) / imgInfo.imgWidth;
        scaleY = static_cast<float>(imgInfo.modelHeight) / imgInfo.imgHeight;
    } else if (static_cast<float>(imgInfo.modelWidth) / imgInfo.imgWidth <=
               static_cast<float>(imgInfo.modelHeight) /
                   imgInfo.imgHeight) {  // scale by width
        scaleX = scaleY =
            static_cast<float>(imgInfo.modelWidth) / imgInfo.imgWidth;
        offsetX = 0;
        offsetY =
            IsResizeNoCenter()
                ? 0
                : (imgInfo.modelHeight - imgInfo.imgHeight * scaleY) / 2.0;
    } else {  // scale by height
        scaleX = scaleY =
            static_cast<float>(imgInfo.modelHeight) / imgInfo.imgHeight;
        offsetX = IsResizeNoCenter()
                      ? 0
                      : (imgInfo.modelWidth - imgInfo.imgWidth * scaleX) / 2.0;
        offsetY = 0;
    }
}

void MxCenterfacePostProcessor::ObjectDetectionOutput(
    std::vector<void *> &featLayerData,
    std::vector<MxBase::ObjectInfo> &objInfos, ImageInfo &imgInfo) {

    auto *scores = static_cast<float *>(featLayerData[0]);   // score
    auto *scaleW = static_cast<float *>(featLayerData[1]);   // 2*H*W
    auto *offsetX = static_cast<float *>(featLayerData[2]);  // 2*H*W
    auto *topind = static_cast<uint32_t *>(featLayerData[4]);  // (y*w+x)

    size_t hwSize = m_nHMWidth_ * m_nHMHeight_;

    cv::Mat transform_output = CVImage::GetAffineTransform(
        imgInfo.imgWidth, imgInfo.imgHeight, m_nHMWidth_, m_nHMHeight_, true);
    cv::Point2d src;
    cv::Point2d dst;

    auto *offsetY = offsetX + hwSize;
    auto *scaleH = scaleW + hwSize;

    float scaleX, scaleY, scale_off_x, scale_off_y;
    imgInfo.modelWidth = m_nHMWidth_;
    imgInfo.modelHeight = m_nHMHeight_;
    calculateScaleCoord(imgInfo, scaleX, scaleY, scale_off_x, scale_off_y);

    for (int index = 0; index < m_nTopKN_; index++) {
        __attribute__((unused)) uint32_t y = *topind / m_nHMWidth_;
        __attribute__((unused)) uint32_t x = *topind % m_nHMWidth_;

        objInfos.resize(objInfos.size() + 1);
        MxBase::ObjectInfo &detect = objInfos[objInfos.size() - 1];
        float c0 = (offsetX[*topind] + x);
        float c1 = (offsetY[*topind] + y);
        float w = std::exp(scaleW[*topind]) * 4;
        float h = std::exp(scaleH[*topind]) * 4;

        detect.x0 = c0 - w / 2.0f;
        detect.y0 = c1 - h / 2.0f;
        detect.x1 = c0 + w / 2.0f;
        detect.y1 = c1 + h / 2.0f;

        if (IsUseAffineTransform()) {
            CVImage::AffineTransform(transform_output, detect.x0, detect.y0);
            CVImage::AffineTransform(transform_output, detect.x1, detect.y1);
        } else {
            detect.x0 = (detect.x0 - scale_off_x) / scaleX;
            detect.y0 = (detect.y0 - scale_off_y) / scaleY;
            detect.x1 = (detect.x1 - scale_off_x) / scaleX;
            detect.y1 = (detect.y1 - scale_off_y) / scaleY;
        }
        detect.confidence = scores[index];
        topind++;
    }
    if (m_isUseSoftNms_) {
        Soft_NMS(objInfos);
    }
    for (size_t index = 0; index < objInfos.size(); index++) {
        if (objInfos[index].confidence < scoreThresh_) {
            objInfos.resize(index);
            break;
        }
    }
}

APP_ERROR MxCenterfacePostProcessor::ReadConfigParams() {
    configData_.GetFileValue<float>("SCORE_THRESH", scoreThresh_);
    configData_.GetFileValue<float>("IOU_THRESH", iouThresh_);
    configData_.GetFileValue<int>("MAX_PER_IMG", maxPerImg_);
    configData_.GetFileValue<int>("AFFINE_TRANSFORM", m_iUseAffineTransform_);
    configData_.GetFileValue<int>("SOFT_NMS", m_isUseSoftNms_);
    return APP_ERR_OK;
}
