/*
 * Copyright(C) 2021 Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "MxBase/Log/Log.h"
#include "MxBase/ModelPostProcessors/ModelPostProcessorBase/ObjectPostDataType.h"
#include "MxBase/PostProcessBases/ObjectPostProcessBase.h"
#include "MxImage.h"
#include "MxUtil.h"
#include "acl/acl.h"

class MxCenterNetPostProcessor : public MxBase::ObjectPostProcessBase {
 public:
    APP_ERROR Init(const std::string &configPath,
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
    APP_ERROR Init(const std::map<std::string, std::shared_ptr<void>>
                       &postConfig) override {
        APP_ERROR ret = LoadConfigDataAndLabelMap(postConfig);
        if (ret != APP_ERR_OK) {
            LogError << "LoadConfigDataAndLabelMap failed. ret=" << ret;
            return ret;
        }
        ReadConfigParams();
        LogDebug << "End to Init centerface postprocessor";
        return APP_ERR_OK;
    }

    /*
     * @description: Do nothing temporarily.
     * @return APP_ERROR error code.
     */
    APP_ERROR DeInit() override {
        // do nothing for this derived class
        return APP_ERR_OK;
    }

    APP_ERROR Process(
        const std::vector<MxBase::TensorBase> &tensors,
        std::vector<std::vector<MxBase::ObjectInfo>> &objectInfos,
        const std::vector<MxBase::ResizedImageInfo> &resizedImageInfos = {},
        const std::map<std::string, std::shared_ptr<void>> &configParamMap = {})
        override {
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
            std::vector<MxBase::ObjectInfo> objInfos;
            featLayerData.reserve(tensors.size());
            std::transform(
                tensors.begin(), tensors.end(), featLayerData.begin(),
                [batch_size, i](const MxBase::TensorBase &tensor) -> void * {
                    return reinterpret_cast<void *>(
                        reinterpret_cast<char *>(tensor.GetBuffer()) +
                        tensor.GetSize() / batch_size * i);
                });
            resizeImgInfo = resizedImageInfos[i];
            this->Process(featLayerData, objInfos, resizeImgInfo);
            objectInfos.push_back(objInfos);
        }
        return APP_ERR_OK;
    }

    APP_ERROR Process(std::vector<void *> &featLayerData,
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
    // default 0 resize to center
    bool IsUseAffineTransform() const { return m_iUseAffineTransform == 1; }
    bool IsResizeNoCenter() const { return m_iUseAffineTransform == -1; }
    bool IsRawResize() const { return m_iUseAffineTransform == -2; }
    bool IsResizeNoMove() const { return m_iUseAffineTransform == -3; }
    bool IsValidTensors(const std::vector<MxBase::TensorBase> &tensors) const {
        return true;
    }

 private:
    struct Point {
        float x;
        float y;
    };

    struct OutputInfo {
        Point bbox[2];
        float score;
        Point keyPoints[17];
        float classId;
    };

    void calculateScaleCoord(const ImageInfo &imgInfo, float &scaleX,
                             float &scaleY, float &offsetX, float &offsetY) {
        offsetX = 0.0;
        offsetY = 0.0;
        if (IsRawResize()) {
            scaleX = static_cast<float>(imgInfo.modelWidth) / imgInfo.imgWidth;
            scaleY =
                static_cast<float>(imgInfo.modelHeight) / imgInfo.imgHeight;
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
            offsetX =
                IsResizeNoCenter()
                    ? 0
                    : (imgInfo.modelWidth - imgInfo.imgWidth * scaleX) / 2.0;
            offsetY = 0;
        }
    }

    void ObjectDetectionOutput(std::vector<void *> &featLayerData,
                               std::vector<MxBase::ObjectInfo> &objInfos,
                               const ImageInfo &imgInfo) {
        cv::Mat transform_output =
            CVImage::GetAffineTransform(imgInfo.imgWidth, imgInfo.imgHeight,
                                        m_nHMWidth_, m_nHMHeight_, true);
        auto *outputObjects = static_cast<OutputInfo *>(featLayerData[0]);

        // convert every outputInfo to an objectinfo
        for (int idx = 0; idx < m_nTopKN; ++idx, ++outputObjects) {
            MxBase::ObjectInfo obj;
            obj.x0 = outputObjects->bbox[0].x;
            obj.y0 = outputObjects->bbox[0].y;
            obj.x1 = outputObjects->bbox[1].x,
            obj.y1 = outputObjects->bbox[1].y;
            obj.confidence = outputObjects->score;
            obj.classId = outputObjects->classId;

            CVImage::AffineTransform(transform_output, obj.x0, obj.y0);
            CVImage::AffineTransform(transform_output, obj.x1, obj.y1);

            for (int i = 0; i < m_nKeyCounts_; ++i) {
                // since we have to use ObjectInfo, and it only has a member
                // variable "mask", which is a vector<vector<int>>, to store key
                // point info, we multiply the x, y value of a point by 1e3 to
                // keep the precision
                float keyPoint_x = outputObjects->keyPoints[i].x;
                float keyPoint_y = outputObjects->keyPoints[i].y;
                CVImage::AffineTransform(transform_output, keyPoint_x,
                                         keyPoint_y);


                auto temp_x = reinterpret_cast<unsigned char*> (&keyPoint_x);
                auto temp_y = reinterpret_cast<unsigned char*> (&keyPoint_y);

                obj.mask.push_back({*temp_x, *(temp_x+1), *(temp_x+2),
                                    *(temp_x+3), *temp_y, *(temp_y+1),
                                    *(temp_y+2), *(temp_y+3)});
            }
            objInfos.push_back(obj);
        }

        if (m_isUseSoftNms) {
            Soft_NMS(objInfos, 0.01);
        }
        for (size_t index = 0; index < objInfos.size(); index++) {
            if (objInfos[index].confidence < scoreThresh_) {
                objInfos.resize(index);
                break;
            }
        }
    }

    void GetValidDetBoxes(std::vector<std::shared_ptr<void>> &featLayerData,
                          std::vector<MxBase::DetectBox> &detBoxes,
                          ImageInfo &imgInfo) const;

    void ConvertObjInfoFromDetectBox(std::vector<MxBase::DetectBox> &detBoxes,
                                     std::vector<ObjDetectInfo> &objInfos,
                                     ImageInfo &imgInfo) const;

    // retrieve this specific config parameters
    APP_ERROR ReadConfigParams() {
        configData_.GetFileValue<float>("SCORE_THRESH", scoreThresh_);
        configData_.GetFileValue<float>("IOU_THRESH", iouThresh_);
        configData_.GetFileValue<int>("MAX_PER_IMG", maxPerImg_);
        configData_.GetFileValue<int>("AFFINE_TRANSFORM",
                                      m_iUseAffineTransform);
        configData_.GetFileValue<int>("SOFT_NMS", m_isUseSoftNms);
        return APP_ERR_OK;
    }

 private:
    int m_nHMWidth_ = 128;
    int m_nHMHeight_ = 128;
    int m_nKeyCounts_ = 17;
    // max top in model
    int m_nTopKN = 100;
    // max object per image find
    int maxPerImg_ = 400;
    // IOU thresh hold
    float iouThresh_ = 0.5;
    // [0]  use normal resize to do image preprocess , resize into center
    // [1]  use affine transform to do image resize
    // [-1] resize at left-top corner , no move
    int m_iUseAffineTransform = 0;
    int m_isUseSoftNms = 1;
};
