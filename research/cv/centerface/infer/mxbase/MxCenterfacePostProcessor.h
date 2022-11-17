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

#ifndef OFFICIAL_CV_CENTERFACE_INFER_MXBASE_MXCENTERFACEPOSTPROCESSOR_H_
#define OFFICIAL_CV_CENTERFACE_INFER_MXBASE_MXCENTERFACEPOSTPROCESSOR_H_
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "MxBase/Log/Log.h"
#include "MxBase/ModelPostProcessors/ModelPostProcessorBase/ObjectPostDataType.h"
#include "MxBase/PostProcessBases/ObjectPostProcessBase.h"
#include "MxImage.h"
#include "MxUtil.h"
#include "acl/acl.h"

class MxCenterfacePostProcessor : public MxBase::ObjectPostProcessBase {
 public:
    // @modify:used directly instead of called within framework
    APP_ERROR Init(const std::string &configPath, const std::string &labelPath);

    APP_ERROR Init(const std::map<std::string, std::shared_ptr<void>>
                       &postConfig) override;

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
        override;

    APP_ERROR Process(std::vector<void *> &featLayerData,
                      std::vector<MxBase::ObjectInfo> &objInfos,
                      const MxBase::ResizedImageInfo &resizeInfo);

    // default 0 resize to center
    bool IsUseAffineTransform() const { return m_iUseAffineTransform_ == 1; }
    bool IsResizeNoCenter() const { return m_iUseAffineTransform_ == -1; }
    bool IsRawResize() const { return m_iUseAffineTransform_ == -2; }
    bool IsResizeNoMove() const { return m_iUseAffineTransform_ == -3; }

 private:
    void calculateScaleCoord(const ImageInfo &imgInfo, float &scaleX,
                             float &scaleY, float &offsetX, float &offsetY);
    void ObjectDetectionOutput(std::vector<void *> &featLayerData,
                               std::vector<MxBase::ObjectInfo> &objInfos,
                               ImageInfo &imgInfo);

    void GetValidDetBoxes(std::vector<std::shared_ptr<void>> &featLayerData,
                          std::vector<MxBase::DetectBox> &detBoxes,
                          ImageInfo &imgInfo) const;

    void ConvertObjInfoFromDetectBox(std::vector<MxBase::DetectBox> &detBoxes,
                                     std::vector<ObjDetectInfo> &objInfos,
                                     ImageInfo &imgInfo) const;

    // retrieve this specific config parameters
    APP_ERROR ReadConfigParams();

 private:
    int m_nHMWidth_ = 208;
    int m_nHMHeight_ = 208;
    int m_nKeyCounts_ = 5;
    // max top in model
    int m_nTopKN_ = 200;
    // max object per image find
    int maxPerImg_ = 400;
    // IOU thresh hold
    float iouThresh_ = 0.5;
    // [0]  use normal resize to do image preprocess , resize into center
    // [1]  use affine transform to do image resize
    // [-1] resize at left-top corner , no move
    int m_iUseAffineTransform_ = 0;
    int m_isUseSoftNms_ = 1;
};

#endif  // OFFICIAL_CV_CENTERFACE_INFER_MXBASE_MXCENTERFACEPOSTPROCESSOR_H_
