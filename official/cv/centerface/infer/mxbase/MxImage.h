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

#ifndef OFFICIAL_CV_CENTERFACE_INFER_MXBASE_MXIMAGE_H_
#define OFFICIAL_CV_CENTERFACE_INFER_MXBASE_MXIMAGE_H_

#include <assert.h>

#include <string>

#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/MemoryHelper/MemoryHelper.h"
#include "acl/acl.h"
#include "opencv2/opencv.hpp"


#define OPTMIZE_SCALE 0.999

class CDvppImage {
 public:
    explicit CDvppImage(MxBase::DvppWrapper *dvpp) : m_pDvppWrapper(dvpp) {}
    ~CDvppImage() { dispose(); }

    operator bool() { return m_oImageInfo.data != nullptr; }

    void dispose() {
        if (*this) {
            aclrtFree(&m_oImageInfo.data);
            m_oImageInfo = MxBase::DvppDataInfo();
        }
    }

    APP_ERROR Load(const std::string &image);

    APP_ERROR Save(const std::string &image, uint32_t level = 0);
    uint32_t Width() const { return m_oImageInfo.width; }
    uint32_t Height() const { return m_oImageInfo.height; }
    CDvppImage Preprocess(uint32_t width, uint32_t height, const char *format);

 private:
    MxBase::DvppDataInfo m_oImageInfo;
    MxBase::DvppWrapper *m_pDvppWrapper;
};

class CVImage {
 public:
    operator bool() const { return !m_oImage.empty(); }
    APP_ERROR Load(const std::string &image) {
        m_oImage = cv::imread(image);
        return m_oImage.empty() ? -1 : APP_ERR_OK;
    }
    APP_ERROR Save(const std::string &image) {
        if (!*this)  // not load yet
            return -1;
        return cv::imwrite(image, m_oImage) ? APP_ERR_OK : -1;
    }
    uint32_t Width() const { return m_oImage.cols; }
    uint32_t Height() const { return m_oImage.rows; }
    CVImage Preprocess(uint32_t width, uint32_t height,
                       const std::string &color, double &scale,
                       bool isCenter = true);
    CVImage WarpAffinePreprocess(uint32_t width, uint32_t height,
                                 const std::string &color);
    CVImage ConvertToDeviceFormat(aclDataType ty = ACL_UINT8,
                                  aclFormat format = ACL_FORMAT_NHWC,
                                  cv::Scalar *means = nullptr,
                                  cv::Scalar *stds = nullptr);
    // Memory should always been allocated before this call
    // If model need float input , means & stds can be used to do normalize for
    // each pixel channel value
    bool FetchToDevice(MxBase::MemoryData &data, aclDataType ty = ACL_UINT8,
                       aclFormat format = ACL_FORMAT_NHWC,
                       cv::Scalar *means = nullptr, cv::Scalar *stds = nullptr);
    // @mark: image buffer should be continuous
    void *FetchImageBuf() { return m_oImage.data; }
    size_t FetchImageBytes() const {
        return m_oImage.total() * m_oImage.elemSize();
    }

    void DrawBox(float x0, float y0, float x1, float y1, float score);

    static void NormalizeImg(cv::Mat &img, cv::Scalar means, cv::Scalar stds);

    static cv::Mat GetAffineTransform(uint32_t width, uint32_t height,
                                      uint32_t dstWidth, uint32_t dstHeight,
                                      bool invert = false);
    /*
       @param:  transform   affine transform matrix used for image convert
       @param:  x0   source x-axis , input & output
       @param:  y0   source y-axis , input & output
    */
    static void AffineTransform(cv::Mat &transform, float &x0, float &y0);

 private:
    cv::Mat m_oImage;
};

#endif  // OFFICIAL_CV_CENTERFACE_INFER_MXBASE_MXIMAGE_H_
