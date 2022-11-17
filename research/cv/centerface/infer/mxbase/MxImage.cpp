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
#include <vector>
#include <string>
#include <algorithm>
#include "MxImage.h"

APP_ERROR CDvppImage::Load(const std::string &image) {
    dispose();
    std::string file(image);
    return m_pDvppWrapper->DvppJpegDecode(file, m_oImageInfo);
}

APP_ERROR CDvppImage::Save(const std::string &image, uint32_t level) {
    if (!*this) return -1;

    std::size_t npos = image.rfind('/');
    std::string dirname;
    if (npos == std::string::npos)
        dirname = ".";
    else
        dirname = image.substr(0, npos);
    std::string filename = basename(image.c_str());
    npos = filename.rfind('.');
    if (npos != std::string::npos) filename.resize(npos);

    return m_pDvppWrapper->DvppJpegEncode(m_oImageInfo, dirname, filename,
                                          level);
}

CDvppImage CDvppImage::Preprocess(uint32_t width, uint32_t height,
                                  const char *format) {
    MxBase::CropRoiConfig crop = {0, 0, 0, 0};
    crop.x1 = m_oImageInfo.width - 1;
    crop.y1 = m_oImageInfo.height - 1;
    if (crop.x1 % 2 == 0) crop.x1--;
    if (crop.y1 % 2 == 0) crop.y1--;

    std::vector<MxBase::DvppDataInfo> outImages;
    MxBase::ResizeConfig resizeConfig;
    resizeConfig.width = width;
    resizeConfig.height = height;
    resizeConfig.scale_x = .5;
    resizeConfig.scale_y = .5;

    CDvppImage dvppImage(m_pDvppWrapper);
    outImages.resize(1);
    if (m_pDvppWrapper->VpcResize(m_oImageInfo, outImages[0], resizeConfig) ==
        APP_ERR_OK) {
        dvppImage.m_oImageInfo = outImages[0];
    }
    return dvppImage;
}

CVImage CVImage::Preprocess(uint32_t width, uint32_t height,
                            const std::string &color, double &scale,
                            bool isCenter) {
    double scale_x = static_cast<double>(width) / (m_oImage.cols);
    double scale_y = static_cast<double>(height) / (m_oImage.rows);

    uint32_t resizeWidth, resizeHeight;
    if (scale_x <= scale_y) {  // scale by width
        scale = scale_x;
        resizeWidth = width;
        resizeHeight = (uint32_t)((m_oImage.rows) * scale);
    } else {  // scale by height
        scale = scale_y;
        resizeHeight = height;
        resizeWidth = (uint32_t)((m_oImage.cols) * scale);
    }
    cv::Mat dstImage(width, height, m_oImage.type(), cv::Scalar(0));
    cv::Mat roi;
    if (isCenter)
        roi = dstImage(cv::Rect((width - m_oImage.cols * scale) / 2.0,
                                (height - m_oImage.rows * scale) / 2.0,
                                resizeWidth, resizeHeight));
    else
        roi = dstImage(cv::Rect(0, 0, resizeWidth, resizeHeight));

    cv::resize(m_oImage, roi, roi.size());
    if (color.empty() || color == "bgr") {
    } else if (color == "rgb") {
        cv::cvtColor(dstImage, dstImage, cv::COLOR_BGR2RGB);
    } else if (color == "yuv") {
        cv::cvtColor(dstImage, dstImage, cv::COLOR_BGR2YUV);
    } else {
        return CVImage();
    }
    CVImage outImage;
    outImage.m_oImage = dstImage;
    return outImage;
}

CVImage CVImage::WarpAffinePreprocess(uint32_t width, uint32_t height,
                                      const std::string &color) {
    cv::Mat trans_input =
        GetAffineTransform(m_oImage.cols, m_oImage.rows, width, height);
    cv::Mat dstImage;

    cv::resize(
        m_oImage, dstImage,
        cv::Size(m_oImage.cols * OPTMIZE_SCALE, m_oImage.rows * OPTMIZE_SCALE));
    cv::warpAffine(dstImage, dstImage, trans_input, cv::Size(width, height));
    if (color.empty() || color == "bgr") {
    } else if (color == "rgb") {
        cv::cvtColor(dstImage, dstImage, cv::COLOR_BGR2RGB);
    } else if (color == "yuv") {
        cv::cvtColor(dstImage, dstImage, cv::COLOR_BGR2YUV);
    } else {
        return CVImage();
    }
    CVImage outImage;
    outImage.m_oImage = dstImage;
    return outImage;
}

CVImage CVImage::ConvertToDeviceFormat(aclDataType ty, aclFormat format,
                                       cv::Scalar *means, cv::Scalar *stds) {
    cv::Mat convert = m_oImage;
    if (ty == ACL_UINT8) {
        if (format == ACL_FORMAT_NCHW)
            convert = cv::dnn::blobFromImage(m_oImage, 1.0, cv::Size(),
                                             cv::Scalar(), false, false, CV_8U);
    } else if (ty == ACL_FLOAT16) {
        if (format == ACL_FORMAT_NCHW) {
            convert = cv::dnn::blobFromImage(m_oImage, 1.0, cv::Size(),
                                             cv::Scalar(), false, false, CV_8U);
            convert.convertTo(convert, CV_16U);
        } else if (format == ACL_FORMAT_NHWC) {
            m_oImage.convertTo(convert, CV_16U);
        } else {
            assert(0);
        }
    } else if (ty == ACL_FLOAT) {
        if (format == ACL_FORMAT_NCHW) {
            convert = cv::dnn::blobFromImage(m_oImage, 1 / 255.0);
        } else if (format == ACL_FORMAT_NHWC) {
            m_oImage.convertTo(convert, CV_32F, 1 / 255.0);
        } else {
            assert(0);
        }
        if (means && stds) NormalizeImg(convert, *means, *stds);
        // NormalizeImg(convert, means, stds);
    } else {
        assert(0);
    }
    if (!convert.isContinuous()) convert = convert.clone();

    CVImage result;
    result.m_oImage = convert;
    return result;
}

bool CVImage::FetchToDevice(MxBase::MemoryData &data, aclDataType ty,
                            aclFormat format, cv::Scalar *means,
                            cv::Scalar *stds) {
    CVImage convert = ConvertToDeviceFormat(ty, format, means, stds);
    if (!convert) {
        return false;
    }
    assert(convert.FetchImageBytes() == data.size);
    MxBase::MemoryData tmp(convert.FetchImageBuf(), data.size);
    MxBase::MemoryHelper::MxbsMemcpy(data, tmp, tmp.size);
    return true;
}

void CVImage::DrawBox(float x0, float y0, float x1, float y1, float score) {
    cv::Rect rect(x0, y0, x1 - x0, y1 - y0);
    // cv::Rect rect(247.2, 373.3, 139.9, 183.6);
    cv::Scalar color(255, 0, 0);
    cv::rectangle(m_oImage, rect, color, 1);
    std::string labelText(std::to_string(score));
    cv::putText(m_oImage, labelText,
                cv::Point(x0 + 5, std::max<int>(y0 - 5, 0)), 2, 0.5, color, 1);
}

void CVImage::NormalizeImg(cv::Mat &img, cv::Scalar means, cv::Scalar stds) {
    // bgr
    assert(img.dims == 4 && img.size[0] == 1 && img.type() == CV_32F);
    size_t plane = img.step1(1);
    // first channel
    float *ptr = reinterpret_cast<float *>(img.ptr(0));
    for (size_t i = 0; i < plane; i++) {
        *ptr = (*ptr - means[0]) / stds[0];
        ptr++;
    }
    // second channel
    for (size_t i = 0; i < plane; i++) {
        *ptr = (*ptr - means[1]) / stds[1];
        ptr++;
    }
    // third channel
    for (size_t i = 0; i < plane; i++) {
        *ptr = (*ptr - means[2]) / stds[2];
        ptr++;
    }
}

cv::Mat CVImage::GetAffineTransform(uint32_t width, uint32_t height,
                                    uint32_t dstWidth, uint32_t dstHeight,
                                    bool invert) {
    cv::Point2f src[3], dst[3];

    uint32_t new_width = uint32_t(width * OPTMIZE_SCALE);
    uint32_t new_height = uint32_t(height * OPTMIZE_SCALE);

    float srcw =
        width > height ? static_cast<float>(width) : static_cast<float>(height);

    src[0] = {static_cast<float>(new_width) * 0.5f,
              static_cast<float>(new_height) * 0.5f};
    dst[0] = {static_cast<float>(dstWidth) * 0.5f,
              static_cast<float>(dstHeight) * 0.5f};

    src[1] = src[0] + cv::Point2f(0, srcw * -0.5f);
    dst[1] = dst[0] + cv::Point2f(0, dstWidth * -0.5f);

    cv::Point2f direct = src[0] - src[1];
    src[2] = src[1] + cv::Point2f(-direct.y, direct.x);

    direct = dst[0] - dst[1];
    dst[2] = dst[1] + cv::Point2f(-direct.y, direct.x);
    return !invert ? cv::getAffineTransform(src, dst)
                   : cv::getAffineTransform(dst, src);
}

void CVImage::AffineTransform(cv::Mat &transform, float &x0, float &y0) {
    assert(transform.dims == 2 && transform.size[0] == 2 &&
           transform.size[1] == 3);
    assert(transform.type() == CV_64F);
    double *ptr = reinterpret_cast<double *>(transform.data);
    x0 = ptr[0] * x0 + ptr[1] * y0 + ptr[2];
    y0 = ptr[3] * x0 + ptr[4] * y0 + ptr[5];

    x0 /= OPTMIZE_SCALE;
    y0 /= OPTMIZE_SCALE;
}
