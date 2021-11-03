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
#include <boost/property_tree/json_parser.hpp>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include "acl/acl.h"
#include "CenterNetMindsporePost.h"
#include "MxBase/CV/ObjectDetection/Nms/Nms.h"

namespace {
// Output Tensor
const int OUTPUT_TENSOR_SIZE = 1;
const int OUTPUT_BBOX_SIZE = 3;
const int OUTPUT_BBOX_TWO_INDEX_SHAPE = 6;
const int OUTPUT_BBOX_INDEX = 0;
// index
const int YUV_DE = 2;
const int YUV_NU = 4;
const int BBOX_INDEX_LX = 0;
const int BBOX_INDEX_LY = 1;
const int BBOX_INDEX_RX = 2;
const int BBOX_INDEX_RY = 3;
const int BBOX_INDEX_SCORE = 4;
const int BBOX_INDEX_CLASS = 5;
const int BBOX_INDEX_SCALE_NUM = 6;
float coco_class_nameid[80] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                       22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                       46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
                       67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90};
}  // namespace

namespace MxBase {

CenterNetMindsporePost &CenterNetMindsporePost::operator=(const CenterNetMindsporePost &other) {
    if (this == &other) {
        return *this;
    }
    ObjectPostProcessBase::operator=(other);
    return *this;
}

APP_ERROR CenterNetMindsporePost::ReadConfigParams() {
    APP_ERROR ret = configData_.GetFileValue<uint32_t>("CLASS_NUM", classNum_);
    if (ret != APP_ERR_OK) {
        LogWarn << GetError(ret) << "No CLASS_NUM in config file, default value(" << classNum_ << ").";
    }

    ret = configData_.GetFileValue<uint32_t>("RPN_MAX_NUM", rpnMaxNum_);
    if (ret != APP_ERR_OK) {
        LogWarn << GetError(ret) << "No RPN_MAX_NUM in config file, default value(" << rpnMaxNum_ << ").";
    }


    LogInfo << "The config parameters of post process are as follows: \n";
    LogInfo << " CLASS_NUM: " << classNum_;
    LogInfo << " RPN_MAX_NUM: " << rpnMaxNum_;
    return APP_ERR_OK;
}

APP_ERROR CenterNetMindsporePost::Init(const std::map<std::string, std::shared_ptr<void>> &postConfig) {
    LogInfo << "Begin to initialize CenterNetMindsporePost.";
    APP_ERROR ret = ObjectPostProcessBase::Init(postConfig);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Fail to superinit  in ObjectPostProcessBase.";
        return ret;
    }

    ReadConfigParams();
    LogInfo << "End to initialize CenterNetMindsporePost.";
    return APP_ERR_OK;
}

APP_ERROR CenterNetMindsporePost::DeInit() {
    LogInfo << "Begin to deinitialize CenterNetMindsporePost.";
    LogInfo << "End to deinitialize CenterNetMindsporePost.";
    return APP_ERR_OK;
}

bool CenterNetMindsporePost::IsValidTensors(const std::vector<TensorBase> &tensors) const {
    if (tensors.size() < OUTPUT_TENSOR_SIZE) {
        LogError << "The number of tensor (" << tensors.size() << ") is less than required (" << OUTPUT_TENSOR_SIZE
                 << ")";
        return false;
    }

    auto bboxShape = tensors[OUTPUT_BBOX_INDEX].GetShape();
    if (bboxShape.size() != OUTPUT_BBOX_SIZE) {
        LogError << "The number of tensor[" << OUTPUT_BBOX_INDEX << "] dimensions (" << bboxShape.size()
                 << ") is not equal to (" << OUTPUT_BBOX_SIZE << ")";
        return false;
    }

    if (bboxShape[VECTOR_SECOND_INDEX] != rpnMaxNum_) {
        LogError << "The output tensor is mismatched: " << rpnMaxNum_ << "/" << bboxShape[VECTOR_SECOND_INDEX] << ").";
        return false;
    }

    if (bboxShape[VECTOR_THIRD_INDEX] != OUTPUT_BBOX_TWO_INDEX_SHAPE) {
        LogError << "The number of bbox[" << VECTOR_THIRD_INDEX << "] dimensions (" << bboxShape[VECTOR_THIRD_INDEX]
                 << ") is not equal to (" << OUTPUT_BBOX_TWO_INDEX_SHAPE << ")";
        return false;
    }
    return true;
}

void CenterNetMindsporePost::Resize_Affine(const cv::Mat &srcDet, cv::Mat &dstDet,
                                            const ResizedImageInfo &resizedImageInfos) {
    int new_width, new_height, width, height;
    float ss = static_cast<float>(YUV_DE);
    new_height = static_cast<int>(floor(resizedImageInfos.heightResize / YUV_NU));
    new_width = static_cast<int>(floor(resizedImageInfos.widthResize / YUV_NU));
    width = static_cast<int>(resizedImageInfos.widthOriginal);
    height = static_cast<int>(resizedImageInfos.heightOriginal);

    cv::Point2f srcPoint2f[3], dstPoint2f[3];
    int max_h_w = std::max(static_cast<int>(resizedImageInfos.widthOriginal),
                           static_cast<int>(resizedImageInfos.heightOriginal));
    srcPoint2f[0] = cv::Point2f(static_cast<float>(width / ss), static_cast<float>(height / ss));
    srcPoint2f[1] = cv::Point2f(static_cast<float>(width / ss),
                                static_cast<float>((height - max_h_w) / ss));
    srcPoint2f[2] = cv::Point2f(static_cast<float>((width - max_h_w) / ss),
                                static_cast<float>((height - max_h_w) / ss));
    dstPoint2f[0] = cv::Point2f(static_cast<float>(new_width) / ss, static_cast<float>(new_height) / ss);
    dstPoint2f[1] = cv::Point2f(static_cast<float>(new_width) / ss, 0.0);
    dstPoint2f[2] = cv::Point2f(0.0, 0.0);

    cv::Mat warp_mat(2, 3, CV_32FC1);
    warp_mat = cv::getAffineTransform(dstPoint2f, srcPoint2f);
    dstDet = warp_mat;
}

void CenterNetMindsporePost::affine_transform(const cv::Mat &A, const cv::Mat &B, cv::Mat &dst) {
    float sum = 0;
    for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j < B.cols; ++j) {
            for (int k = 0; k < A.cols; ++k) {
                double s, l;
                s = A.at<double>(i, k);
                l = B.at<float>(k, j);
                sum += s * l;
                dst.at<float>(i, j) = sum;
            }
            sum = 0;
        }
    }
}

void CenterNetMindsporePost::soft_nms(cv::Mat &src, int s, const float sigma,
                                      const float Nt, const float threshold) {
    for (int i = 0; i < s; i++) {
        float tx1, tx2, ty1, ty2, ts, maxscore;
        int pos, maxpos;
        maxscore = src.at<float>(i, YUV_NU);
        maxpos = i;
        tx1 = src.at<float>(i, BBOX_INDEX_LX);
        ty1 = src.at<float>(i, BBOX_INDEX_LY);
        tx2 = src.at<float>(i, BBOX_INDEX_RX);
        ty2 = src.at<float>(i, BBOX_INDEX_RY);
        ts = src.at<float>(i, BBOX_INDEX_SCORE);
        pos = i + 1;
        // get max box
        while (pos < s) {
            float ss = src.at<float>(i, YUV_NU);
            if (maxscore < ss) {
                maxscore = ss;
                maxpos = pos;
            }
            pos = pos + 1;
        }
        // add max box as a detection
        src.at<float>(i, BBOX_INDEX_LX) = src.at<float>(maxpos, BBOX_INDEX_LX);
        src.at<float>(i, BBOX_INDEX_LY) = src.at<float>(maxpos, BBOX_INDEX_LY);
        src.at<float>(i, BBOX_INDEX_RX) = src.at<float>(maxpos, BBOX_INDEX_RY);
        src.at<float>(i, BBOX_INDEX_RY) = src.at<float>(maxpos, BBOX_INDEX_RY);
        src.at<float>(i, BBOX_INDEX_SCORE) = src.at<float>(maxpos, BBOX_INDEX_SCORE);

        // swap ith box with position of max box
        src.at<float>(maxpos, BBOX_INDEX_LX) = tx1;
        src.at<float>(maxpos, BBOX_INDEX_LY) = ty1;
        src.at<float>(maxpos, BBOX_INDEX_RX) = tx2;
        src.at<float>(maxpos, BBOX_INDEX_RY) = ty2;
        src.at<float>(maxpos, BBOX_INDEX_SCORE) = ts;

        tx1 = src.at<float>(i, BBOX_INDEX_LX);
        ty1 = src.at<float>(i, BBOX_INDEX_LY);
        tx2 = src.at<float>(i, BBOX_INDEX_RX);
        ty2 = src.at<float>(i, BBOX_INDEX_RY);
        ts = src.at<float>(i, BBOX_INDEX_SCORE);
        pos = i +1;
        // NMS iterations, note that N changes if detection boxes fall below threshold
        while (pos < s) {
            float x1, x2, y1, y2, area, iw;
            x1 = src.at<float>(pos, BBOX_INDEX_LX);
            y1 = src.at<float>(pos, BBOX_INDEX_LY);
            x2 = src.at<float>(pos, BBOX_INDEX_RX);
            y2 = src.at<float>(pos, BBOX_INDEX_RY);

            area = (x2 - x1 + 1) * (y2 - y1 + 1);
            iw = (std::min(tx2, x2) - std::max(tx1, x1) + 1);
            if (iw > 0) {
                float ih;
                ih = (std::min(ty2, y2) - std::max(ty1, y1) + 1);
                if (ih > 0) {
                    float weight, ov, ua;
                    ua = static_cast<float>((tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area - iw * ih);
                    ov = iw * ih / ua;  // iou between max box and detection box
                    // gaussian
                    weight = std::exp(-(ov * ov)/sigma);
                    src.at<float>(pos, YUV_NU) = weight * (src.at<float>(pos, YUV_NU));
                    // if box score falls below threshold, discard the box by swapping with last box
                    // updata s
                    if ((src.at<float>(pos, YUV_NU)) < threshold) {
                        float ss1 = s - 1;
                        src.at<float>(pos, BBOX_INDEX_LX) = src.at<float>(ss1, BBOX_INDEX_LX);
                        src.at<float>(pos, BBOX_INDEX_LY) = src.at<float>(ss1, BBOX_INDEX_LY);
                        src.at<float>(pos, BBOX_INDEX_RX) = src.at<float>(ss1, BBOX_INDEX_RX);
                        src.at<float>(pos, BBOX_INDEX_RY) = src.at<float>(ss1, BBOX_INDEX_RY);
                        src.at<float>(pos, BBOX_INDEX_SCORE) = src.at<float>(ss1, BBOX_INDEX_SCORE);
                        s = s - 1;
                        pos = pos - 1;
                    }
                }
            }
            pos = pos + 1;
        }
    }
}

void CenterNetMindsporePost::sort_id(float src[][6], const int sum) {
    for (int k = sum ; k > 0; k--) {
        for (int m = 0; m < k - 1; m++) {
            if (src[m][BBOX_INDEX_CLASS] > src[m+1][BBOX_INDEX_CLASS]) {
                float t0 = src[m][BBOX_INDEX_LX];
                float t1 = src[m][BBOX_INDEX_LY];
                float t2 = src[m][BBOX_INDEX_RX];
                float t3 = src[m][BBOX_INDEX_RY];
                float t4 = src[m][BBOX_INDEX_SCORE];
                float t5 = src[m][BBOX_INDEX_CLASS];
                src[m][BBOX_INDEX_LX] = src[m+1][BBOX_INDEX_LX];
                src[m][BBOX_INDEX_LY] = src[m+1][BBOX_INDEX_LY];
                src[m][BBOX_INDEX_RX] = src[m+1][BBOX_INDEX_RX];
                src[m][BBOX_INDEX_RY] = src[m+1][BBOX_INDEX_RY];
                src[m][BBOX_INDEX_SCORE] = src[m+1][BBOX_INDEX_SCORE];
                src[m][BBOX_INDEX_CLASS] = src[m+1][BBOX_INDEX_CLASS];
                src[m+1][BBOX_INDEX_LX] = t0;
                src[m+1][BBOX_INDEX_LY] = t1;
                src[m+1][BBOX_INDEX_RX] = t2;
                src[m+1][BBOX_INDEX_RY] = t3;
                src[m+1][BBOX_INDEX_SCORE] = t4;
                src[m+1][BBOX_INDEX_CLASS] = t5;
            }
        }
    }
}

void CenterNetMindsporePost::set_nms(float data[][6], int (*p)[2], const int num) {
    int s1 = 0;
    int s2 = 0;
    float sigma = 0.5;
    float Nt = 0.5;
    float threshold = 0.001;
    for (int s = 0; s < num; s++) {
        int r = *(*(p + s) + 1);
        if (r !=0) {
            float class0[r][5];
            for (int t = 0; t < r; t++) {
                class0[t][BBOX_INDEX_LX] = data[s1][BBOX_INDEX_LX];
                class0[t][BBOX_INDEX_LY] = data[s1][BBOX_INDEX_LY];
                class0[t][BBOX_INDEX_RX] = data[s1][BBOX_INDEX_RX];
                class0[t][BBOX_INDEX_RY] = data[s1][BBOX_INDEX_RY];
                class0[t][BBOX_INDEX_SCORE] = data[s1][BBOX_INDEX_SCORE];
                s1++;
            }
            cv::Mat class1(r, 5, CV_32FC1, (reinterpret_cast<float*>(class0)));
            soft_nms(class1, r, sigma, Nt, threshold);
            // output and transfer data after soft_nms
            for (int u = 0; u < r; u++) {
                data[s2][BBOX_INDEX_LX] = class1.at<float>(u, BBOX_INDEX_LX);
                data[s2][BBOX_INDEX_LY] = class1.at<float>(u, BBOX_INDEX_LY);
                data[s2][BBOX_INDEX_RX] = class1.at<float>(u, BBOX_INDEX_RX);
                data[s2][BBOX_INDEX_RY] = class1.at<float>(u, BBOX_INDEX_RY);
                data[s2][BBOX_INDEX_SCORE] = class1.at<float>(u, BBOX_INDEX_SCORE);
                s2++;
            }
        }
    }
}

void CenterNetMindsporePost::GetValidDetBoxes(const std::vector<TensorBase> &tensors, std::vector<DetectBox> &detBoxes,
                                               const ResizedImageInfo &resizedImageInfos, uint32_t batchNum) {
    LogInfo << "Begin to GetValidDetBoxes.";
    auto *bboxPtr = reinterpret_cast<float *>(GetBuffer(tensors[OUTPUT_BBOX_INDEX], batchNum));  // 1 * 100 *6
    size_t total = rpnMaxNum_;
    int tol = rpnMaxNum_;
    int cnum = classNum_;
    float first[100][6] = {};
    float det0[100][2] = {};
    float det1[100][2] = {};
    std::string cName[100] = {};
    int i = 0;
    for (size_t index = 0; index < total; ++index) {
        size_t startIndex = index * BBOX_INDEX_SCALE_NUM;
        first[i][BBOX_INDEX_LX]  = bboxPtr[startIndex + BBOX_INDEX_LX];
        first[i][BBOX_INDEX_LY]  = bboxPtr[startIndex + BBOX_INDEX_LY];
        first[i][BBOX_INDEX_RX]  = bboxPtr[startIndex + BBOX_INDEX_RX];
        first[i][BBOX_INDEX_RY]  = bboxPtr[startIndex + BBOX_INDEX_RY];
        first[i][BBOX_INDEX_SCORE]  = bboxPtr[startIndex + BBOX_INDEX_SCORE];
        first[i][BBOX_INDEX_CLASS]  = bboxPtr[startIndex + BBOX_INDEX_CLASS];
        det0[i][0] = bboxPtr[startIndex + BBOX_INDEX_LX];
        det0[i][1] = bboxPtr[startIndex + BBOX_INDEX_LY];
        det1[i][0] = bboxPtr[startIndex + BBOX_INDEX_RX];
        det1[i][1] = bboxPtr[startIndex + BBOX_INDEX_RY];
        i += 1;
    }
    cv::Mat Det0(100, 2, CV_32FC1, (reinterpret_cast<float*>(det0)));
    cv::Mat Det1(100, 2, CV_32FC1, (reinterpret_cast<float*>(det1)));
    cv::Mat Dst0(2, 3, CV_32FC1);
    Resize_Affine(Det0, Dst0, resizedImageInfos);
    // bbox affine
    cv::Mat D0 = cv::Mat::ones(3, 1, CV_32FC1);
    cv::Mat D1 = cv::Mat::ones(3, 1, CV_32FC1);
    cv::Mat Dst1(2, 1, CV_32FC1);
    cv::Mat Dst2(2, 1, CV_32FC1);
    for (int a = 0; a < tol; a++) {
        D0.at<float>(0, 0) = first[a][BBOX_INDEX_LX];
        D0.at<float>(0, 1) = first[a][BBOX_INDEX_LY];
        D1.at<float>(0, 0) = first[a][BBOX_INDEX_RX];
        D1.at<float>(0, 1) = first[a][BBOX_INDEX_RY];
        affine_transform(Dst0, D0, Dst1);
        affine_transform(Dst0, D1, Dst2);
        float X1 = Dst1.at<float>(0, 0);
        float Y1 = Dst1.at<float>(1, 0);
        float X2 = Dst2.at<float>(0, 0);
        float Y2 = Dst2.at<float>(1, 0);
        first[a][BBOX_INDEX_LX] = X1;
        first[a][BBOX_INDEX_LY] = Y1;
        first[a][BBOX_INDEX_RX] = X2;
        first[a][BBOX_INDEX_RY] = Y2;
    }
    sort_id(first, tol);
    int class_id[cnum][2];  // save class_id and number
    for (int i0 = 0; i0 < cnum; i0++) {
        class_id[i0][0] = i0;
        class_id[i0][1] = 0;
    }
    for (int a0 = 0; a0 < tol; a0++) {
        int c0 = 0;
        int id1 = static_cast<int>(first[a0][BBOX_INDEX_CLASS]);
        while (c0 < cnum) {
            if (id1 == c0) {
                class_id[c0][1]++;
            }
            c0++;
        }
    }
    int (*p)[2];
    p = class_id;
    set_nms(first, p, cnum);
    // use new class_names replace old class_names
    for (int d = 0; d < tol; d++) {
        int id0 = static_cast<int>(first[d][5]);
        first[d][5] = coco_class_nameid[id0];
        cName[d] = configData_.GetClassName(id0);;
    }
    for ( int f = 0; f < tol; f++ ) {
        float XX1 = first[f][BBOX_INDEX_LX];
        float YY1 = first[f][BBOX_INDEX_LY];
        float XX2 = first[f][BBOX_INDEX_RX];
        float YY2 = first[f][BBOX_INDEX_RY];
        XX2 -= XX1;
        YY2 -= YY1;
        MxBase::DetectBox detBox;
        detBox.x = (XX1 + XX2) / COORDINATE_PARAM;
        detBox.y = (YY1 + YY2) / COORDINATE_PARAM;  // COORDINATE_PARAM = 2
        detBox.width = XX2 - XX1;
        detBox.height = YY2 - YY1;
        detBox.prob = first[f][BBOX_INDEX_SCORE];
        detBox.classID = first[f][BBOX_INDEX_CLASS];
        detBox.className = cName[f];
        detBoxes.push_back(detBox);
    }
}

void CenterNetMindsporePost::ConvertObjInfoFromDetectBox(std::vector<DetectBox> &detBoxes,
                                                          std::vector<ObjectInfo> &objectInfos,
                                                          const ResizedImageInfo &resizedImageInfo) {
     for (auto &detBoxe : detBoxes) {
        if (detBoxe.classID < 0) {
            continue;
        }
        ObjectInfo objInfo = {};
        objInfo.classId = static_cast<float>(detBoxe.classID);
        objInfo.className = detBoxe.className;
        objInfo.confidence = detBoxe.prob;

        objInfo.x0 = static_cast<float>(detBoxe.x - detBoxe.width / COORDINATE_PARAM);
        objInfo.y0 = static_cast<float>(detBoxe.y - detBoxe.height / COORDINATE_PARAM);
        objInfo.x1 = static_cast<float>(detBoxe.x + detBoxe.width / COORDINATE_PARAM);
        objInfo.y1 = static_cast<float>(detBoxe.y + detBoxe.height / COORDINATE_PARAM);
        objectInfos.push_back(objInfo);
    }
}

void CenterNetMindsporePost::ObjectDetectionOutput(const std::vector<TensorBase> &tensors,
                                                    std::vector<std::vector<ObjectInfo>> &objectInfos,
                                                    const std::vector<ResizedImageInfo> &resizedImageInfos) {
    LogDebug << "CenterNetMindsporePost start to write results.";
    auto shape = tensors[OUTPUT_BBOX_INDEX].GetShape();
    uint32_t batchSize = shape[0];
    for (uint32_t i = 0; i < batchSize; ++i) {
        std::vector<MxBase::DetectBox> detBoxes;
        std::vector<ObjectInfo> objectInfo;
        GetValidDetBoxes(tensors, detBoxes, resizedImageInfos[i], i);
        ConvertObjInfoFromDetectBox(detBoxes, objectInfo, resizedImageInfos[i]);
        objectInfos.push_back(objectInfo);
    }
    LogDebug << "CenterNetMindsporePost write results succeeded.";
}

APP_ERROR CenterNetMindsporePost::Process(const std::vector<TensorBase> &tensors,
                                           std::vector<std::vector<ObjectInfo>> &objectInfos,
                                           const std::vector<ResizedImageInfo> &resizedImageInfos,
                                           const std::map<std::string, std::shared_ptr<void>> &configParamMap) {
    LogDebug << "Begin to process CenterNetMindsporePost.";
    auto inputs = tensors;
    APP_ERROR ret = CheckAndMoveTensors(inputs);
    if (ret != APP_ERR_OK) {
        LogError << "CheckAndMoveTensors failed, ret=" << ret;
        return ret;
    }
    ObjectDetectionOutput(inputs, objectInfos, resizedImageInfos);
    LogInfo << "End to process CenterNetMindsporePost.";
    return APP_ERR_OK;
}

extern "C" {
std::shared_ptr<MxBase::CenterNetMindsporePost> GetObjectInstance() {
    LogInfo << "Begin to get CenterNetMindsporePost instance.";
    auto instance = std::make_shared<CenterNetMindsporePost>();
    LogInfo << "End to get CenterNetMindsporePost Instance";
    return instance;
}
}

}  // namespace MxBase
