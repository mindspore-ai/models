/*
 * Copyright(C) 2021. Huawei Technologies Co.,Ltd. All rights reserved.
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

#include <math.h>
#include <utility>
#include "SsdGhost_MindsporePost.h"
#include "MxBase/Log/Log.h"

namespace {
    const int LEFTTOPY  = 0;
    const int LEFTTOPX  = 1;
    const int RIGHTBOTY = 2;
    const int RIGHTBOTX = 3;
}
namespace MxBase {
    float SsdGhostPostProcess::CalcIou(DetectBox boxA, DetectBox boxB, IOUMethod method) {
        float left = std::max(boxA.x - boxA.width / 2.f, boxB.x - boxB.width / 2.f);
        float right = std::min(boxA.x + boxA.width / 2.f, boxB.x + boxB.width / 2.f);
        float top = std::max(boxA.y - boxA.height / 2.f, boxB.y - boxB.height / 2.f);
        float bottom = std::min(boxA.y + boxA.height / 2.f, boxB.y + boxB.height / 2.f);
        if (top > bottom || left > right) {
            return 0.0f;
        }
        // intersection / union
        float area = (right - left + 1) * (bottom - top + 1);
        if (method == IOUMethod::MAX) {
            return area / std::max(boxA.width * boxA.height, boxB.width * boxB.height);
        }
        if (method == IOUMethod::MIN) {
            return area / std::min(boxA.width * boxA.height, boxB.width * boxB.height);
        }
        return area / ((boxA.width + 1) * (boxA.height + 1)  + (boxB.width + 1) * (boxB.height + 1) - area);
    }

    void SsdGhostPostProcess::FilterByIou(std::vector<DetectBox> dets,
                                                    std::vector<DetectBox>& sortBoxes,
                                                    float iouThresh, IOUMethod method) {
        int count = 0;
        for (unsigned int m = 0; m < dets.size(); ++m) {
            auto& item = dets[m];
            sortBoxes.push_back(item);
            count++;
            if (count >= maxBboxPerClass_) {
                break;
            }
            for (unsigned int n = m + 1; n < dets.size(); ++n) {
                if (CalcIou(item, dets[n], method) > iouThresh) {
                    dets.erase(dets.begin() + n);
                    --n;
                }
            }
        }
    }

    void SsdGhostPostProcess::NmsSort(std::vector<DetectBox>& detBoxes, float iouThresh, IOUMethod method) {
        std::vector<DetectBox> sortBoxes;
        std::map<int, std::vector<DetectBox>> resClassMap;
        for (const auto& item : detBoxes) {
            resClassMap[item.classID].push_back(item);
        }

        std::map<int, std::vector<DetectBox>>::iterator iter;
        for (iter = resClassMap.begin(); iter != resClassMap.end(); ++iter) {
            std::sort(iter->second.begin(), iter->second.end(), [=](const DetectBox& a, const DetectBox& b) {
                return a.prob < b.prob;
            });
            std::reverse(iter->second.begin(), iter->second.end());
            FilterByIou(iter->second, sortBoxes, iouThresh, method);
        }
        detBoxes = std::move(sortBoxes);
    }
    SsdGhostPostProcess &
    SsdGhostPostProcess::operator=(const SsdGhostPostProcess &other) {
        if (this == &other) {
            return *this;
        }
        ObjectPostProcessBase::operator=(other);
        return *this;
    }

    APP_ERROR SsdGhostPostProcess::Init(const std::map<std::string, std::shared_ptr<void>> &postConfig) {
        LogDebug << "Start to Init SsdGhostPostProcess.";
        ObjectPostProcessBase::Init(postConfig);
        APP_ERROR ret = configData_.GetFileValue<int>("OBJECT_NUM", objectNum_);
        if (ret != APP_ERR_OK) {
            LogWarn << GetError(ret) << "Fail to read OBJECT_NUM from config, default value(" << DEFAULT_OBJECT_NUM
                    << ") will be used as objectNum_.";
        }
        ret = configData_.GetFileValue<float>("IOU_THRESH", iouThresh_);
        if (ret != APP_ERR_OK) {
            LogWarn << GetError(ret) << "Fail to read iouThresh_ from config, default value(" << DEFAULT_IOU_THRESH
                    << ") will be used as iouThresh_.";
        }
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "Fail to superInit in ObjectPostProcessBase.";
            return ret;
        }
        LogDebug << "End to Init SsdGhostPostProcess.";
        return APP_ERR_OK;
    }

    APP_ERROR SsdGhostPostProcess::DeInit() {
        return APP_ERR_OK;
    }


    bool SsdGhostPostProcess::IsValidTensors(const std::vector<TensorBase> &tensors) const {
        auto shape = tensors[0].GetShape();
        if (tensors.size() < VECTOR_THIRD_INDEX) {
            LogError << "number of tensors (" << tensors.size() << ") " << "is less than required ("
                     << VECTOR_THIRD_INDEX << ")";
            return false;
        }
        if (shape.size() != VECTOR_FOURTH_INDEX) {
            LogError << "number of tensor[0] dimensions (" << shape.size() << ") " << "is not equal to ("
                     << VECTOR_FOURTH_INDEX << ")";
            return false;
        }
        if (shape[VECTOR_SECOND_INDEX] != (uint32_t)objectNum_) {
            LogError << "dimension of tensor[0][1] (" << shape[VECTOR_SECOND_INDEX] << ") " << "is not equal to ("
                     << objectNum_ << ")";
            return false;
        }
        if (shape[VECTOR_THIRD_INDEX] != BOX_DIM) {
            LogError << "dimension of tensor[0][2] (" << shape[VECTOR_THIRD_INDEX] << ") " << "is not equal to ("
                     << BOX_DIM << ")";
            return false;
        }
        shape = tensors[1].GetShape();
        if (shape.size() != VECTOR_FOURTH_INDEX) {
            LogError << "number of tensor[1] dimensions (" << shape.size() << ") " << "is not equal to ("
                     << VECTOR_FOURTH_INDEX << ")";
            return false;
        }
        if (shape[VECTOR_SECOND_INDEX] != (uint32_t)objectNum_) {
            LogError << "dimension of tensor[1][1] (" << shape[VECTOR_SECOND_INDEX] << ") " << "is not equal to ("
                     << objectNum_ << ")";
            return false;
        }
        if (shape[VECTOR_THIRD_INDEX] != (uint32_t)classNum_) {
            LogError << "dimension of tensor[1][2] (" << shape[VECTOR_THIRD_INDEX] << ") " << "is not equal to ("
                     << classNum_ << ")";
            return false;
        }
        return true;
    }

    void PredictBoxDecode(float decodeBoxs[][4], float *objectBboxPtr) {
        float boxes[1917][4];
        float defaultBoxes[1917][4];
        int dIndex = 0;
        float scales[] = { 0.2, 0.35, 0.5, 0.65, 0.8, 0.95, 1.0 };
        float fk[] = { 18.75, 9.375, 4.6875, 3.0, 2.0, 1.0 };
        int featureSizeLen = 6;
        int featureSize[6] = {19, 10, 5, 3, 2, 1};
        int aspectRatios[6][2] = {{2, 0}, {2, 3}, {2, 3}, {2, 3}, {2, 3}, {2, 3}};

        for (int i = 0; i < featureSizeLen; i++) {
            float sk1 = scales[i];
            float sk2 = scales[i + 1];
            float sk3 = sqrt(sk1 * sk2);

            int allSizeLen = 6;
            if (i == 0) allSizeLen = 3;
            float allSizes[allSizeLen][2];

            if (i == 0) {
                float w = sk1 * sqrt(2);
                float h = sk1 / sqrt(2);
                allSizes[0][0] = 0.1;
                allSizes[0][1] = 0.1;
                allSizes[1][0] = w;
                allSizes[1][1] = h;
                allSizes[2][0] = h;
                allSizes[2][1] = w;
            } else {
                allSizes[0][0] = sk1;
                allSizes[0][1] = sk1;
                int index = 0;

                for (int j = 0; j < 2; j++) {
                    int aspectRatio = aspectRatios[i][j];
                    float w = sk1 * sqrt(aspectRatio);
                    float h = sk1 / sqrt(aspectRatio);

                    allSizes[++index][0] = w;
                    allSizes[index][1] = h;
                    allSizes[++index][0] = h;
                    allSizes[index][1] = w;
                }

                allSizes[++index][0] = sk3;
                allSizes[index][1] = sk3;
            }

            for (int ii = 0; ii < featureSize[i]; ii++) {
                for (int jj = 0; jj < featureSize[i]; jj++) {
                    for (int k = 0; k < allSizeLen; k++) {
                        float cx = (jj + 0.5) / fk[i];
                        float cy = (ii + 0.5) / fk[i];
                        defaultBoxes[dIndex][0] = cy;
                        defaultBoxes[dIndex][1] = cx;
                        defaultBoxes[dIndex][2] = allSizes[k][1];
                        defaultBoxes[dIndex++][3] = allSizes[k][0];
                    }
                }
            }
        }

        for (int i = 0; i < 1917; i++) {
            boxes[i][0] = objectBboxPtr[i * BOX_DIM + 0] * 0.1 * defaultBoxes[i][2] + defaultBoxes[i][0];
            boxes[i][1] = objectBboxPtr[i * BOX_DIM + 1] * 0.1 * defaultBoxes[i][3] + defaultBoxes[i][1];
            boxes[i][2] = exp(objectBboxPtr[i * BOX_DIM + 2] * 0.2) * defaultBoxes[i][2];
            boxes[i][3] = exp(objectBboxPtr[i * BOX_DIM + 3] * 0.2) * defaultBoxes[i][3];
        }

        for (int i = 0; i < 1917; i++) {
            decodeBoxs[i][0] = boxes[i][0] - boxes[i][2] / 2;
            decodeBoxs[i][1] = boxes[i][1] - boxes[i][3] / 2;
            decodeBoxs[i][2] = boxes[i][0] + boxes[i][2] / 2;
            decodeBoxs[i][3] = boxes[i][1] + boxes[i][3] / 2;
        }

        for (int i = 0; i < 1917; i++) {
            for (int j = 0; j < 4; j++) {
                if (decodeBoxs[i][j] > 1) decodeBoxs[i][j] = 1;
                if (decodeBoxs[i][j] < 0) decodeBoxs[i][j] = 0;
            }
        }
    }

    void SsdGhostPostProcess::NonMaxSuppression(std::vector<MxBase::DetectBox>& detBoxes,
                                                          TensorBase &bboxTensor, TensorBase &confidenceTensor,
                                                          int stride, const ResizedImageInfo &imgInfo,
                                                          uint32_t batchNum, uint32_t batchSize) {
        // Find the class with largest confidence except background.

        float *objectBboxPtr = reinterpret_cast<float*>(bboxTensor.GetBuffer()) +
                batchNum * bboxTensor.GetByteSize() / batchSize;
        float *objectConfidencePtr = reinterpret_cast<float*>(confidenceTensor.GetBuffer()) +
                batchNum * confidenceTensor.GetByteSize() / batchSize;
        for (int k = 1; k < static_cast<int>(classNum_); k++) {
            for (int j = 0; j < stride; j++) {
                if (objectConfidencePtr[j * classNum_ + k] > scoreThresh_) {
                    MxBase::DetectBox det;
                    det.classID = k;
                    det.prob = objectConfidencePtr[j * classNum_ + k];

                    float bboxes[1917][4] = {0};
                    PredictBoxDecode(bboxes, objectBboxPtr);

                    float x1 = bboxes[j][LEFTTOPX] * imgInfo.widthOriginal;
                    float y1 = bboxes[j][LEFTTOPY] * imgInfo.heightOriginal;
                    float x2 = bboxes[j][RIGHTBOTX] * imgInfo.widthOriginal;
                    float y2 = bboxes[j][RIGHTBOTY] * imgInfo.heightOriginal;
                    det.x = (x1 + x2) / COORDINATE_PARAM;
                    det.y = (y1 + y2) / COORDINATE_PARAM;
                    det.width = (x2 - x1 > 0) ? (x2 - x1) : (x1 - x2);
                    det.height = (y2 - y1 > 0) ? (y2 - y1) : (y1 - y2);

                    detBoxes.emplace_back(det);
                }
            }
        }

        NmsSort(detBoxes, iouThresh_, UNION);
    }

    void SsdGhostPostProcess::ObjectDetectionOutput(const std::vector<TensorBase> &tensors,
                                                              std::vector<std::vector<ObjectInfo>> &objectInfos,
                                                              const std::vector<ResizedImageInfo> &resizedImageInfos) {
        LogDebug << "SsdGhostPostProcess start to write results.";
        auto shape = tensors[objectConfidenceTensor_].GetShape();
        auto tensor0 = tensors[objectBboxTensor_];
        auto tensor1 = tensors[objectConfidenceTensor_];
        uint32_t batchSize = shape[0];

        for (uint32_t i = 0; i < batchSize; i++) {
            std::vector<MxBase::DetectBox> detBoxes;
            NonMaxSuppression(detBoxes, tensor0, tensor1, objectNum_, resizedImageInfos[i], i, batchSize);
            std::vector<ObjectInfo> objectInfos1;
            for (unsigned int k = 0; k < detBoxes.size(); k++) {
                ObjectInfo objInfo;
                objInfo.classId = detBoxes[k].classID;
                objInfo.confidence = detBoxes[k].prob;
                objInfo.x0 = (static_cast<float>(detBoxes[k].x - detBoxes[k].width / COORDINATE_PARAM) > 0) ?
                             static_cast<float>((detBoxes[k].x - detBoxes[k].width / COORDINATE_PARAM)) : 0;

                objInfo.y0 = (static_cast<float>(detBoxes[k].y - detBoxes[k].height / COORDINATE_PARAM) > 0) ?
                             static_cast<float>((detBoxes[k].y - detBoxes[k].height / COORDINATE_PARAM)) : 0;

                objInfo.x1 = ((detBoxes[k].x + detBoxes[k].width / COORDINATE_PARAM) <=
                              resizedImageInfos[i].widthOriginal) ?
                             static_cast<float>((detBoxes[k].x + detBoxes[k].width / COORDINATE_PARAM)) :
                             resizedImageInfos[i].widthOriginal;

                objInfo.y1 = ((detBoxes[k].y + detBoxes[k].height / COORDINATE_PARAM) <=
                              resizedImageInfos[i].heightOriginal) ?
                             static_cast<float>((detBoxes[k].y + detBoxes[k].height / COORDINATE_PARAM)) :
                             resizedImageInfos[i].heightOriginal;
                objInfo.className = configData_.GetClassName(objInfo.classId);

                objectInfos1.push_back(objInfo);
            }

            objectInfos.push_back(objectInfos1);
        }
        LogDebug << "SsdGhostPostProcess write results succeessed.";
    }

    APP_ERROR SsdGhostPostProcess::Process(const std::vector<TensorBase> &tensors,
                                                     std::vector<std::vector<ObjectInfo>> &objectInfos,
                                                     const std::vector<ResizedImageInfo> &resizedImageInfos,
                                                     const std::map<std::string, std::shared_ptr<void>>
                                                     &configParamMap) {
        LogDebug << "Start to Process SsdGhostPostProcess.";
        APP_ERROR ret = APP_ERR_OK;
        auto inputs = tensors;
        ret = CheckAndMoveTensors(inputs);
        if (ret != APP_ERR_OK) {
            LogError << "MoveTensorsAndCheck failed. ret=" << ret;
            return ret;
        }

        ObjectDetectionOutput(inputs, objectInfos, resizedImageInfos);
        LogDebug << "End to Process SsdGhostPostProcess.";
        return APP_ERR_OK;
    }

    extern "C" {
    std::shared_ptr<MxBase::SsdGhostPostProcess> GetObjectInstance() {
        LogInfo << "Begin to get SsdGhostPostProcess instance.";
        auto instance = std::make_shared<SsdGhostPostProcess>();
        LogInfo << "End to get SsdGhostPostProcess instance.";
        return instance;
    }
    }
}  // namespace MxBase
