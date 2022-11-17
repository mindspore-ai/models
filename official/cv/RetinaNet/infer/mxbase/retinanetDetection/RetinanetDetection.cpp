/*
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "RetinanetDetection.h"

#include <unistd.h>
#include <sys/stat.h>
#include <map>
#include <algorithm>
#include <fstream>

#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"

APP_ERROR RetinanetDetection::Init(const InitParam &initParam) {
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
        LogError << "DvppWrapper Init failed, ret=" << ret << ".";
        return ret;
    }
    model_ = std::make_shared<MxBase::ModelInferenceProcessor>();
    ret = model_->Init(initParam.modelPath, modelDesc_);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInferenceProcessor Init failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR RetinanetDetection::DeInit() {
    dvppWrapper_->DeInit();
    model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

APP_ERROR RetinanetDetection::read_image(const std::string &imgPath, cv::Mat &imageMat) {
    imageMat = cv::imread(imgPath, cv::IMREAD_COLOR);
    return APP_ERR_OK;
}

APP_ERROR RetinanetDetection::inference(const std::vector<MxBase::TensorBase> &inputs,
                                        std::vector<MxBase::TensorBase> &outputs) {
    auto dtypes = model_->GetOutputDataType();
    for (size_t i = 0; i < modelDesc_.outputTensors.size(); ++i) {
        std::vector<uint32_t> shape = {};
        for (size_t j = 0; j < modelDesc_.outputTensors[i].tensorDims.size(); ++j) {
            shape.push_back((uint32_t) modelDesc_.outputTensors[i].tensorDims[j]);
        }
        MxBase::TensorBase tensor(shape, dtypes[i], MxBase::MemoryData::MemoryType::MEMORY_DEVICE, deviceId_);
        APP_ERROR ret = MxBase::TensorBase::TensorBaseMalloc(tensor);
        if (ret != APP_ERR_OK) {
            LogError << "TensorBaseMalloc failed, ret=" << ret << ".";
            return ret;
        }
        outputs.push_back(tensor);
    }
    MxBase::DynamicInfo dynamicInfo = {};
    // Set the type to static batch
    dynamicInfo.dynamicType = MxBase::DynamicType::STATIC_BATCH;
    auto startTime = std::chrono::high_resolution_clock::now();
    APP_ERROR ret = model_->ModelInference(inputs, outputs, dynamicInfo);
    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    g_infer_cost.push_back(costMs);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR RetinanetDetection::get_tensor_output(size_t index, MxBase::TensorBase output,
                                             InitParam &initParam) {
    // get inference result from output
    std::vector<std::vector<uint32_t>> dimList = {
            {initParam.numRetinanetBoxes, initParam.boxDim},
            {initParam.numRetinanetBoxes, initParam.classNum},
    };

    // check tensor is available
    std::vector<uint32_t> outputShape = output.GetShape();
    uint32_t len = outputShape.size();
    for (uint32_t i = 0; i < len; ++i) {
        LogInfo << "output" << index << " shape dim " << i << " is: " << outputShape[i] << std::endl;
    }

    LogInfo << "image height : " << initParam.height;
    LogInfo << "image width : " << initParam.width;
    float *outputPtr = reinterpret_cast<float *>(output.GetBuffer());

    uint32_t C = dimList[index][0];  // row
    uint32_t H = dimList[index][1];  // col
    std::vector<float> outputVec;

    for (size_t c = 0; c < C; c++) {
        for (size_t h = 0; h < H; h++) {
            float value = *(outputPtr + c * H + h);
            outputVec.push_back(value);
        }
    }
    cv::Mat outputs = cv::Mat(outputVec).reshape(0, C).clone();
    std::vector<float> outputList;
    if (index == 0) {
        for (int i = 0; i < outputs.rows; i++) {
            for (int j = 0; j < outputs.cols; j++) {
                outputList.push_back(outputs.at<float>(i, j));
            }
            initParam.boxes.push_back(outputList);
            outputList.clear();
        }
    }
    if (index == 1) {
        for (int i = 0; i < outputs.rows; i++) {
            for (int j = 0; j < outputs.cols; j++) {
                outputList.push_back(outputs.at<float>(i, j));
            }
            initParam.scores.push_back(outputList);
            outputList.clear();
        }
    }
    LogInfo << "initParam.boxes size  " << initParam.boxes.size();
    std::string buffer;
    std::ifstream inputFile(initParam.labelPath);
    if (!inputFile) {
        LogInfo << "coco.names file pen error" << std::endl;
        return 0;
    }
    while (getline(inputFile, buffer)) {
        initParam.label.push_back(buffer);
    }

    if (index == 1) {
        int ret = get_anm_result(initParam);
        if (ret != APP_ERR_OK) {
            LogError << "get_anm_result Init failed, ret=" << ret << ".";
            return ret;
        }
    }
    return APP_ERR_OK;
}

APP_ERROR RetinanetDetection::get_anm_result(InitParam &initParam) {
    // get anm result from output
    std::vector<std::vector<std::vector<float>>> final_boxes;
    std::vector<std::vector<std::string>> final_label;
    std::vector<std::vector<float>> final_score;

    for (unsigned int c = 1; c < initParam.classNum; ++c) {
        int ret;
        std::vector<float> class_box_scores;
        std::vector<float> class_box_scores_new;
        std::vector<int> score_mask;
        std::vector<std::string> class_box_label;
        std::vector<std::vector<float>> class_boxes;
        std::vector<std::vector<float>> class_boxes_final;
        ret = get_column_data(class_box_scores, initParam.scores, static_cast<int>(c));
        if (ret != APP_ERR_OK) {
            LogError << "get_column_data failed, ret=" << ret << ".";
            return ret;
        }
        for (unsigned int k = 0; k < class_box_scores.size(); ++k) {
            if (class_box_scores[k] > initParam.minScore) {
                score_mask.push_back(k);
            }
        }
        for (unsigned int k = 0; k < score_mask.size(); ++k) {
            class_box_scores_new.push_back(class_box_scores[score_mask[k]]);
            class_boxes.push_back(initParam.boxes[score_mask[k]]);
        }
        for (unsigned int l = 0; l < class_boxes.size(); ++l) {
            for (unsigned int m = 0; m < class_boxes[l].size(); ++m) {
                if (m % 2 == 0) class_boxes[l][m] *= initParam.height;
                else
                    class_boxes[l][m] *= initParam.width;
            }
        }
        class_box_scores.clear();
        if (score_mask.size() > 0) {
            std::vector<int> nms_index;
            ret = apply_nms(class_boxes, class_box_scores_new, nms_index, initParam);
            if (ret != APP_ERR_OK) {
                LogError << "apply_nms failed, ret=" << ret << ".";
                return ret;
            }
            for (unsigned int j = 0; j < nms_index.size(); ++j) {
                float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
                x1 = class_boxes[nms_index[j]][1];
                y1 = class_boxes[nms_index[j]][0];
                x2 = class_boxes[nms_index[j]][3] - class_boxes[nms_index[j]][1];
                y2 = class_boxes[nms_index[j]][2] - class_boxes[nms_index[j]][0];
                std::vector<float> box_end{x1, y1, x2, y2};
                class_boxes_final.push_back(box_end);
                LogInfo << "class_boxes : " << x1 << " " << y1 << " " << x2 << " " << y2;
                class_box_scores.push_back(class_box_scores_new[nms_index[j]]);
                LogInfo << "class_box_scores : " << class_box_scores_new[nms_index[j]];
                class_box_label.push_back(initParam.label[c]);
                LogInfo << "class_box_label : " << initParam.label[c];
                box_end.clear();
            }
            final_boxes.push_back(class_boxes_final);
            final_score.push_back(class_box_scores);
            final_label.push_back(class_box_label);
            nms_index.clear();
        }
        class_box_scores.clear();
        class_box_scores_new.clear();
        score_mask.clear();
        class_box_label.clear();
        class_boxes.clear();
        class_boxes_final.clear();
    }

    int ret = write_result(final_boxes, final_score, final_label);
    final_boxes.clear();
    final_score.clear();
    final_label.clear();
    if (ret != APP_ERR_OK) {
        LogError << "write_result failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR RetinanetDetection::write_result(std::vector<std::vector<std::vector<float>>> final_boxes,
                                           std::vector<std::vector<float>> final_score,
                                           std::vector<std::vector<std::string>> final_label) {
    // save the result to a file
    std::string resultPathName = "./result";

    // create result directory when it does not exit
    if (access(resultPathName.c_str(), 0) != 0) {
        int ret = mkdir(resultPathName.c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
        if (ret != 0) {
            LogError << "Failed to create result directory: " << resultPathName << ", ret = " << ret;
            return APP_ERR_COMM_OPEN_FAIL;
        }
    }
    // create result file under result directory
    resultPathName = resultPathName + "/output.txt";

    std::ofstream tfile(resultPathName, std::ofstream::app);
    if (tfile.fail()) {
        LogError << "Failed to open result file: " << resultPathName;
        return APP_ERR_COMM_OPEN_FAIL;
    }

    for (unsigned int j = 0; j < final_score.size(); j++) {
        for (unsigned int k = 0; k < final_score[j].size(); k++) {
            tfile << "bbox:" << final_boxes[j][k][0] << " " << final_boxes[j][k][1] << " " << final_boxes[j][k][2]
                  << " " << final_boxes[j][k][3] << " " << "score:" << final_score[j][k]
                  << " " << "category:" << final_label[j][k] << std::endl;
        }
    }
    tfile.close();

    return APP_ERR_OK;
}

APP_ERROR RetinanetDetection::apply_nms(std::vector<std::vector<float>> &class_boxes,
                                        std::vector<float> &class_box_scores, std::vector<int> &keep,
                                        const InitParam &initParam) {
    // apply nms to get index
    std::vector<float> y1;
    std::vector<float> x1;
    std::vector<float> y2;
    std::vector<float> x2;
    int ret = get_column_data(y1, class_boxes, 0);
    if (ret != APP_ERR_OK) {
        LogError << "get_column_data failed, ret=" << ret << ".";
        return ret;
    }
    ret = get_column_data(x1, class_boxes, 1);
    if (ret != APP_ERR_OK) {
        LogError << "get_column_data failed, ret=" << ret << ".";
        return ret;
    }
    ret = get_column_data(y2, class_boxes, 2);
    if (ret != APP_ERR_OK) {
        LogError << "get_column_data failed, ret=" << ret << ".";
        return ret;
    }
    ret = get_column_data(x2, class_boxes, 3);
    if (ret != APP_ERR_OK) {
        LogError << "get_column_data failed, ret=" << ret << ".";
        return ret;
    }

    std::vector<float> areas;
    for (unsigned int i = 0; i < class_boxes.size(); ++i) {
        areas.push_back((x2[i] - x1[i] + 1) * (y2[i] - y1[i] + 1));
    }
    std::vector<int> order;
    std::vector<float> class_box_scores_order = class_box_scores;
    for (unsigned int i = 0; i < class_box_scores.size(); ++i) {
        int maxPosition = max_element(class_box_scores_order.begin(),
                                      class_box_scores_order.end()) - class_box_scores_order.begin();
        order.push_back(maxPosition);
        class_box_scores_order[maxPosition] = 0;
    }
    while (order.size() > 0) {
        int i = order[0];
        keep.push_back(i);
        if (keep.size() >= initParam.maxBoxes) {
            break;
        }
        std::vector<float> yy1, xx1, yy2, xx2;
        ret = maxiMum(y1[i], y1, order, yy1);
        if (ret != APP_ERR_OK) {
            LogError << "maxiMum failed, ret=" << ret << ".";
            return ret;
        }
        ret = maxiMum(x1[i], x1, order, xx1);
        if (ret != APP_ERR_OK) {
            LogError << "maxiMum failed, ret=" << ret << ".";
            return ret;
        }
        ret = miniMum(y2[i], y2, order, yy2);
        if (ret != APP_ERR_OK) {
            LogError << "miniMum failed, ret=" << ret << ".";
            return ret;
        }
        ret = miniMum(x2[i], x2, order, xx2);
        if (ret != APP_ERR_OK) {
            LogError << "miniMum failed, ret=" << ret << ".";
            return ret;
        }

        std::vector<float> inter;
        for (unsigned int j = 0; j < xx1.size(); ++j) {
            float w, h;
            w = (xx2[j] - xx1[j] + 1) > 0.0 ? xx2[j] - xx1[j] + 1: 0.0;
            h = (yy2[j] - yy1[j] + 1) > 0.0 ? yy2[j] - yy1[j] + 1: 0.0;
            inter.push_back(w * h);
        }
        ret = get_order_data(areas, inter, order, initParam);
        if (ret != APP_ERR_OK) {
            LogError << "get_order_data failed, ret=" << ret << ".";
            return ret;
        }
    }
    return APP_ERR_OK;
}

APP_ERROR RetinanetDetection::get_order_data(const std::vector<float> &areas, const std::vector<float> &inter,
                                             std::vector<int> &order, const InitParam &initParam) {
    int i = order[0];
    std::vector<float> ovr;
    for (unsigned int j = 1; j < order.size(); ++j) {
        ovr.push_back(inter[j - 1] / (areas[i] + areas[order[j]] - inter[j - 1]));
    }
    std::vector<int> inds;
    for (unsigned int j = 0; j < ovr.size(); ++j) {
        if (ovr[j] <= initParam.nmsThershold) {
            inds.push_back(j);
        }
    }
    std::vector<int> order_new;
    for (unsigned int j = 0; j < inds.size(); ++j) {
        order_new.push_back(order[inds[j] + 1]);
    }
    order.swap(order_new);
    return APP_ERR_OK;
}

APP_ERROR RetinanetDetection::get_column_data(std::vector<float> &get_vector,
                                              std::vector<std::vector<float>> &input_vector, int index) {
    for (unsigned int i = 0; i < input_vector.size(); ++i) {
        get_vector.push_back(input_vector[i][index]);
    }
    return APP_ERR_OK;
}

APP_ERROR RetinanetDetection::maxiMum(float x, std::vector<float> &other_x,
                                      std::vector<int> &order, std::vector<float> &get_x) {
    for (unsigned int i = 1; i < order.size(); ++i) {
        if (x > other_x[order[i]]) {
            get_x.push_back(x);
        } else {
            get_x.push_back(other_x[order[i]]);
        }
    }
    return APP_ERR_OK;
}

APP_ERROR RetinanetDetection::miniMum(float x, std::vector<float> &other_x,
                                      std::vector<int> &order, std::vector<float> &get_x) {
    for (unsigned int i = 1; i < order.size(); ++i) {
        if (x < other_x[order[i]]) {
            get_x.push_back(x);
        } else {
            get_x.push_back(other_x[order[i]]);
        }
    }
    return APP_ERR_OK;
}

APP_ERROR RetinanetDetection::post_process(std::vector<MxBase::TensorBase> outputs, InitParam &initParam) {
    // post process
    for (size_t index = 0; index < outputs.size(); index++) {
        APP_ERROR ret = outputs[index].ToHost();
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "tohost fail.";
            return ret;
        }
        get_tensor_output(index, outputs[index], initParam);
    }
    return APP_ERR_OK;
}

APP_ERROR RetinanetDetection::cvmat_to_tensorbase(const cv::Mat &imageMat, MxBase::TensorBase &tensorBase) {
    const uint32_t dataSize = imageMat.cols * imageMat.rows * MxBase::YUV444_RGB_WIDTH_NU;
    MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(imageMat.data, dataSize, MxBase::MemoryData::MEMORY_HOST_MALLOC);

    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }
    std::vector<uint32_t> shape = {1, MxBase::YUV444_RGB_WIDTH_NU, static_cast<uint32_t>(imageMat.rows),
                                   static_cast<uint32_t>(imageMat.cols)};
    tensorBase = MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_FLOAT32);
    return APP_ERR_OK;
}

APP_ERROR RetinanetDetection::resize(cv::Mat &srcImageMat, cv::Mat &dstImageMat, const InitParam &initParam) {
    uint32_t resizeHeight = initParam.resizeHeight;
    uint32_t resizeWidth = initParam.resizeWidth;
    cv::resize(srcImageMat, dstImageMat, cv::Size(resizeWidth, resizeHeight));
    return APP_ERR_OK;
}

APP_ERROR RetinanetDetection::process(const std::string &imgName, InitParam &initParam) {
    cv::Mat imageMat;
    APP_ERROR ret = read_image(imgName, imageMat);
    if (ret != APP_ERR_OK) {
        LogError << "read_image failed, ret=" << ret << ".";
        return ret;
    }
    initParam.width = imageMat.cols;
    initParam.height = imageMat.rows;
    ret = resize(imageMat, imageMat, initParam);
    if (ret != APP_ERR_OK) {
        LogError << "resize failed, ret=" << ret << ".";
        return ret;
    }

    // preprocess a photo to tensor
    std::vector<MxBase::TensorBase> inputs = {};
    std::vector<MxBase::TensorBase> outputs = {};
    MxBase::TensorBase tensorBase;
    ret = cvmat_to_tensorbase(imageMat, tensorBase);
    if (ret != APP_ERR_OK) {
        LogError << "cvmat_to_tensorbase failed, ret=" << ret << ".";
        return ret;
    }

    inputs.push_back(tensorBase);
    ret = inference(inputs, outputs);
    if (ret != APP_ERR_OK) {
        LogError << "inference failed, ret=" << ret << ".";
        return ret;
    }

    ret = post_process(outputs, initParam);
    if (ret != APP_ERR_OK) {
        LogError << "post_process failed, ret=" << ret << ".";
        return ret;
    }

    imageMat.release();
    return APP_ERR_OK;
}
