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
#include "PsenetDetection.h"
#include<vector>
#include<algorithm>
#include <queue>
#include <utility>
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"
APP_ERROR PsenetDetection::Init(const InitParam &initParam) {
    deviceId_ = initParam.deviceId;
    APP_ERROR ret = MxBase::DeviceManager::GetInstance()->InitDevices();
    if (ret != APP_ERR_OK) {
        LogError << "Init devices failed, ret=" << ret << ".";
        return ret;
    }
    ret = MxBase::TensorContext::GetInstance()->SetContext(initParam.deviceId);
    if (ret != APP_ERR_OK) {
        LogError << "Set context failed, ret = " << ret << ".";
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
    MxBase::ConfigData configData;
    const std::string checkTensor = initParam.checkTensor ? "true" : "false";

    configData.SetJsonValue("KERNEL_NUM", std::to_string(initParam.kernelNum));
    configData.SetJsonValue("PSE_SCALE", std::to_string(initParam.pseScale));
    configData.SetJsonValue("MIN_KERNEL_AREA", std::to_string(initParam.minKernelArea));
    configData.SetJsonValue("MIN_SCORE", std::to_string(initParam.minScore));
    configData.SetJsonValue("MIN_AREA", std::to_string(initParam.minArea));

    configData.SetJsonValue("CHECK_MODEL", checkTensor);
    auto jsonStr = configData.GetCfgJson().serialize();
    std::map<std::string, std::shared_ptr<void>> config;
    config["postProcessConfigContent"] = std::make_shared<std::string>(jsonStr);
    config["labelPath"] = std::make_shared<std::string>(initParam.labelPath);

    post_ = std::make_shared<MxBase::PSENetPostProcess>();
    ret = post_->Init(config);
    if (ret != APP_ERR_OK) {
        LogError << "PSENetPostProcess init failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR PsenetDetection::DeInit() {
    dvppWrapper_->DeInit();
    model_->DeInit();
    post_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

APP_ERROR PsenetDetection::ReadImage(const std::string &imgPath, cv::Mat &imageMat) {
    imageMat = cv::imread(imgPath, cv::IMREAD_COLOR);
    cv::cvtColor(imageMat, imageMat, cv::COLOR_RGB2BGR);
    imageWidth_ = imageMat.cols;
    imageHeight_ = imageMat.rows;
    return APP_ERR_OK;
}
APP_ERROR PsenetDetection::Pad(cv::Mat &srcImageMat, cv::Mat &dstImageMat) {
    int width = srcImageMat.cols, height = srcImageMat.rows;
    if (width > height) {
        cv::copyMakeBorder(srcImageMat, dstImageMat, 0, width - height, 0, 0,
            cv::BorderTypes::BORDER_CONSTANT, cv::Scalar(0, 0, 0));     // 指定常量像素填充
    } else {
        cv::copyMakeBorder(srcImageMat, dstImageMat, 0, 0, 0, height - width,
            cv::BorderTypes::BORDER_CONSTANT, cv::Scalar(0, 0, 0));     // 指定常量像素填充
    }
    cv::imwrite("my.jpg", dstImageMat);
    return APP_ERR_OK;
}
APP_ERROR PsenetDetection::Resize(cv::Mat &srcImageMat, cv::Mat &dstImageMat) {
    static constexpr uint32_t resizeHeight = 1920;
    static constexpr uint32_t resizeWidth = 1920;
    cv::resize(srcImageMat, dstImageMat, cv::Size(resizeWidth, resizeHeight));
    return APP_ERR_OK;
}
APP_ERROR PsenetDetection::CVMatToTensorBase(const cv::Mat &imageMat, MxBase::TensorBase &tensorBase) {
        const uint32_t dataSize = imageMat.cols * imageMat.rows * MxBase::YUV444_RGB_WIDTH_NU;

    MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(imageMat.data, dataSize, MxBase::MemoryData::MEMORY_HOST_MALLOC);
    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }
    std::vector<uint32_t> shape = {imageMat.rows * MxBase::YUV444_RGB_WIDTH_NU, static_cast<uint32_t>(imageMat.cols)};
    tensorBase = MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_UINT8);
    return APP_ERR_OK;
}
void PsenetDetection::TensorBaseToCVMat(std::vector<cv::Mat> &imageMat,
                                             const MxBase::TensorBase &tensor, int type) {
    MxBase::TensorBase Data = tensor;
    auto shape = Data.GetShape();
    int C = 0, W = 0, H = 0;
    if (shape.size() == 4) {
        C = shape[1];
        W = shape[2];
        H = shape[3];
    }
    auto *data = reinterpret_cast<float *>(tensor.GetBuffer());
    for (int c = 0; c < C; c++) {
        cv::Mat kernel;
        if (type == 0) {
            kernel = cv::Mat::zeros(W, H, CV_32FC1);
        } else {
            kernel = cv::Mat::zeros(W, H, CV_8UC1);
        }
        for (int w = 0; w < W; w++) {
            for (int h = 0; h < H; h++) {
                if (type == 0) {
                    kernel.at<float>(w, h) = data[c * (H * W) + w * H + h];
                } else if (type == 1) {
                    kernel.at<unsigned char>(w, h) = (unsigned char)data[c * (H * W) + w * H + h];
                }
            }
        }
        imageMat.push_back(kernel);
    }
}
APP_ERROR PsenetDetection::Inference(const std::vector<MxBase::TensorBase> &inputs,
    std::vector<MxBase::TensorBase> &outputs) {
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
        outputs.push_back(tensor);
    }

    MxBase::DynamicInfo dynamicInfo = {};
    dynamicInfo.dynamicType = MxBase::DynamicType::STATIC_BATCH;
    APP_ERROR ret = model_->ModelInference(inputs, outputs, dynamicInfo);


    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

void PsenetDetection::growing_text_line(const std::vector<cv::Mat> &kernels,
    std::vector<std::vector<int>> *text_line, float min_area) {
    cv::Mat label_mat;
    int label_num = cv::connectedComponents(kernels[kernels.size() - 1], label_mat, 4);

    std::vector<int> area(label_num + 1, 0);
    for (int x = 0; x < label_mat.rows; ++x) {
        for (int y = 0; y < label_mat.cols; ++y) {
            int label = label_mat.at<int>(x, y);
            if (label == 0) continue;
            area[label] += 1;
        }
    }

    std::queue<cv::Point> queue, next_queue;
    for (int x = 0; x < label_mat.rows; ++x) {
        std::vector<int> row(label_mat.cols);
        for (int y = 0; y < label_mat.cols; ++y) {
            int label = label_mat.at<int>(x, y);
            if (label == 0) continue;
            if (area[label] < min_area) continue;
            cv::Point point(x, y);
            queue.push(point);
            row[y] = label;
        }
        text_line->emplace_back(row);
    }

    int dx[] = {-1, 1, 0, 0};
    int dy[] = {0, 0, -1, 1};

    for (int kernel_id = kernels.size() - 2; kernel_id >= 0; --kernel_id) {
        while (!queue.empty()) {
            cv::Point point = queue.front();
            queue.pop();
            int x = point.x;
            int y = point.y;
            int label = text_line->at(x)[y];
            bool is_edge = true;
            for (int d = 0; d < 4; ++d) {
                int tmp_x = x + dx[d];
                int tmp_y = y + dy[d];

                if (tmp_x < 0 || tmp_x >= static_cast<int>(text_line->size())) continue;
                if (tmp_y < 0 || tmp_y >= static_cast<int>(text_line->at(1).size())) continue;
                if (kernels[kernel_id].at<char>(tmp_x, tmp_y) == 0) {continue;}
                if (text_line->at(tmp_x)[tmp_y] > 0) {continue;}

                cv::Point point_tmp(tmp_x, tmp_y);
                queue.push(point_tmp);
                text_line->at(tmp_x)[tmp_y] = label;
                is_edge = false;
            }

            if (is_edge) {
                next_queue.push(point);
            }
        }
        swap(queue, next_queue);
    }
}
int mod(int x, int y) {
    return (x % y + y) % y;
}
bool cmp(const std::pair<cv::Point, int> &point1p, const std::pair<cv::Point, int> &point2p) {
    cv::Point point1 = point1p.first, point2 = point2p.first;

    int degree1 = atan2(point1.y, point1.x) * 45.0 / atan(1.0);
    int degree2 = atan2(point2.y, point2.x) * 45.0 / atan(1.0);
    degree1 = fmod(fmod(-135 - degree1, 360) + 360, 360);
    degree2 = fmod(fmod(-135 - degree2, 360) + 360, 360);
    return degree1 > degree2;
}
APP_ERROR PsenetDetection::PostProcess(std::vector<MxBase::TensorBase> &outputs,
                                             std::vector<std::vector<MxBase::TextObjectInfo>> &objInfos) {
    MxBase::ResizedImageInfo imgInfo;
    imgInfo.widthOriginal = imageWidth_;
    imgInfo.heightOriginal = imageHeight_;
    imgInfo.widthResize = 1920;
    imgInfo.heightResize = 1920;
    imgInfo.resizeType = MxBase::RESIZER_STRETCHING;
    std::vector<MxBase::ResizedImageInfo> imageInfoVec = {};
    imageInfoVec.push_back(imgInfo);
    std::vector<cv::Mat> scores, kernels;
    outputs[0].ToHost();
    outputs[1].ToHost();
    TensorBaseToCVMat(scores, outputs[0], 0);
    TensorBaseToCVMat(kernels, outputs[1], 1);
    cv::Mat score = scores[0];
    std::vector<std::vector<int>> text_line;
    growing_text_line(kernels, &text_line, 5.0);
    int label_num = 0;
    for (unsigned i = 0; i < text_line.size(); i ++) {
        for (unsigned int b : text_line[i]) {
            label_num = std::max(label_num, static_cast<int>(b));
        }
    }
    label_num++;
    double scale = std::max(imageWidth_, imageHeight_) * 1.0 / 1920;

    std::vector<cv::Mat>bboxes;
    for (int i = 1; i < label_num; i++) {
        std::vector<cv::Point> points;
        for (unsigned int x = 0; x < text_line.size(); x++) {
            for (unsigned int y = 0; y < text_line[x].size(); y++) {
                if (text_line[x][y] == i) {
                    points.push_back({static_cast<int>(y), static_cast<int>(x)});
                }
            }
        }
        if (points.size() < 600) {continue;}
        float s = 0.0;
        for (unsigned int j = 0; j < points.size(); j++) {
            int x = points[j].y, y = points[j].x;
            s += score.at<float>(x, y);
        }
        s /= points.size();
        if (s< 0.93) {continue;}
        cv::RotatedRect  rect = cv::minAreaRect(points);
        cv::Mat boxPts, tboxPts;

        cv::boxPoints(rect, boxPts);
        for (int x = 0; x < boxPts.rows; x++) {
            for (int y = 0; y < boxPts.cols; y++) {
                boxPts.at<float>(x, y) = boxPts.at<float>(x, y) * scale;
            }
        }
        boxPts.convertTo(tboxPts, CV_32S);
        bboxes.push_back(tboxPts);
    }
    std::vector<MxBase::TextObjectInfo>objInfo;
    MxBase::TextObjectInfo textObjectInfo;
    for (unsigned int i = 0; i < bboxes.size(); i++) {
        cv::Mat bbox = bboxes[i];
        double centerx = 0.0, centery = 0.0;
        std::vector<cv::Point> points;
        std::vector<std::pair<cv::Point, int>> centerpoints;
        for (int j = 0; j < bbox.rows; j++) {
            centerx += bbox.at<int>(j, 0);
            centery += bbox.at<int>(j, 1);
            cv::Point point(bbox.at<int>(j, 0), bbox.at<int>(j, 1));
            points.push_back(point);
        }
        centerx /= 4;
        centery /= 4;
        for (int j = 0; j < bbox.rows; j++) {
            centerpoints.push_back({cv::Point(bbox.at<int>(j, 0) - centerx,
                bbox.at<int>(j, 1) - centery), j});
        }
        std::sort(centerpoints.begin(), centerpoints.end(), cmp);

        textObjectInfo.x0 = points[centerpoints[0].second].x;
        textObjectInfo.y0 = points[centerpoints[0].second].y;
        textObjectInfo.x1 = points[centerpoints[1].second].x;
        textObjectInfo.y1 = points[centerpoints[1].second].y;
        textObjectInfo.x2 = points[centerpoints[2].second].x;
        textObjectInfo.y2 = points[centerpoints[2].second].y;
        textObjectInfo.x3 = points[centerpoints[3].second].x;
        textObjectInfo.y3 = points[centerpoints[3].second].y;
        objInfo.push_back(textObjectInfo);
    }
    objInfos.push_back(objInfo);

    return APP_ERR_OK;
}

APP_ERROR PsenetDetection::SaveResult(const std::string &imgPath,
    const std::vector<std::vector<MxBase::TextObjectInfo>> &batchTextObjectInfos) {
    LogInfo << "image path" << imgPath;
    std::string file_name = imgPath.substr(imgPath.find_last_of("/") + 1);
    size_t dot = file_name.find_last_of(".");
    std::string resFileName = "res/submit_ic15/res_"  + file_name.substr(0, dot) + ".txt";
    LogInfo << "file path for saving result" << resFileName;

    std::ofstream outfile(resFileName, std::fstream::out);
    uint32_t batchIndex = 0;
    for (auto textObjectInfos : batchTextObjectInfos) {
        for (auto textObjectInfo : textObjectInfos) {
            std::string resultStr = "";
            resultStr += std::to_string(static_cast<int>(textObjectInfo.x3)) + "," +
             std::to_string(static_cast<int>(textObjectInfo.y3)) + "," +
            std::to_string(static_cast<int>(textObjectInfo.x0)) + "," +
            std::to_string(static_cast<int>(textObjectInfo.y0)) + "," +
            std::to_string(static_cast<int>(textObjectInfo.x1)) + "," +
            std::to_string(static_cast<int>(textObjectInfo.y1)) + "," +
            std::to_string(static_cast<int>(textObjectInfo.x2)) + "," +
            std::to_string(static_cast<int>(textObjectInfo.y2));
            outfile << resultStr << std::endl;
        }
        batchIndex++;
    }
    outfile.close();
    return APP_ERR_OK;
}

APP_ERROR PsenetDetection::Process(const std::string &imgPath) {
    // process image
    cv::Mat imageMat;
    APP_ERROR ret = ReadImage(imgPath, imageMat);
    if (ret != APP_ERR_OK) {
        LogError << "ReadImage failed, ret=" << ret << ".";
        return ret;
    }
    ret = Pad(imageMat, imageMat);
    if (ret != APP_ERR_OK) {
        LogError << "Pad failed, ret=" << ret << ".";
        return ret;
    }
    ret = Resize(imageMat, imageMat);
    if (ret != APP_ERR_OK) {
        LogError << "Resize failed, ret=" << ret << ".";
        return ret;
    }

    std::vector<MxBase::TensorBase> inputs = {};
    std::vector<MxBase::TensorBase> outputs = {};
    MxBase::TensorBase tensorBase;
    ret = CVMatToTensorBase(imageMat, tensorBase);
    if (ret != APP_ERR_OK) {
        LogError << "CVMatToTensorBase failed, ret=" << ret << ".";
        return ret;
    }

    inputs.push_back(tensorBase);
    ret = Inference(inputs, outputs);
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }
    std::vector<std::vector<MxBase::TextObjectInfo>> objInfos;
    ret = PostProcess(outputs, objInfos);
    if (ret != APP_ERR_OK) {
        LogError << "PostProcess failed, ret=" << ret << ".";
        return ret;
    }
    ret = SaveResult(imgPath, objInfos);
    if (ret != APP_ERR_OK) {
        LogError << "Save result failed, ret=" << ret << ".";
        return ret;
    }
    imageMat.release();
    return APP_ERR_OK;
}
