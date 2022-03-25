/*
 * Copyright(C) 2022. Huawei Technologies Co.,Ltd.
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

#include <unistd.h>
#include <sys/stat.h>
#include <math.h>
#include <cmath>
#include <iomanip>
#include <string>
#include <memory>
#include <vector>
#include <algorithm>
#include "opencv2/opencv.hpp"
#include "FaceboxesDetection.h"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"
#include "acl/acl.h"

namespace {
    const uint32_t YUV_BYTE_NU = 3;
    const uint32_t YUV_BYTE_DE = 2;

    const uint32_t MODEL_HEIGHT = 2496;
    const uint32_t MODEL_WIDTH = 1056;
    const uint32_t OUT_FORMAT = 54897;

    const uint32_t IMAGE_SIZE[2] = {2496, 1056};
    const int MIN_SIZES[3][3] = {{32, 64, 128}, {256, 0, 0}, {512, 0, 0}};
    const uint32_t STEPS[3] = {32, 64, 128};
    const float VAR[2] = {0.1, 0.2};
    const float ADD_32 = 0.25;
    const float ADD_REMAIN = 0.5;
    const float SCALE[4] = {1056.0, 2496.0, 1056.0, 2496.0};

    const uint32_t BOX_INDEX = 0;
    const uint32_t CONF_INDEX = 1;
    const float IOU_THRESHOLD = 0.05;
    const float NMS_THRESHOLD = 0.4;
}    // namespace


void GetAnchors(std::vector <std::vector <float> > &anchors,
                const int &k,
                const int &first,
                const int &second) {
    int min_size[3] = {MIN_SIZES[k][0], MIN_SIZES[k][1], MIN_SIZES[k][2]};
    float dense_cx[4];
    float dense_cy[4];
    for (int j = 0; j < 3; j++) {
        float s_kx = static_cast<float>(min_size[j]) / IMAGE_SIZE[1];
        float s_ky = static_cast<float>(min_size[j]) / IMAGE_SIZE[0];
        if (min_size[j] > 0) {
            if (min_size[j] == 32) {
                float second_add = static_cast<float>(second);
                float first_add = static_cast<float>(first);
                for (int second_num = 0; second_num < 4; second_num++) {
                    dense_cx[second_num] = static_cast<float>(second_add * STEPS[k]) / IMAGE_SIZE[1];
                    dense_cy[second_num] = static_cast<float>(first_add * STEPS[k]) / IMAGE_SIZE[0];
                    second_add = second_add + ADD_32;
                    first_add = first_add + ADD_32;
                }
                for (int index = 0; index < 4; index++) {
                    for (int veve = 0; veve < 4; veve++) {
                        std::vector<float> temp;
                        temp.push_back(dense_cx[veve]);
                        temp.push_back(dense_cy[index]);
                        temp.push_back(s_kx);
                        temp.push_back(s_ky);
                        anchors.push_back(temp);
                    }
                }
            } else if (min_size[j] == 64) {
                float second_add = static_cast<float>(second);
                float first_add = static_cast<float>(first);
                for (int second_num = 0; second_num < 2; second_num++) {
                    dense_cx[second_num] = static_cast<float>(second_add * STEPS[k]) / IMAGE_SIZE[1];
                    dense_cy[second_num] = static_cast<float>(first_add * STEPS[k]) / IMAGE_SIZE[0];
                    second_add = second_add + ADD_REMAIN;
                    first_add = first_add + ADD_REMAIN;
                }
                for (int index = 0; index < 2; index++) {
                    for (int veve = 0; veve < 2; veve++) {
                        std::vector<float> temp;
                        temp.push_back(dense_cx[veve]);
                        temp.push_back(dense_cy[index]);
                        temp.push_back(s_kx);
                        temp.push_back(s_ky);
                        anchors.push_back(temp);
                    }
                }
            } else {
                float dense_cx0 = (static_cast<float>(second) + ADD_REMAIN) * STEPS[k] / IMAGE_SIZE[1];
                float dense_cy0 = (static_cast<float>(first) + ADD_REMAIN) * STEPS[k] / IMAGE_SIZE[0];
                std::vector<float> temp;
                temp.push_back(dense_cx0);
                temp.push_back(dense_cy0);
                temp.push_back(s_kx);
                temp.push_back(s_ky);
                anchors.push_back(temp);
            }
        }
    }
}


void PriorBox(std::vector <std::vector <float> > &anchors) {
    int feature_maps[3][2];
    for (int i = 0; i < 3; i ++) {
        feature_maps[i][0] = std::ceil(static_cast<float>(IMAGE_SIZE[0]) / STEPS[i]);
        feature_maps[i][1] = std::ceil(static_cast<float>(IMAGE_SIZE[1]) / STEPS[i]);
    }

    for (int k = 0; k < 3; k++) {
        for (int first = 0; first < feature_maps[k][0]; first++) {
            for (int second = 0; second < feature_maps[k][1]; second++) {
                GetAnchors(anchors, k, first, second);
            }
        }
    }
}


void DecodeBox(const std::vector <std::vector <float> > &anchors,
               std::vector<MxBase::TensorBase> &outputs,
               float boxes[][4]) {
    float* chr = nullptr;
    outputs[BOX_INDEX].ToHost();    // data about bbox from dicice to Host
    chr = static_cast<float *>(outputs[BOX_INDEX].GetBuffer());
    for (int i = 0; i < OUT_FORMAT; i++) {
        boxes[i][0] = anchors[i][0] + chr[0 + 4 * i] * VAR[0] * anchors[i][2];
        boxes[i][1] = anchors[i][1] + chr[1 + 4 * i] * VAR[0] * anchors[i][3];
        boxes[i][2] = anchors[i][2] *  exp(chr[2 + 4 * i] * VAR[1]);
        boxes[i][3] = anchors[i][3] *  exp(chr[3 + 4 * i] * VAR[1]);
        boxes[i][0] = boxes[i][0] - boxes[i][2] / 2;
        boxes[i][1] = boxes[i][1] - boxes[i][3] / 2;
        boxes[i][2] = boxes[i][2] + boxes[i][0];
        boxes[i][3] = boxes[i][3] + boxes[i][1];
    }
}


std::vector <int> Argsort(const std::vector <float>& array) {
    const int array_len(array.size());   // array_len = array.size()
    std::vector <int> array_index(array_len, 0);
    for (int i = 0; i < array_len; ++i)
        array_index[i] = i;

    std::sort(array_index.begin(), array_index.end(),
        [&array](int pos1, int pos2) {return (array[pos1] < array[pos2]);});

    return array_index;
}


void SortTOBoxAndScore(std::vector<MxBase::TensorBase> &outputs,
                       float boxes[][4],
                       std::vector <std::vector <float> > &postbox,
                       std::vector <float> &postscore,
                       std::vector <std::vector <float> > &postboxsort,
                       std::vector <float> &postscoresort,
                       std::vector <std::vector <float> > &dets,
                       std::vector <int> &index) {
    float* chr = nullptr;
    outputs[CONF_INDEX].ToHost();     // data about conf from dicice to Host
    chr = static_cast<float *>(outputs[CONF_INDEX].GetBuffer());
    for (int j = 0; j < OUT_FORMAT; j++) {
        if (chr[j * 2 + 1] > IOU_THRESHOLD) {
            index.push_back(j);
        }
    }

    for (int k = 0; k < index.size(); k++) {
        std::vector <float> postboxtemp = {};
        postboxtemp.push_back(boxes[index[k]][0]);
        postboxtemp.push_back(boxes[index[k]][1]);
        postboxtemp.push_back(boxes[index[k]][2]);
        postboxtemp.push_back(boxes[index[k]][3]);
        postbox.push_back(postboxtemp);
        postscore.push_back(chr[index[k] * 2 + 1]);
    }

    std::vector <int> order0 = Argsort(postscore);
    for (int i = 0; i < index.size(); i++) {
        std::vector <float> postboxtemp = {};
        postboxtemp.push_back(postbox[order0[index.size() - 1 - i]][0]);
        postboxtemp.push_back(postbox[order0[index.size() - 1 - i]][1]);
        postboxtemp.push_back(postbox[order0[index.size() - 1 - i]][2]);
        postboxtemp.push_back(postbox[order0[index.size() - 1 - i]][3]);
        postboxsort.push_back(postboxtemp);
        postscoresort.push_back(postscore[order0[index.size() - 1 - i]]);
    }

    for (int i = 0; i < index.size(); i++) {
        std::vector <float> postboxtemp = {};
        postboxtemp.push_back(postboxsort[i][0]);
        postboxtemp.push_back(postboxsort[i][1]);
        postboxtemp.push_back(postboxsort[i][2]);
        postboxtemp.push_back(postboxsort[i][3]);
        postboxtemp.push_back(postscoresort[i]);
        dets.push_back(postboxtemp);
    }
}


void MaxForm(std::vector <float> &x,
             std::vector <int> &order,
             std::vector <float> &max_x,
             int &tempi) {
    for (int i = 1; i < order.size(); i++) {
        if (x[tempi] > x[order[i]]) {
            max_x.push_back(x[tempi]);
        } else {
            max_x.push_back(x[order[i]]);
        }
    }
}


void MinForm(std::vector <float> &x,
             std::vector <int> &order,
             std::vector <float> &max_x,
             int &tempi) {
    for (int i = 1; i < order.size(); i++) {
        if (x[tempi] < x[order[i]]) {
            max_x.push_back(x[tempi]);
        } else {
            max_x.push_back(x[order[i]]);
        }
    }
}


void LenthOfInedx(const std::vector <float> &min_x,
                  const std::vector <float> &max_x,
                  std::vector <int> &order,
                  std::vector <float> &intersect) {
    for (int i = 0; i < order.size() - 1; i++) {
        if ((min_x[i] - max_x[i] + 1) > 0.0) {
            intersect.push_back(min_x[i] - max_x[i] + 1);
        } else {
            intersect.push_back(0.0);
        }
    }
}


void Nms(std::vector <std::vector <float> > &dets,
         std::vector <int> &index,
         std::vector <int> &reserved_boxes) {
    std::vector <float> x1 = {};
    std::vector <float> y1 = {};
    std::vector <float> x2 = {};
    std::vector <float> y2 = {};
    std::vector <float> scores = {};
    std::vector <float> areas = {};
    for (int i = 0; i < index.size(); i++) {
        x1.push_back(dets[i][0]);
        y1.push_back(dets[i][1]);
        x2.push_back(dets[i][2]);
        y2.push_back(dets[i][3]);
        scores.push_back(dets[i][4]);
        areas.push_back((x2[i] - x1[i] + 1) * (y2[i] - y1[i] + 1));
    }
    std::vector <int> order0 = Argsort(scores);
    std::vector <int> order = {};
    for (int i = 0; i < index.size(); i++) {
        order.push_back(order0[index.size() - 1 - i]);
    }
    while (order.size() > 0) {
        int tempi = order[0];
        reserved_boxes.push_back(tempi);
        std::vector <float> max_x1 = {};
        std::vector <float> max_y1 = {};
        std::vector <float> min_x2 = {};
        std::vector <float> min_y2 = {};
        std::vector <float> intersect_w = {};
        std::vector <float> intersect_h = {};
        std::vector <float> intersect_area = {};
        std::vector <int> nms_index = {};
        MaxForm(x1, order, max_x1, tempi);
        MaxForm(y1, order, max_y1, tempi);
        MinForm(x2, order, min_x2, tempi);
        MinForm(y2, order, min_y2, tempi);
        LenthOfInedx(min_x2, max_x1, order, intersect_w);
        LenthOfInedx(min_y2, max_y1, order, intersect_h);

        for (int i = 1; i < order.size(); i++) {
            intersect_area.push_back(intersect_w[i - 1] * intersect_h[i - 1]);
            float ovr = intersect_area[i - 1] / (areas[tempi] + areas[order[i]] - intersect_area[i - 1]);
            if (ovr <= NMS_THRESHOLD) {
                nms_index.push_back(i);
            }
        }

        std::vector <int> ordertemp = {};
        for (int i = 0; i < nms_index.size(); i++) {
            ordertemp.push_back(order[nms_index[i]]);
        }
        order = {};
        for (int i = 0; i < nms_index.size(); i++) {
            order.push_back(ordertemp[i]);
        }
    }
    std::cout << "reserved_boxes.size() is : "<< reserved_boxes.size() << std::endl;
    for (int i = 0; i < reserved_boxes.size(); i++) {
        std::cout << "--------------------reserved_boxes " << reserved_boxes[i] << std::endl;
    }
}


APP_ERROR FaceboxesDetection::Init(const InitParam &initParam) {
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
    return APP_ERR_OK;
}


APP_ERROR FaceboxesDetection::DeInit() {
    dvppWrapper_->DeInit();
    model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}


APP_ERROR FaceboxesDetection::ReadImage(std::string &imgPath, MxBase::DvppDataInfo &output, ImageShape &imgShape) {
    APP_ERROR ret = dvppWrapper_->DvppJpegDecode(imgPath, output);
    if (ret != APP_ERR_OK) {
        LogError << "DvppWrapper DvppJpegDecode failed, ret=" << ret << ".";
        return ret;
    }
    imgShape.width = output.width;
    imgShape.height = output.height;
    return APP_ERR_OK;
}


APP_ERROR FaceboxesDetection::Resize(const MxBase::DvppDataInfo &input, MxBase::TensorBase &outputTensor) {
    MxBase::CropRoiConfig cropRoi = {0, input.width, input.height, 0};
    float ratio = std::min(static_cast<float>(MODEL_WIDTH) / input.width,
                           static_cast<float>(MODEL_HEIGHT) / input.height);
    MxBase::CropRoiConfig pasteRoi = {0, 0, 0, 0};
    LogInfo << "Ratio: " << ratio << " input.width" << input.width << " input.height" << input.height
            << " input.widthStride" << input.widthStride << " input.heightStride" << input.heightStride;

    pasteRoi.x1 = input.width * ratio;
    pasteRoi.y1 = input.height * ratio;

    MxBase::MemoryData memoryData(MODEL_WIDTH * MODEL_HEIGHT * YUV_BYTE_NU / YUV_BYTE_DE,
                                  MxBase::MemoryData::MemoryType::MEMORY_DVPP, deviceId_);
    APP_ERROR ret = MxBase::MemoryHelper::MxbsMalloc(memoryData);
    if (ret != APP_ERR_OK) {
        LogError << "Fail to allocate dvpp memory.";
        MxBase::MemoryHelper::MxbsFree(memoryData);
        return APP_ERR_COMM_INVALID_PARAM;
    }

    ret = MxBase::MemoryHelper::MxbsMemset(memoryData, 0, memoryData.size);
    if (ret != APP_ERR_OK) {
        LogError << "Fail to set 0.";
        MxBase::MemoryHelper::MxbsFree(memoryData);
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
    outputTensor = MxBase::TensorBase(memoryData, false, shape, MxBase::TENSOR_DTYPE_UINT8);
    LogInfo << "Output data height: " << output.height << ", width: " << output.width << ".";
    LogInfo << "Output data widthStride: " << output.widthStride << ", heightStride: " << output.heightStride << "."
            << std::endl;
    return APP_ERR_OK;
}


APP_ERROR FaceboxesDetection::ImagePreprocess(std::string &imgPath) {    // padding and minus means（104 117 123）
    cv::Mat imageMat;
    cv::Mat imageMatpad(IMAGE_SIZE[0], IMAGE_SIZE[1], CV_8UC3, cv::Scalar(104, 117, 123));
    imageMat = cv::imread(imgPath, cv::IMREAD_COLOR);
    for (int i = 0; i < imageMatpad.rows; i++) {
        for (int j = 0; j < imageMatpad.cols; j++) {
            if ((i < imageMat.rows) && (j < imageMat.cols)) {
                imageMatpad.at<cv::Vec3b>(i, j)[0] = imageMat.at<cv::Vec3b>(i, j)[0];
                imageMatpad.at<cv::Vec3b>(i, j)[1] = imageMat.at<cv::Vec3b>(i, j)[1];
                imageMatpad.at<cv::Vec3b>(i, j)[2] = imageMat.at<cv::Vec3b>(i, j)[2];
            }
        }
    }
    imgPath = "../testpad.jpg";
    cv::imwrite(imgPath, imageMatpad);
    std::cout << "done" << std::endl;
    return APP_ERR_OK;
}


APP_ERROR FaceboxesDetection::Inference(const std::vector<MxBase::TensorBase> &inputs,
                                        std::vector<MxBase::TensorBase> &outputs) {
    auto dtypes = model_->GetOutputDataType();
    for (size_t i = 0; i < modelDesc_.outputTensors.size(); ++i) {
        std::vector<uint32_t> shape = {};
        for (size_t j = 0; j < modelDesc_.outputTensors[i].tensorDims.size(); ++j) {
            shape.push_back((uint32_t)modelDesc_.outputTensors[i].tensorDims[j]);
        }
        MxBase::TensorBase tensor(shape, dtypes[i], MxBase::MemoryData::MemoryType::MEMORY_DEVICE, deviceId_);
        std::cout << "tensor info" << tensor.GetDesc() << std::endl;
        APP_ERROR ret = MxBase::TensorBase::TensorBaseMalloc(tensor);
        if (ret != APP_ERR_OK) {
            LogError << "TensorBaseMalloc failed, ret=" << ret << ".";
            return ret;
        }
        outputs.push_back(tensor);
    }
    MxBase::DynamicInfo dynamicInfo = {};
    // static batch
    dynamicInfo.dynamicType = MxBase::DynamicType::STATIC_BATCH;
    APP_ERROR ret = model_->ModelInference(inputs, outputs, dynamicInfo);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    float* chr = nullptr;
    outputs[CONF_INDEX].ToHost();     // data from dicice to Host
    chr = static_cast<float *>(outputs[CONF_INDEX].GetBuffer());
    for (int conf_index = 0; conf_index < OUT_FORMAT; conf_index++) {
        float temp_fst = std::exp(chr[conf_index * 2]);
        float temp_snd = std::exp(chr[conf_index * 2 + 1]);
        chr[conf_index * 2] = temp_fst / (temp_fst + temp_snd);
        chr[conf_index * 2 + 1] = temp_snd / (temp_fst + temp_snd);
    }

    return APP_ERR_OK;
}


APP_ERROR FaceboxesDetection::Process(std::string &imgPath) {
    ImageShape imageShape{};
    MxBase::DvppDataInfo dvppData = {};
    ImagePreprocess(imgPath);
    APP_ERROR ret = ReadImage(imgPath, dvppData, imageShape);
    if (ret != APP_ERR_OK) {
        LogError << "ReadImage failed, ret=" << ret << ".";
        return ret;
    }
    MxBase::TensorBase outTensor;
    ret = Resize(dvppData, outTensor);
    if (ret != APP_ERR_OK) {
        LogError << "Resize failed, ret=" << ret << ".";
        return ret;
    }

    std::cout << "tensorBase info :" << outTensor.GetDesc() << std::endl;
    std::vector<MxBase::TensorBase> inputs = {};
    std::vector<MxBase::TensorBase> outputs = {};
    inputs.push_back(outTensor);

    ret = Inference(inputs, outputs);
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }
    PostProcess(outputs);

    return APP_ERR_OK;
}


APP_ERROR FaceboxesDetection::PostProcess(std::vector<MxBase::TensorBase> &outputs) {
    std::vector <std::vector <float> > anchors = {};
    float boxes[OUT_FORMAT][4];
    std::vector <int> index = {};
    std::vector <std::vector <float> > postbox = {};
    std::vector <float> postscore = {};
    std::vector <std::vector <float> > postboxsort = {};
    std::vector <float> postscoresort = {};
    std::vector <std::vector <float> > dets = {};
    std::vector <int> reserved_boxes = {};
    std::vector <std::vector <float> > dets_result = {};
    PriorBox(anchors);
    DecodeBox(anchors, outputs, boxes);
    for (int i = 0; i < OUT_FORMAT; i++) {
        boxes[i][0] = boxes[i][0] * SCALE[0];
        boxes[i][1] = boxes[i][1] * SCALE[1];
        boxes[i][2] = boxes[i][2] * SCALE[2];
        boxes[i][3] = boxes[i][3] * SCALE[3];
    }
    SortTOBoxAndScore(outputs, boxes, postbox, postscore, postboxsort, postscoresort, dets, index);
    Nms(dets, index, reserved_boxes);
    for (int i = 0; i < reserved_boxes.size(); i++) {
        std::vector <float> tempdets = {};
        tempdets.push_back(dets[reserved_boxes[i]][0]);
        tempdets.push_back(dets[reserved_boxes[i]][1]);
        tempdets.push_back(dets[reserved_boxes[i]][2]);
        tempdets.push_back(dets[reserved_boxes[i]][3]);
        tempdets.push_back(dets[reserved_boxes[i]][4]);
        dets_result.push_back(tempdets);
    }
    for (int i = 0; i < reserved_boxes.size(); i++) {
        dets_result[i][2] = static_cast<float>(static_cast<int>(dets_result[i][2]) -
                            static_cast<int>(dets_result[i][0]));
        dets_result[i][3] = static_cast<float>(static_cast<int>(dets_result[i][3]) -
                            static_cast<int>(dets_result[i][1]));
    }
    std::cout << "dets_result is: " << std::endl;
    for (int i = 0; i < reserved_boxes.size(); i++) {
        dets_result[i][0] = static_cast<float>(static_cast<int>(dets_result[i][0]));
        dets_result[i][1] = static_cast<float>(static_cast<int>(dets_result[i][1]));
        dets_result[i][2] = static_cast<float>(static_cast<int>(dets_result[i][2]));
        dets_result[i][3] = static_cast<float>(static_cast<int>(dets_result[i][3]));
        std::cout << dets_result[i][0] << " ," << dets_result[i][1] << " ," << dets_result[i][2] << " ,"
        << dets_result[i][3] << " ," << dets_result[i][4] << std::endl;
    }

    return APP_ERR_OK;
}
