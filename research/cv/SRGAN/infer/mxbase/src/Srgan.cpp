/*
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "Srgan.h"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"


using MxBase::TensorDesc;
using MxBase::TensorBase;
using MxBase::MemoryData;
using MxBase::MemoryHelper;
using MxBase::TENSOR_DTYPE_FLOAT32;
using MxBase::DynamicInfo;
using MxBase::DynamicType;

const int IMG_CHANNEL = 3, IMG_HEIGHT = 126, IMG_WIDTH = 126;
const double MAX_PX_VALUE = 255.0, MIN_PX_VALUE = 0.0, TEN = 10.0;
const int ZERO = 0, ONE = 1, SCALE = 4, CROP_SIZE = 4;

void PrintTensorShape(const std::vector<TensorDesc> &tensorDescVec, const std::string &tensorName) {
    LogInfo << "The shape of " << tensorName << " is as follows:";
    for (size_t i = 0; i < tensorDescVec.size(); ++i) {
        LogInfo << "  Tensor " << i << ":";
        for (size_t j = 0; j < tensorDescVec[i].tensorDims.size(); ++j) {
            LogInfo << "   dim: " << j << ": " << tensorDescVec[i].tensorDims[j];
        }
    }
}

void PrintInputShape(const std::vector<MxBase::TensorBase> &input) {
    MxBase::TensorBase img = input[0];
    LogInfo << "  -------------------------input0 ";
    LogInfo << img.GetDataType();
    LogInfo << img.GetShape()[0] << ", " << img.GetShape()[1]
    << ", "  << img.GetShape()[2] << ", " << img.GetShape()[3];
    LogInfo << img.GetSize();
}


APP_ERROR Srgan::Init(const InitParam &initParam) {
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
    srPath_ = initParam.srPath;
    gtPath_ = initParam.gtPath;
    PrintTensorShape(modelDesc_.inputTensors, "Model Input Tensors");
    PrintTensorShape(modelDesc_.outputTensors, "Model Output Tensors");


    return APP_ERR_OK;
}

APP_ERROR Srgan::DeInit() {
    dvppWrapper_->DeInit();
    model_->DeInit();

    MxBase::DeviceManager::GetInstance()->DestroyDevices();

    return APP_ERR_OK;
}


APP_ERROR Srgan::ReadImage(const std::string &imgPath, cv::Mat *imageMat) {
    *imageMat = cv::imread(imgPath, cv::IMREAD_COLOR);
    return APP_ERR_OK;
}

APP_ERROR Srgan::PadImage(const cv::Mat &imageMat, cv::Mat *imgPad) {
    size_t W_o = imageMat.cols, H_o = imageMat.rows;

    for (size_t c=0; c < IMG_CHANNEL; c++) {
        size_t h = 0;
        size_t h_o = 0;
        bool h_b = true;
        while (h < IMG_HEIGHT) {
            size_t w_o = 0, w = 0;
            bool w_b = true;
            while (w < IMG_WIDTH) {
                imgPad->at<cv::Vec3b>(h, w)[c] = imageMat.at<cv::Vec3b>(h_o, w_o)[c];
                if (w_o == W_o - ONE) {
                    if (w_b)
                        w_b = !w_b;
                    else
                        w_o--;
                } else if (w_o == ZERO) {
                    if (!w_b)
                        w_b = !w_b;
                    else
                        w_o++;
                } else if (!w_b) {
                    w_o--;
                } else {
                    w_o++;
                }
                w++;
            }
            if (h_o == H_o - ONE) {
                if (h_b)
                    h_b = !h_b;
                else
                    h_o--;
            } else if (h_o == ZERO) {
                if (!h_b)
                    h_b = !h_b;
                else
                    h_o++;
            } else if (h_b) {
                h_o++;
            } else {
                h_o--;
            }
            h++;
        }
    }

    return APP_ERR_OK;
}


APP_ERROR Srgan::CVMatToTensorBase(const cv::Mat &imageMat, MxBase::TensorBase *tensorBase) {
    uint32_t dataSize = 1;
    for (size_t i = 0; i < modelDesc_.inputTensors.size(); ++i) {
        std::vector <uint32_t> shape = {};
        for (size_t j = 0; j < modelDesc_.inputTensors[i].tensorDims.size(); ++j) {
            shape.push_back((uint32_t) modelDesc_.inputTensors[i].tensorDims[j]);
        }
        for (uint32_t s = 0; s < shape.size(); ++s) {
            dataSize *= shape[s];
        }
    }

    cv::Mat imgPad(IMG_HEIGHT, IMG_WIDTH, CV_8UC3);

    APP_ERROR ret = PadImage(imageMat, &imgPad);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Img pad error";
        return ret;
    }


    // mat NHWC to NCHW, BGR to RGB
    size_t H = IMG_HEIGHT, W = IMG_WIDTH, C = IMG_CHANNEL;

    float *mat_data = new float[dataSize];
    dataSize = dataSize * 4;
    int id;
    for (size_t c = 0; c < C; c++) {
        for (size_t h = 0; h < H; h++) {
            for (size_t w = 0; w < W; w++) {
                id = (C - c - 1) * (H * W) + h * W + w;
                mat_data[id] = (imgPad.at<cv::Vec3b>(h, w)[c] -
                (MAX_PX_VALUE - MIN_PX_VALUE)/2.0) / ((MAX_PX_VALUE - MIN_PX_VALUE)/2.0);
            }
        }
    }

    MemoryData memoryDataDst(dataSize, MemoryData::MEMORY_DEVICE, deviceId_);
    MemoryData memoryDataSrc(reinterpret_cast<void *>(&mat_data[0]), dataSize, MemoryData::MEMORY_HOST_MALLOC);

    ret = MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }
    std::vector <uint32_t> shape = {1, IMG_CHANNEL, IMG_HEIGHT, IMG_WIDTH};
    *tensorBase = TensorBase(memoryDataDst, false, shape, TENSOR_DTYPE_FLOAT32);
    return APP_ERR_OK;
}


APP_ERROR Srgan::Inference(const std::vector<MxBase::TensorBase> &inputs,
                           std::vector<MxBase::TensorBase> *outputs) {
    auto dtypes = model_->GetOutputDataType();
    for (size_t i = 0; i < modelDesc_.outputTensors.size(); ++i) {
        std::vector<uint32_t> shape = {};
        for (size_t j = 0; j < modelDesc_.outputTensors[i].tensorDims.size(); ++j) {
            shape.push_back((uint32_t)modelDesc_.outputTensors[i].tensorDims[j]);
        }
        TensorBase tensor(shape, dtypes[i], MemoryData::MemoryType::MEMORY_DEVICE, deviceId_);
        APP_ERROR ret = TensorBase::TensorBaseMalloc(tensor);
        if (ret != APP_ERR_OK) {
            LogError << "TensorBaseMalloc failed, ret=" << ret << ".";
            return ret;
        }
        outputs->push_back(tensor);
    }
    DynamicInfo dynamicInfo = {};
    dynamicInfo.dynamicType = DynamicType::STATIC_BATCH;
    dynamicInfo.batchSize = 1;

    PrintInputShape(inputs);

    APP_ERROR ret = model_->ModelInference(inputs, *outputs, dynamicInfo);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}


APP_ERROR Srgan::PostProcess(std::vector<MxBase::TensorBase> outputs, cv::Mat *resultImg) {
    LogInfo << "output_size:" << outputs.size();
    LogInfo <<  "output0_datatype:" << outputs[0].GetDataType();
    LogInfo << "output0_shape:" << outputs[0].GetShape()[0] << ", "
    << outputs[0].GetShape()[1] << ", "  << outputs[0].GetShape()[2] << ", "
    << outputs[0].GetShape()[3];
    LogInfo << "output0_bytesize:"  << outputs[0].GetByteSize();

    APP_ERROR ret = outputs[0].ToHost();
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "tohost fail.";
        return ret;
    }
    float *outputPtr = reinterpret_cast<float *>(outputs[0].GetBuffer());

    size_t  H = IMG_HEIGHT * SCALE, W = IMG_WIDTH * SCALE, C = IMG_CHANNEL,
     org_H = resultImg->rows, org_W = resultImg->cols;

    cv::Mat outputImg(H, W, CV_8UC3);

    float tmpNum;
    for (size_t c = 0; c < C; c++) {
        for (size_t h = 0; h < H; h++) {
            for (size_t w = 0; w < W; w++) {
                tmpNum = *(outputPtr + (C - c - 1) * (H * W) + h * W + w) *
                (MAX_PX_VALUE - MIN_PX_VALUE)/2.0
                + (MAX_PX_VALUE - MIN_PX_VALUE)/2.0;
                outputImg.at<cv::Vec3b>(h, w)[c] = static_cast<int>(tmpNum);
            }
        }
    }
    for (size_t c = 0; c < C; c++) {
        for (size_t h = 0; h < org_H; h++) {
            for (size_t w = 0; w < org_W; w++) {
                resultImg->at<cv::Vec3b>(h, w)[c] = outputImg.at<cv::Vec3b>(h, w)[c];
            }
        }
    }
    return APP_ERR_OK;
}

APP_ERROR Srgan::CalPSNR(const cv::Mat &Img1, const cv::Mat &Img2, double_t *psnr) {
    if ((Img1.cols != Img2.cols) || (Img1.rows !=Img2.rows))
        return APP_ERR_COMM_FAILURE;

    cv::Rect area(CROP_SIZE, CROP_SIZE, Img1.cols - 2 * CROP_SIZE, Img1.rows - 2 * CROP_SIZE);
    cv::Mat crop_Img1 = Img1(area);
    cv::Mat crop_Img2 = Img2(area);

    cv::Mat Diff;

    cv::absdiff(crop_Img1, crop_Img2, Diff);

    Diff.convertTo(Diff, CV_32F);

    Diff = Diff.mul(Diff);

    cv::Scalar S = cv::sum(Diff);

    double sse;
    if (crop_Img1.channels() ==3)
        sse = S.val[0]+S.val[1]+S.val[2];
    else
        sse = S.val[0];

    if (sse <= 1e-10) {
        return 0;
    } else {
        double mse = sse / static_cast<double>((crop_Img1.channels() * crop_Img1.total()));
        *psnr = TEN * log10((MAX_PX_VALUE * MAX_PX_VALUE)/mse);
    }

    return APP_ERR_OK;
}


APP_ERROR Srgan::SaveResult(const cv::Mat &resultImg, const std::string &imgName) {
    DIR *dirPtr = opendir(srPath_.c_str());
    if (dirPtr == nullptr) {
        std::string path1 = "mkdir -p " + srPath_;
        system(path1.c_str());
    }
    cv::imwrite(srPath_ + "/" + imgName , resultImg);
    return APP_ERR_OK;
}

APP_ERROR Srgan::Process(const std::string &imgPath, const std::string &imgName) {
    cv::Mat imageMat;
    APP_ERROR ret = ReadImage(imgPath, &imageMat);
    if (ret != APP_ERR_OK) {
        LogError << "ReadImage failed, ret=" << ret << ".";
        return ret;
    }

    size_t o_img_W = imageMat.cols, o_img_H = imageMat.rows;

    std::vector<MxBase::TensorBase> inputs = {};
    std::vector<MxBase::TensorBase> outputs = {};

    TensorBase tensorBase;
    ret = CVMatToTensorBase(imageMat, &tensorBase);
    if (ret != APP_ERR_OK) {
        LogError << "CVMatToTensorBase failed, ret=" << ret << ".";
        return ret;
    }
    inputs.push_back(tensorBase);


    auto startTime = std::chrono::high_resolution_clock::now();
    ret = Inference(inputs, &outputs);
    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();  // save time
    inferCostTimeMilliSec += costMs;
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }
    cv::Mat resultImg(o_img_H*4, o_img_W*4, CV_8UC3);
    ret = PostProcess(outputs, &resultImg);
    if (ret != APP_ERR_OK) {
        LogError << "PostProcess failed, ret=" << ret << ".";
        return ret;
    }

    cv::Mat gtImg;
    ret = ReadImage(gtPath_+'/'+imgName, &gtImg);
    if (ret != APP_ERR_OK) {
        LogError << "Read GT Image failed, ret=" << ret << ".";
        return ret;
    }

    ret = CalPSNR(resultImg, gtImg, &psnr_);
    if (ret != APP_ERR_OK) {
        LogError << "Cal PSNR failed, ret=" << ret << ".";
        return ret;
    }
    LogInfo << "The img named[" << imgName << "] PSNR :"
    << psnr_ << " dB.";

    ret = SaveResult(resultImg, imgName);
    if (ret != APP_ERR_OK) {
        LogError << "Save infer results into file failed. ret = " << ret << ".";
        return ret;
    }

    return APP_ERR_OK;
}
