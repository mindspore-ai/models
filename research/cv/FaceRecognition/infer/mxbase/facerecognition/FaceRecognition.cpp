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

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <memory>
#include "FaceRecognition.h"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"

namespace {
    const uint32_t YUV_BYTE_NU = 3;
    const uint32_t YUV_BYTE_DE = 2;
    const uint32_t VPC_H_ALIGN = 2;
    const int emb_size = 256;
}

union Fp32 {
    uint32_t u;
    float f;
};

// calculate the acc
std::vector<int> FaceRecognition::cal_topk(int idx, std::string str_zj,
std::map<std::string, std::vector<std::string> >& zj2jk_pairs,
std::map<std::string, std::vector<float> >& test_embedding_tot,
std::vector<std::vector<float> >& dis_embedding_tot_np,
const std::map<int, std::vector<std::vector<float> > >& jk_all_embedding,
std::vector<std::string>& dis_label) {
    std::ofstream zj_outfile;
    std::ofstream jk_outfile;
    std::vector<int>correct(2, 0);

    std::vector<std::string>jk_all = zj2jk_pairs[str_zj];
    std::vector<float>zj_embedding = test_embedding_tot[str_zj];
    std::vector<float>mm(dis_embedding_tot_np.size(), 0);
    float mm_max = FLT_MIN;
    int zj_index = 0;

    for (int i = 0; i < dis_embedding_tot_np.size(); i++) {
        for (int j = 0; j < zj_embedding.size(); j++) {
            mm[i] += zj_embedding[j]*dis_embedding_tot_np[i][j];
        }
        if (mm_max < mm[i]) {
            mm_max = mm[i];
            zj_index = i;
        }
    }

    std::vector<float>mm_max_jk_all(jk_all.size(), FLT_MIN);
    std::vector<int>jk_index;

    float sum;
    for (int k = 0; k < jk_all.size(); k++) {
        std::vector<float>jk_embedding = test_embedding_tot[jk_all[k]];
        int index = 0;
        for (int i = 0; i < dis_embedding_tot_np.size(); i++) {
                sum = 0;
                for (int j = 0; j < jk_embedding.size(); j++) {
                    sum += jk_embedding[j]*dis_embedding_tot_np[i][j];
                }
                if (mm_max_jk_all[k] < sum) {
                    mm_max_jk_all[k] = sum;
                    index = i;
                }
            }
        jk_index.push_back(index);
    }
    // the first step is write the groundtruth to the zj and jk
    zj_outfile.open("./zj_result.txt", std::ios::app);
    jk_outfile.open("./jk_result.txt", std::ios::app);
    for (int i = 0; i < jk_all.size(); i++) {
        std::vector<float>jk_embedding = test_embedding_tot[jk_all[i]];
        zj_outfile << str_zj << " ";
        jk_outfile << jk_all[i] << " ";
        // 22:23
        float similarity = inner_product(jk_embedding.begin(), jk_embedding.end(), zj_embedding.begin(), 0.0);
        if (similarity > mm_max) {
            correct[0]++;
            zj_outfile << str_zj << "\n";
        } else {
            zj_outfile << dis_label[zj_index] << "\n";
        }
        if (similarity > mm_max_jk_all[i]) {
            jk_outfile << jk_all[i] << "\n";
            correct[1]++;
        } else {
            jk_outfile << dis_label[jk_index[i]] << "\n";
        }
        // pass ci
        test_embedding_tot[jk_all[i]] = jk_embedding;
    }
    // pass ci
    zj2jk_pairs[str_zj] = jk_all;
    std::string temp = dis_label[0];
    dis_label[0] = temp;
    zj_outfile.close();
    jk_outfile.close();
    return correct;
}

void FaceRecognition::l2normalize(std::vector<std::vector<float>>& out) {
    std::vector<std::vector<float> >out_o(out.begin(), out.end());

    float epsilon = 1e-12;
    std::vector<float>out1(out.size(), 0);
    for (int i = 0; i < out_o.size(); i++) {
        for (int j = 0; j < 256; j++) {
            out_o[i][j] = abs(out_o[i][j]);
            out_o[i][j] = pow(out_o[i][j], 2);
            out1[i] += out_o[i][j];
        }
        out1[i] = pow(out1[i], 0.5);
        if (out1[i] < 0 && out1[i] > -epsilon) {
            out1[i] = -epsilon;
        } else if (out1[i] >= 0 && out1[i] < epsilon) {
            out1[i] = epsilon;
        }
    }
    for (int i = 0; i < out.size(); i++) {
        for (int j = 0; j < 256; j++) {
            out[i][j] = out[i][j]/out1[i];
        }
    }
}

int FaceRecognition::line_of_txt(std::string s) {
    int count = 0;
    std::ifstream fin(s);
    std::string ss;
    while (getline(fin, ss)) {
        count++;
    }
    return count;
}

void FaceRecognition::deal_txt_img(std::string s, int& count, int batch_img, int out_vector_len,
std::vector<std::vector<float>>& test_embedding_tot_np,
std::map<std::string, std::vector<float>>& test_embedding_tot) {
    std::ifstream fin(s);
    std::string ss;
    while (getline(fin, ss)) {
        // read image
        int pos = 0;
        for (int i = 0; i < ss.size(); i++) {
            if (ss[i] == ' ') {
                pos = i;
                break;
            }
        }
        std::string imgPath = ss.substr(0, pos);

        // put the tensor to the net, and get the output
        std::vector<std::vector<float> >out(batch_img, std::vector<float>(out_vector_len, 1));
        Process(imgPath, out);


        // l2normalize
        l2normalize(out);

        // The result of regularization is stored in test_embedding_tot_np
        test_embedding_tot_np[count] = out[0];
        test_embedding_tot[ss] = test_embedding_tot_np[count];

        count++;
    }
}

void FaceRecognition::deal_txt_img_dis(std::string s, int& count, int batch_img, int out_vector_len,
std::vector<std::vector<float>>& test_embedding_tot_np,
std::map<std::string, std::vector<float>>& test_embedding_tot,
std::vector<std::string>& dis_label) {
    std::ifstream fin(s);
    std::string ss;
    while (getline(fin, ss)) {
        std::string imgPath = ss;
        // store the image path
        dis_label.push_back(imgPath);
        // put the tensor to the net, and get the output
        std::vector<std::vector<float> >out(batch_img, std::vector<float>(out_vector_len, 1));
        Process(imgPath, out);

        // l2normalize
        l2normalize(out);

        // The result of regularization is stored in test_embedding_tot_np
        test_embedding_tot_np[count] = out[0];
        test_embedding_tot[ss] = test_embedding_tot_np[count];

        count++;
    }
}

std::string FaceRecognition::get_lable_num(std::string s) {
    int pos = 0;
    for (int i = 0; i < s.size(); i++) {
        if (s[i] == ' ') {
            pos = i;
            break;
        }
    }
    return s.substr(pos+1);
}

void FaceRecognition::txt_to_pair(std::map<std::string, std::vector<std::string>>& zj2jk_pairs,
int img_tot_zj, int img_tot_jk, std::vector<int>& ID_nums,
const std::string &zj_list_path, const std::string &jk_list_path) {
    std::ifstream zjFile;
    zjFile.open(zj_list_path);
    std::string str_zj;
    getline(zjFile, str_zj, '\n');
    std::string lable_num = get_lable_num(str_zj);

    std::ifstream jkFile;
    jkFile.open(jk_list_path);
    std::string str_jk;

    int id_nums = 0;
    for (int i = 0; i < img_tot_jk; i++) {
        getline(jkFile, str_jk, '\n');
        if (lable_num == get_lable_num(str_jk)) {
            id_nums++;
            zj2jk_pairs[str_zj].push_back(str_jk);
        } else {
            ID_nums.push_back(id_nums);
            id_nums = 1;
            getline(zjFile, str_zj, '\n');
            lable_num = get_lable_num(str_zj);
            zj2jk_pairs[str_zj].push_back(str_jk);
        }
    }
    ID_nums.push_back(id_nums);
}

void FaceRecognition::get_jk_all_embedding(std::vector<int>& ID_nums,
std::map<std::string, std::vector<float> >& test_embedding_tot, std::map<int,
std::vector<std::vector<float> > >& jk_all_embedding, const std::string &jk_list_path) {
    std::ifstream jkFile;
    jkFile.open(jk_list_path);
    std::string str_jk;
    for (int idx = 0; idx < ID_nums.size(); idx++) {
        for (int i = 0; i < ID_nums[idx]; i++) {
            getline(jkFile, str_jk, '\n');
            jk_all_embedding[idx].push_back(test_embedding_tot[str_jk]);
        }
    }
}

APP_ERROR FaceRecognition::Init(const InitParam &initParam) {
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
    model_ = std::make_shared<MxBase::ModelInferenceProcessor>();
    ret = model_->Init(initParam.modelPath, modelDesc_);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInferenceProcessor init failed, ret=" << ret << ".";
        return ret;
    }
    MxBase::ConfigData configData;
    const std::string softmax = initParam.softmax ? "true" : "false";
    const std::string checkTensor = initParam.checkTensor ? "true" : "false";

    configData.SetJsonValue("CLASS_NUM", std::to_string(initParam.classNum));
    configData.SetJsonValue("TOP_K", std::to_string(initParam.topk));
    configData.SetJsonValue("SOFTMAX", softmax);
    configData.SetJsonValue("CHECK_MODEL", checkTensor);

    auto jsonStr = configData.GetCfgJson().serialize();
    std::map<std::string, std::shared_ptr<void>> config;
    config["postProcessConfigContent"] = std::make_shared<std::string>(jsonStr);
    config["labelPath"] = std::make_shared<std::string>(initParam.labelPath);

    post_ = std::make_shared<MxBase::Resnet50PostProcess>();
    ret = post_->Init(config);
    if (ret != APP_ERR_OK) {
        LogError << "Resnet50PostProcess init failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR FaceRecognition::DeInit() {
    model_->DeInit();
    post_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

bool FaceRecognition::ReadImage(const std::string &imgPath, cv::Mat &imageMat) {
    imageMat = cv::imread(imgPath, cv::IMREAD_COLOR);
    if (!imageMat.empty()) {
        return true;
    }
    return false;
}

APP_ERROR FaceRecognition::ResizeImage(const cv::Mat &srcImageMat, cv::Mat &dstImageMat) {
    static constexpr uint32_t resizeHeight = 112;
    static constexpr uint32_t resizeWidth = 112;

    cv::resize(srcImageMat, dstImageMat, cv::Size(resizeWidth, resizeHeight));
    return APP_ERR_OK;
}

APP_ERROR FaceRecognition::CVMatToTensorBase(const cv::Mat &imageMat, MxBase::TensorBase &tensorBase) {
    const uint32_t dataSize =  imageMat.cols *  imageMat.rows * MxBase::YUV444_RGB_WIDTH_NU;
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

APP_ERROR FaceRecognition::Inference(const std::vector<MxBase::TensorBase> &inputs,
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
    auto startTime = std::chrono::high_resolution_clock::now();
    APP_ERROR ret = model_->ModelInference(inputs, outputs, dynamicInfo);
    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();  // save time
    inferCostTimeMilliSec += costMs;
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

float FaceRecognition::uint6_cov_float(uint16_t value) {
    const Fp32 magic = { (254U - 15U) << 23 };
    const Fp32 was_infnan = { (127U + 16U) << 23 };
    Fp32 out;
    out.u = (value & 0x7FFFU) << 13;   /* exponent/mantissa bits */
    out.f *= magic.f;                  /* exponent adjust */
    if (out.f >= was_infnan.f) {
        out.u |= 255U << 23;
    }
    out.u |= (value & 0x8000U) << 16;  /* sign bit */
    return out.f;
}

APP_ERROR FaceRecognition::PostProcess(std::vector<MxBase::TensorBase> &tensors,
                                        std::vector<std::vector<float>> &out) {
    APP_ERROR ret = APP_ERR_OK;
    for (MxBase::TensorBase &input : tensors) {
        ret = input.ToHost();
        if (ret != APP_ERR_OK) {
            LogError << "----------Error occur!!" << std::endl;
        }
    }
    auto inputs = tensors;
    if (ret != APP_ERR_OK) {
        LogError << "CheckAndMoveTensors failed. ret=" << ret;
        return ret;
    }
    const uint32_t softmaxTensorIndex = 0;
    auto softmaxTensor = inputs[softmaxTensorIndex];
    void *softmaxTensorPtr = softmaxTensor.GetBuffer();
    uint32_t topk = 256;
    std::vector<uint32_t> idx = {};
    for (uint32_t j = 0; j < topk; j++) {
        idx.push_back(j);
    }
    for (uint32_t j = 0; j < topk; j++) {
        float value = *(static_cast<float *>(softmaxTensorPtr) + 0 * topk + j);
        out[0][j] = value;
    }
    return APP_ERR_OK;
}

// Add another parameter out, passing in the outgoing parameter
APP_ERROR FaceRecognition::Process(const std::string &imgPath, std::vector<std::vector<float>>& out) {
    LogInfo << "processing ---------" << imgPath << std::endl;
    // read image
    cv::Mat imageMat;
    bool readTrue = ReadImage(imgPath, imageMat);
    if (!readTrue) {
        LogError << "ReadImage failed!!!" << std::endl;
    }
    // resize image
    ResizeImage(imageMat, imageMat);
    std::vector<MxBase::TensorBase> inputs = {};
    std::vector<MxBase::TensorBase> outputs = {};
    MxBase::TensorBase tensorBase;
    // convert to tensor
    APP_ERROR ret = CVMatToTensorBase(imageMat, tensorBase);
    if (ret != APP_ERR_OK) {
        LogError << "CVMatToTensorBase failed, ret=" << ret << ".";
        return ret;
    }
    inputs.push_back(tensorBase);
    auto startTime = std::chrono::high_resolution_clock::now();
    // infer
    ret = Inference(inputs, outputs);
    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();  // save time
    inferCostTimeMilliSec += costMs;
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }
    // to postprocess
    ret = PostProcess(outputs, out);
    if (ret != APP_ERR_OK) {
        LogError << "PostProcess failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR FaceRecognition::main(const std::string &zj_list_path, const std::string &jk_list_path,
const std::string &dis_list_path) {
    // read zj jk dis txt content
    std::string s_zj = zj_list_path;
    std::string s_jk = jk_list_path;
    std::string s_dis = dis_list_path;
    // total number of zj and jk txt length
    int img_tot = 0;
    img_tot += line_of_txt(s_zj);
    int img_tot_zj = img_tot;
    img_tot += line_of_txt(s_jk);
    int img_tot_jk = img_tot - img_tot_zj;
    LogInfo << img_tot << std::endl;
    // out shape is 1,256
    int batch_img = 1;
    int out_vector_len = 256;
    // Define test_tot_np (stored according to idx) to store out
    std::vector<std::vector<float> >test_embedding_tot_np(img_tot, std::vector<float>(out_vector_len, 0));
    // Define test_tot (stored according to label) to store out
    std::map<std::string, std::vector<float> >test_embedding_tot;
    // Process the images in the read txt
    int count = 0;
    LogInfo << "-------------------zj--------------------" << std::endl;
    deal_txt_img(s_zj, count, batch_img, out_vector_len, test_embedding_tot_np, test_embedding_tot);
    LogInfo << "-------------------jk--------------------" << std::endl;
    deal_txt_img(s_jk, count, batch_img, out_vector_len, test_embedding_tot_np, test_embedding_tot);
    // for dis images
    int dis_img_tot = 0;
    dis_img_tot += line_of_txt(s_dis);
    std::vector<std::vector<float> >dis_embedding_tot_np(dis_img_tot, std::vector<float>(out_vector_len, 0));
    std::map<std::string, std::vector<float> >dis_embedding_tot;
    int dis_count = 0;
    LogInfo << "-------------------dis--------------------" << std::endl;
    // dis_label
    std::vector<std::string> dis_label;
    deal_txt_img_dis(s_dis, dis_count, batch_img, out_vector_len, dis_embedding_tot_np, dis_embedding_tot, dis_label);
    // step3
    // get zj2jk_pairs
    LogInfo << "----------------step 3---------------" << std::endl;
    std::map<std::string, std::vector<std::string> >zj2jk_pairs;
    std::vector<int>ID_nums;
    txt_to_pair(zj2jk_pairs, img_tot_zj, img_tot_jk, ID_nums, zj_list_path, jk_list_path);
    LogInfo << "----------------step 3 over!!!------------" << std::endl;
    // into cal_topk
    std::ifstream zjFile;
    zjFile.open(zj_list_path);
    std::string str_zj;
    // get jk_all_embedding
    std::map<int, std::vector<std::vector<float> > >jk_all_embedding;
    get_jk_all_embedding(ID_nums, test_embedding_tot, jk_all_embedding, jk_list_path);
    int task_num = 1;
    std::vector<int>correct(2*task_num, 0);
    std::vector<int>tot(task_num, 0);
    for (int idx = 0; idx < zj2jk_pairs.size(); idx++) {
        getline(zjFile, str_zj, '\n');
        std::vector<int>out1;
        out1 = cal_topk(idx, str_zj, zj2jk_pairs, test_embedding_tot,
        dis_embedding_tot_np, jk_all_embedding, dis_label);
        correct[0] += out1[0];
        correct[1] += out1[1];
        tot[0] += zj2jk_pairs[str_zj].size();
    }
    LogInfo << "tot[0] is " << tot[0] << std::endl;
    LogInfo << "correct[0] is " << correct[0] << std::endl;
    LogInfo << "correct[1] is " << correct[1] << std::endl;

    float zj2jk_acc = static_cast<float>(correct[0]) / static_cast<float>(tot[0]);
    float jk2zj_acc = static_cast<float>(correct[1]) / static_cast<float>(tot[0]);
    float avg_acc = (zj2jk_acc + jk2zj_acc) / 2;
    LogInfo << " zj2jk_acc---------" << zj2jk_acc << std::endl;
    LogInfo << " jk2zj_acc---------" << jk2zj_acc << std::endl;
    LogInfo << " avg_acc---------" << avg_acc << std::endl;
}



