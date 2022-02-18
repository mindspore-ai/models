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
#include "PAMTRIClassifyOpencv.h"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"
using MxBase::DynamicInfo;
using MxBase::DynamicType;
using MxBase::MemoryData;
using MxBase::MemoryHelper;
using MxBase::TensorBase;
namespace {
struct ImageNet {
    float mean[3] = {0.485, 0.456, 0.406};
    float std[3] = {0.229, 0.224, 0.225};
    float mean_avg = (0.485 + 0.456 + 0.406) / 3.0;
    float std_avg = (0.229 + 0.224 + 0.225) / 3.0;
};
const struct ImageNet imagenet;
const float PI = 3.14159265358979323846;
const int SEG_HW[] = {64, 64};
const int SEGS[][4] = {{5, 15, 16, 17},  {5, 6, 12, 15},   {6, 10, 11, 12},
                       {23, 33, 34, 35}, {23, 24, 30, 33}, {24, 28, 29, 30},
                       {10, 11, 29, 28}, {11, 12, 30, 29}, {12, 13, 31, 30},
                       {13, 14, 32, 31}, {14, 15, 33, 32}, {15, 16, 34, 33},
                       {16, 17, 35, 34}};
const int FLIP_PAIRS[18][2] = {{0, 18},  {1, 19},  {2, 20},  {3, 21},  {4, 22},
                               {5, 23},  {6, 24},  {7, 25},  {8, 26},  {9, 27},
                               {10, 28}, {11, 29}, {12, 30}, {13, 31}, {14, 32},
                               {15, 33}, {16, 34}, {17, 35}};
const int INPUT_HEIGHT = 256;
const int INPUT_WIDTH = 256;
const int INPUT_IMG_SIZE = INPUT_HEIGHT * INPUT_WIDTH * sizeof(float);
const int NUM_JOINTS = 36;
const int PN_OUTPUT_SIZE = NUM_JOINTS * SEG_HW[0] * SEG_HW[1] * sizeof(float);
const int IMG_CHNLS = 3;
const int SEGMENT_CHNLS = 13;
const int HEATMAP_CHNLS = 36;
const int TOTAL_CHNLS = IMG_CHNLS + SEGMENT_CHNLS + HEATMAP_CHNLS;
const int VKPT_NUM = 108;
const int PIXEL_STD = 200;

cv::Point2f get_dir(const cv::Point2f &src_point, const float &rot_rad) {
    float sn = sin(rot_rad);
    float cs = cos(rot_rad);

    cv::Point2f src_result(0, 0);
    src_result.x = src_point.x * cs - src_point.y * sn;
    src_result.y = src_point.x * sn + src_point.y * cs;

    return src_result;
}

cv::Point2f get_3rd_point(const cv::Point2f &a, const cv::Point2f &b) {
    cv::Point2f direct = a - b;
    return b + cv::Point2f(-direct.y, direct.x);
}

cv::Point2f affine_transform(const cv::Point2f &pt, const cv::Mat &t) {
    double *t_ptr = reinterpret_cast<double *>(t.data);
    float new_pt[] = {pt.x, pt.y, 1.0};
    return cv::Point2f(
        new_pt[0] * t_ptr[0] + new_pt[1] * t_ptr[1] + new_pt[2] * t_ptr[2],
        new_pt[0] * t_ptr[3] + new_pt[1] * t_ptr[4] + new_pt[2] * t_ptr[5]);
}

int sign(float x) { return x > 0 ? 1 : (x < 0 ? -1 : 0); }

void norm_multitask_input(float *vec, int input_dim1) {
    for (int chnl = 0; chnl < IMG_CHNLS; chnl++) {
        for (int row = 0; row < INPUT_HEIGHT; row++) {
            for (int col = 0; col < INPUT_WIDTH; col++) {
            vec[0] = (vec[0] / 255 - imagenet.mean[chnl]) / imagenet.std[chnl];
            ++vec;
            }
        }
    }
    for (int chnl = 0; chnl < input_dim1 - IMG_CHNLS; chnl++) {
        for (int row = 0; row < INPUT_HEIGHT; row++) {
            for (int col = 0; col < INPUT_WIDTH; col++) {
            vec[0] = (vec[0] / 255 - imagenet.mean_avg) / imagenet.std_avg;
            ++vec;
            }
        }
    }
}

}  // namespace

APP_ERROR PAMTRIClassifyOpencv::init_(const InitParam &initParam) {
    pn_input_shape_ =
        std::vector<int>({batch_size_, IMG_CHNLS, INPUT_HEIGHT, INPUT_WIDTH});
    mt_img_input_shape_ =
        std::vector<int>({batch_size_, TOTAL_CHNLS, INPUT_HEIGHT, INPUT_WIDTH});
    if (heatmap_aware_ && segment_aware_) {
        mt_img_input_shape_[1] = TOTAL_CHNLS;
    } else if (heatmap_aware_) {
        mt_img_input_shape_[1] = TOTAL_CHNLS - SEGMENT_CHNLS;
    } else if (segment_aware_) {
        mt_img_input_shape_[1] = TOTAL_CHNLS - HEATMAP_CHNLS;
    } else {
        mt_img_input_shape_[1] = IMG_CHNLS;
    }
    mt_kpt_input_shape_ = std::vector<int>({batch_size_, VKPT_NUM});
    result_file_ = std::ofstream(initParam.resultPath, std::ofstream::out);
    if (result_file_.fail()) {
        LogError << "Failed to open result file: " << initParam.resultPath;
        return APP_ERR_COMM_FAILURE;
    }
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
    model_poseest_ = std::make_shared<MxBase::ModelInferenceProcessor>();
    ret = model_poseest_->Init(initParam.PoseEstNetPath, poseestnet_desc_);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInferenceProcessor init failed, ret=" << ret << ".";
        return ret;
    }
    model_multitask_ = std::make_shared<MxBase::ModelInferenceProcessor>();
    ret = model_multitask_->Init(initParam.MultiTaskNetPath, multitasknet_desc_);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInferenceProcessor init failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR PAMTRIClassifyOpencv::deinit_() {
    model_poseest_->DeInit();
    model_multitask_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    result_file_.close();
    return APP_ERR_OK;
}

void PAMTRIClassifyOpencv::read_image(const std::string &img_path,
                                          cv::Mat *image_mat) {
    *image_mat = cv::imread(img_path, 1);
}

cv::Mat PAMTRIClassifyOpencv::get_affinetransform(
        const cv::Point2f &center, const cv::Point2f &scale, const float rot,
        const cv::Size &out_size, const cv::Point2f &shift, bool inv) {
    cv::Point2f scale_tmp = scale * PIXEL_STD;
    float src_w = scale_tmp.x;
    int32_t dst_w = out_size.width;
    int32_t dst_h = out_size.height;

    float rot_rad = (PI * rot) / 180;
    cv::Point2f src_dir = get_dir(cv::Point2f(0, src_w * -0.5), rot_rad);
    cv::Point2f dst_dir(0, dst_w * -0.5);

    std::vector<cv::Point2f> src = std::vector<cv::Point2f>(3);
    std::vector<cv::Point2f> dst = std::vector<cv::Point2f>(3);
    src[0] = center + cv::Point2f(scale_tmp.x * shift.x, scale_tmp.y * shift.y);
    src[1] = center + src_dir +
            cv::Point2f(scale_tmp.x * shift.x, scale_tmp.y * shift.y);

    dst[0] = {static_cast<float>(dst_w * 0.5), static_cast<float>(dst_h * 0.5)};
    dst[1] = cv::Point2f(dst_w * 0.5, dst_h * 0.5) + dst_dir;

    src[2] = get_3rd_point(src[0], src[1]);
    dst[2] = get_3rd_point(dst[0], dst[1]);
    cv::Mat trans;
    if (inv) {
        trans = cv::getAffineTransform(dst.data(), src.data());
    } else {
        trans = cv::getAffineTransform(src.data(), dst.data());
    }
    return trans;
}

APP_ERROR hwc2chw_(const cv::Mat &src_image_mat, cv::Mat *dst_image_mat) {
    std::vector<cv::Mat> image;
    cv::split(src_image_mat, image);
    // Stretch one-channel images to vector
    cv::Mat img_r = image[0].reshape(1, 1);
    cv::Mat img_g = image[1].reshape(1, 1);
    cv::Mat img_b = image[2].reshape(1, 1);
    // rearrange channels
    cv::Mat matArray[] = {img_r, img_g, img_b};
    // Concatenate three vectors to one
    cv::hconcat(matArray, 3, *dst_image_mat);
    return APP_ERR_OK;
}

void PAMTRIClassifyOpencv::preprocess_image(const cv::Mat &src_bgr_mat,
                                                   cv::Mat *dst_rgb_mat,
                                                   cv::Mat *mt_rgb_input) {
    uint32_t width = src_bgr_mat.cols;
    uint32_t height = src_bgr_mat.rows;
    cv::cvtColor(src_bgr_mat, *dst_rgb_mat, cv::COLOR_BGR2RGB, 3);

    dst_rgb_mat->copyTo(*mt_rgb_input);
    mt_rgb_input->convertTo(*mt_rgb_input, CV_32F);

    cv::Point2f center(width * 0.5, height * 0.5);
    if (width > height) {
        height = width;
    } else {
        width = height;
    }
    cv::Point2f scale(width * 1.25 / PIXEL_STD, height * 1.25 / PIXEL_STD);
    cv::Mat trans = get_affinetransform(center, scale, 0.0,
                                        cv::Size(INPUT_HEIGHT, INPUT_WIDTH),
                                        cv::Point2f(0, 0), false);
    cv::warpAffine(*dst_rgb_mat, *dst_rgb_mat, trans,
                    cv::Size(INPUT_HEIGHT, INPUT_WIDTH), 1, 0);
    dst_rgb_mat->convertTo(*dst_rgb_mat, CV_32F);

    float *vec = reinterpret_cast<float *>(dst_rgb_mat->data);
    // normalize
    for (int row = 0; row < INPUT_HEIGHT; row++) {
        for (int col = 0; col < INPUT_WIDTH; col++) {
        vec[0] = (vec[0] / 255 - imagenet.mean[0]) / imagenet.std[0];
        vec[1] = (vec[1] / 255 - imagenet.mean[1]) / imagenet.std[1];
        vec[2] = (vec[2] / 255 - imagenet.mean[2]) / imagenet.std[2];
        vec += 3;
        }
    }

    hwc2chw_(*dst_rgb_mat, dst_rgb_mat);
}

APP_ERROR PAMTRIClassifyOpencv::mat_to_tensorbase(
    const cv::Mat &src_mat, MxBase::TensorBase *dst_tensorBase,
    std::vector<int> src_shape) {
    uint32_t data_size = sizeof(float);
    std::vector<uint32_t> shape;
    for (auto i : src_shape) {
        shape.emplace_back((uint32_t)i);
        data_size *= i;
    }

    MemoryData memoryDataDst(data_size, MemoryData::MEMORY_DEVICE, deviceId_);
    MemoryData memoryDataSrc(src_mat.data, data_size,
                            MemoryData::MEMORY_HOST_MALLOC);

    APP_ERROR ret = MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }
    *dst_tensorBase =
        TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_FLOAT32);

    return APP_ERR_OK;
}

APP_ERROR PAMTRIClassifyOpencv::get_posenet_input(
    const std::vector<std::string> &img_path, cv::Mat *pn_input,
    std::vector<cv::Point2f> *center, std::vector<cv::Point2f> *scale,
    std::vector<cv::Mat> *mt_rgb_inputs) {
    *pn_input =
        cv::Mat::zeros(pn_input_shape_.size(), pn_input_shape_.data(), CV_32F);
    *center = std::vector<cv::Point2f>(batch_size_);
    *scale = std::vector<cv::Point2f>(batch_size_);
    *mt_rgb_inputs = std::vector<cv::Mat>(batch_size_);
    for (int b = 0; b < batch_size_; b++) {
        cv::Mat imageMat;
        cv::Mat trans_mat;
        read_image(img_path[b], &imageMat);
        int width = imageMat.cols;
        int height = imageMat.rows;
        (*center)[b].x = width * 0.5;
        (*center)[b].y = height * 0.5;

        if (width > height) {
            height = width;
        } else {
            width = height;
        }
        (*scale)[b].x = width * 1.25 / PIXEL_STD;
        (*scale)[b].y = height * 1.25 / PIXEL_STD;
        preprocess_image(imageMat, &trans_mat, &(*mt_rgb_inputs)[b]);
        memcpy(pn_input->data + b * IMG_CHNLS * INPUT_IMG_SIZE, trans_mat.data,
            IMG_CHNLS * INPUT_IMG_SIZE);
    }
    return APP_ERR_OK;
}

APP_ERROR PAMTRIClassifyOpencv::get_posenet_pred(
    const MxBase::TensorBase &pn_output, std::vector<cv::Point2f> *preds,
    std::vector<float> *maxvals, const std::vector<cv::Point2f> &center,
    const std::vector<cv::Point2f> &scale) {
    *preds = std::vector<cv::Point2f>(batch_size_ * NUM_JOINTS);
    *maxvals = std::vector<float>(batch_size_ * NUM_JOINTS);
    for (int b = 0; b < batch_size_; b++) {
        std::vector<int> idx(NUM_JOINTS);
        cv::Mat trans = get_affinetransform(center[b], scale[b], 0,
                                        cv::Size(SEG_HW[0], SEG_HW[1]),
                                        cv::Point2f(0, 0), true);

        for (int i = 0; i < NUM_JOINTS; i++) {
        float *start_ptr = reinterpret_cast<float *>(
            reinterpret_cast<char *>(pn_output.GetBuffer()) +
            b * PN_OUTPUT_SIZE +
            i * PN_OUTPUT_SIZE / NUM_JOINTS);
        float *max_ptr =
            std::max_element(start_ptr, start_ptr + SEG_HW[0] * SEG_HW[1]);
        idx[i] = max_ptr - start_ptr;
        (*maxvals)[b * NUM_JOINTS + i] = *max_ptr;

        // get the coordinate of heatmap
        if ((*maxvals)[b * NUM_JOINTS + i] > 0) {
            (*preds)[b * NUM_JOINTS + i] =
                cv::Point2f(idx[i] % SEG_HW[0], floor(idx[i] / SEG_HW[1]));
        } else {
            (*preds)[b * NUM_JOINTS + i] = cv::Point2f(0, 0);
        }

        //  if test
        int32_t px = int32_t(floor((*preds)[b * NUM_JOINTS + i].x + 0.5));
        int32_t py = int32_t(floor((*preds)[b * NUM_JOINTS + i].y + 0.5));
        if (1 < px && px < (SEG_HW[0] - 1) && 1 < py && py < (SEG_HW[1] - 1)) {
            float diff[] = {*(start_ptr + SEG_HW[0] * py + px + 1) -
                                *(start_ptr + SEG_HW[1] * py + px - 1),
                            *(start_ptr + SEG_HW[0] * (py + 1) + px) -
                                *(start_ptr + SEG_HW[1] * (py - 1) + px)};
            (*preds)[b * NUM_JOINTS + i] =
                cv::Point2f((*preds)[b * NUM_JOINTS + i].x +
                                (diff[0] > 0 ? 1 : (diff[0] < 0 ? -1 : 0)) * .25,
                            (*preds)[b * NUM_JOINTS + i].y +
                                (diff[1] > 0 ? 1 : (diff[1] < 0 ? -1 : 0)) * .25);
        }

        (*preds)[b * NUM_JOINTS + i] =
            affine_transform((*preds)[b * NUM_JOINTS + i], trans);
        }
    }

    return APP_ERR_OK;
}

bool is_convex(cv::Point kpts[]) {
    int dx0 = kpts[3].x - kpts[2].x;
    int dy0 = kpts[3].y - kpts[2].y;
    int dx1 = kpts[0].x - kpts[3].x;
    int dy1 = kpts[0].y - kpts[3].y;
    int x_prod = dx0 * dy1 - dy0 * dx1;

    int sign_init = sign(x_prod);
    if (sign_init == 0) {
        return false;
    }

    for (int k = 1; k < 4; k++) {
        dx0 = kpts[k - 1].x - kpts[(k - 2) < 0 ? 3 : k - 2].x;
        dy0 = kpts[k - 1].y - kpts[(k - 2) < 0 ? 3 : k - 2].y;
        dx1 = kpts[k].x - kpts[k - 1].x;
        dy1 = kpts[k].y - kpts[k - 1].y;
        x_prod = dx0 * dy1 - dy0 * dx1;
        if (sign_init != static_cast<int>(sign(x_prod))) {
            return false;
        }
    }

    return true;
}

APP_ERROR PAMTRIClassifyOpencv::get_multitask_input(
    const std::vector<cv::Mat> &rgb_mat, const MxBase::TensorBase &pn_output,
    const std::vector<cv::Point2f> &kpt_pred, const std::vector<float> &maxvals,
    cv::Mat *multiTask_img_input, cv::Mat *vkpts) {
    *multiTask_img_input = cv::Mat::zeros(mt_img_input_shape_.size(),
                                            mt_img_input_shape_.data(), CV_32FC1);
    *vkpts = cv::Mat::zeros(mt_kpt_input_shape_.size(),
                            mt_kpt_input_shape_.data(), CV_32FC1);
    float *vkpts_ptr = reinterpret_cast<float *>(vkpts->data);
    uint8_t heatmap_u8[SEG_HW[0] * SEG_HW[1]];
    for (int b = 0; b < mt_img_input_shape_[0]; b++) {
        int height_orig = rgb_mat[b].rows;
        int width_orig = rgb_mat[b].cols;
        cv::Mat img_trans;
        rgb_mat[b].convertTo(img_trans, CV_32F);
        cv::resize(img_trans, img_trans, cv::Size(INPUT_HEIGHT, INPUT_WIDTH));
        hwc2chw_(img_trans, &img_trans);
        // append rgb chnls to multiTaskInputData
        memcpy(multiTask_img_input->data +
               b * mt_img_input_shape_[1] * INPUT_IMG_SIZE,
               img_trans.data, IMG_CHNLS * INPUT_IMG_SIZE);
        // append heatmaps to multiTaskInputData
        if (heatmap_aware_) {
            cv::Mat heatmapTrans;
            auto pix =  reinterpret_cast<float *>(
                reinterpret_cast<char *>(pn_output.GetBuffer()) +
                b * PN_OUTPUT_SIZE);
            for (int i = 0; i < PN_OUTPUT_SIZE / static_cast<int>(sizeof(float)); i++) {
                pix[i] *= 255;
                if (pix[i] < 0) {
                    pix[i] = 0;
                } else if (pix[i] > 255) {
                    pix[i] = 255;
                }
            }
            for (int h = 0; h < NUM_JOINTS; h++) {
                for (int i = 0; i < SEG_HW[0] * SEG_HW[1]; i++) {
                    heatmap_u8[i] =
                        (uint8_t)(static_cast<int>(pix[h * SEG_HW[0] * SEG_HW[1] + i]));
                }
                cv::Mat heatmap(SEG_HW[0], SEG_HW[1], CV_8UC1, heatmap_u8);
                cv::resize(heatmap, heatmapTrans, cv::Size(width_orig, height_orig));
                heatmapTrans.convertTo(heatmapTrans, CV_32FC1);
                cv::resize(heatmapTrans, heatmapTrans, cv::Size(INPUT_HEIGHT, INPUT_WIDTH));
                memcpy(multiTask_img_input->data +
                       b * mt_img_input_shape_[1] * INPUT_IMG_SIZE +
                       (IMG_CHNLS + h) * INPUT_IMG_SIZE, heatmapTrans.data, INPUT_IMG_SIZE);
            }
        }
        if (segment_aware_) {
            for (int s = 0; s < SEGMENT_CHNLS; s++) {
                cv::Mat segment = cv::Mat::zeros(height_orig, width_orig, CV_8UC1);
                cv::Point kpts_seg[1][4];
                bool segment_flag = true;
                for (int i = SEGS[s][0], j = 0; j < 4; j++, i = SEGS[s][j]) {
                    kpts_seg[0][j] = kpt_pred[b * NUM_JOINTS + i];
                    if (maxvals[b * NUM_JOINTS + i] < 0.5) {
                        segment_flag = false;
                    }
                }
                int npts = 4;
                const cv::Point *kpts_[] = {kpts_seg[0]};
                if (is_convex(kpts_seg[0]) && segment_flag) {
                    cv::fillPoly(segment, kpts_, &npts, 1, cv::Scalar(255));
                    cv::resize(segment, segment, cv::Size(SEG_HW[0], SEG_HW[1]));
                    cv::resize(segment, segment, cv::Size(width_orig, height_orig));
                    segment.convertTo(segment, CV_32F);
                    cv::resize(segment, segment, cv::Size(INPUT_HEIGHT, INPUT_WIDTH));
                } else {
                    segment = cv::Mat::zeros(INPUT_HEIGHT, INPUT_WIDTH, CV_32F);
                }
                // append segments to multiTaskInputData
                memcpy(multiTask_img_input->data +
                        b * mt_img_input_shape_[1] * INPUT_IMG_SIZE +
                        (mt_img_input_shape_[1] - SEGMENT_CHNLS + s) * INPUT_IMG_SIZE,
                    segment.data, INPUT_IMG_SIZE);
            }
        }
        float *vec =
            reinterpret_cast<float *>(multiTask_img_input->data +
                                    b * mt_img_input_shape_[1] * INPUT_IMG_SIZE);
        norm_multitask_input(vec, mt_img_input_shape_[1]);
        float width_scale = INPUT_HEIGHT / static_cast<float>(width_orig);
        float height_scale = INPUT_WIDTH / static_cast<float>(height_orig);
        for (int h = 0; h < NUM_JOINTS; h++) {
            //  shift: 0.5
            vkpts_ptr[b * VKPT_NUM + h * 3] =
                (kpt_pred[b * NUM_JOINTS + h].x * width_scale) / INPUT_HEIGHT - 0.5;
            vkpts_ptr[b * VKPT_NUM + h * 3 + 1] =
                (kpt_pred[b * NUM_JOINTS + h].y * height_scale) / INPUT_WIDTH - 0.5;
            vkpts_ptr[b * VKPT_NUM + h * 3 + 2] = maxvals[b * NUM_JOINTS + h] - 0.5;
        }
    }
    return APP_ERR_OK;
}

APP_ERROR PAMTRIClassifyOpencv::posenet_infer(
    const std::vector<MxBase::TensorBase> &inputs,
    std::vector<MxBase::TensorBase> *outputs) {
    auto dtypes = this->model_poseest_->GetOutputDataType();
    for (size_t i = 0; i < this->poseestnet_desc_.outputTensors.size(); ++i) {
        std::vector<uint32_t> shape = {};
        for (size_t j = 0;
            j < this->poseestnet_desc_.outputTensors[i].tensorDims.size(); ++j) {
            shape.push_back(
                (uint32_t)this->poseestnet_desc_.outputTensors[i].tensorDims[j]);
        }
        TensorBase tensor(shape, dtypes[i], MemoryData::MemoryType::MEMORY_DEVICE,
                        deviceId_);
        APP_ERROR ret = TensorBase::TensorBaseMalloc(tensor);
        if (ret != APP_ERR_OK) {
            LogError << "TensorBaseMalloc failed, ret=" << ret << ".";
            return ret;
        }
        outputs->push_back(tensor);
    }
    DynamicInfo dynamicInfo = {};
    dynamicInfo.dynamicType = DynamicType::STATIC_BATCH;
    APP_ERROR ret =
        this->model_poseest_->ModelInference(inputs, *outputs, dynamicInfo);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR PAMTRIClassifyOpencv::multitask_infer(
    const std::vector<MxBase::TensorBase> &inputs,
    std::vector<MxBase::TensorBase> *outputs) {
    auto dtypes = this->model_multitask_->GetOutputDataType();
    for (size_t i = 0; i < this->multitasknet_desc_.outputTensors.size(); ++i) {
        std::vector<uint32_t> shape = {};
        for (size_t j = 0;
            j < this->multitasknet_desc_.outputTensors[i].tensorDims.size(); ++j) {
            shape.push_back(
                (uint32_t)this->multitasknet_desc_.outputTensors[i].tensorDims[j]);
        }
        TensorBase tensor(shape, dtypes[i], MemoryData::MemoryType::MEMORY_DEVICE,
                        deviceId_);
        APP_ERROR ret = TensorBase::TensorBaseMalloc(tensor);
        if (ret != APP_ERR_OK) {
            LogError << "TensorBaseMalloc failed, ret=" << ret << ".";
            return ret;
        }
        outputs->push_back(tensor);
    }
    DynamicInfo dynamicInfo = {};
    dynamicInfo.dynamicType = DynamicType::STATIC_BATCH;

    APP_ERROR ret =
        this->model_multitask_->ModelInference(inputs, *outputs, dynamicInfo);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR PAMTRIClassifyOpencv::flip_(const cv::Mat &pn_input,
                                     MxBase::TensorBase *orig_output) {
    std::vector<MxBase::TensorBase> pn_inputs;
    std::vector<MxBase::TensorBase> pn_outputs;
    MxBase::TensorBase pn_tensorBase;
    auto pn_indata =
        reinterpret_cast<float(*)[IMG_CHNLS][INPUT_HEIGHT][INPUT_WIDTH]>(
            pn_input.data);
    float temp;
    for (int b = 0; b < batch_size_; b++) {
        // flip input data
        for (int c = 0; c < IMG_CHNLS; c++) {
            for (int h = 0; h < INPUT_HEIGHT; h++) {
                for (int w = 0; w < INPUT_WIDTH / 2; w++) {
                temp = pn_indata[b][c][h][w];
                pn_indata[b][c][h][w] = pn_indata[b][c][h][255 - w];
                pn_indata[b][c][h][255 - w] = temp;
                }
            }
        }
    }

    APP_ERROR ret = mat_to_tensorbase(pn_input, &pn_tensorBase, pn_input_shape_);
    if (ret != APP_ERR_OK) {
        LogError << "CVMatToTensorBase failed, ret=" << ret << ".";
        return ret;
    }

    pn_inputs.push_back(pn_tensorBase);
    ret = posenet_infer(pn_inputs, &pn_outputs);
    if (ret != APP_ERR_OK) {
        LogError << "Flip PoseEstNetInference failed, ret=" << ret << ".";
        return ret;
    }
    MxBase::TensorBase &tensorPoseNet = pn_outputs[0];
    ret = tensorPoseNet.ToHost();
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Tensor deploy to host failed.";
        return ret;
    }
    // flip output data
    auto pn_out_data =
        reinterpret_cast<float(*)[NUM_JOINTS][SEG_HW[0]][SEG_HW[1]]>(
            tensorPoseNet.GetBuffer());
    for (int b = 0; b < batch_size_; b++) {
        for (int c = 0; c < NUM_JOINTS; c++) {
            for (int h = 0; h < SEG_HW[0]; h++) {
                for (int w = 0; w < SEG_HW[0] / 2; w++) {
                temp = pn_out_data[b][c][h][w];
                pn_out_data[b][c][h][w] = pn_out_data[b][c][h][63 - w];
                pn_out_data[b][c][h][63 - w] = temp;
                }
            }
        }
    }
    float temp_f[SEG_HW[0] * SEG_HW[1]];
    for (int b = 0; b < batch_size_; b++) {
        auto pn_output_f =
            reinterpret_cast<float(*)[NUM_JOINTS][SEG_HW[0] * SEG_HW[1]]>(
                tensorPoseNet.GetBuffer());
        for (int i = 0; i < 18; i++) {
            // sizeof(float32): 4
            memcpy(temp_f, pn_output_f[b][FLIP_PAIRS[i][0]],
                    SEG_HW[0] * SEG_HW[1] * 4);
            memcpy(pn_output_f[b][FLIP_PAIRS[i][0]], pn_output_f[b][FLIP_PAIRS[i][1]],
                    SEG_HW[0] * SEG_HW[1] * 4);
            memcpy(pn_output_f[b][FLIP_PAIRS[i][1]], temp_f,
                    SEG_HW[0] * SEG_HW[1] * 4);
        }
    }
    // shift
    auto pn_output_s =
        reinterpret_cast<float(*)[SEG_HW[1]]>(tensorPoseNet.GetBuffer());
    float temp_s[SEG_HW[1]];
    for (int i = 0; i < batch_size_ * NUM_JOINTS * SEG_HW[0]; i++) {
        // sizeof(float): 4
        memcpy(temp_s, pn_output_s[i], SEG_HW[1] * 4);
        memcpy(pn_output_s[i] + 1, temp_s, SEG_HW[1] * 4);
    }

    float *pn_output_p = reinterpret_cast<float *>(tensorPoseNet.GetBuffer());
    float *ori_output_p = reinterpret_cast<float *>(orig_output->GetBuffer());
    for (int i = 0; i < batch_size_ * NUM_JOINTS * SEG_HW[0] * SEG_HW[1]; i++) {
        ori_output_p[i] = (ori_output_p[i] + pn_output_p[i]) * 0.5;
    }

    return APP_ERR_OK;
}

APP_ERROR PAMTRIClassifyOpencv::process_(
    const std::vector<std::string> &imgPath) {
    batch_size_ = imgPath.size();
    std::vector<cv::Point2f> center;
    std::vector<cv::Point2f> scale;
    std::vector<cv::Mat> mt_rgb_input;
    std::vector<MxBase::TensorBase> pn_inputs = {};
    std::vector<MxBase::TensorBase> pn_outputs = {};
    std::vector<MxBase::TensorBase> mt_inputs = {};
    std::vector<MxBase::TensorBase> mt_outputs = {};
    TensorBase pn_tensorBase;
    TensorBase mt_img_tensorBase;
    TensorBase mt_vkpts_tensorBase;
    cv::Mat pn_input;
    APP_ERROR ret = get_posenet_input(imgPath, &pn_input, &center, &scale,
                                            &mt_rgb_input);
    if (ret != APP_ERR_OK) {
        LogError << "GetPoseEstNetInputData failed, ret=" << ret << ".";
        return ret;
    }
    ret = mat_to_tensorbase(pn_input, &pn_tensorBase, pn_input_shape_);
    if (ret != APP_ERR_OK) {
        LogError << "CVMatToTensorBase failed, ret=" << ret << ".";
        return ret;
    }
    pn_inputs.push_back(pn_tensorBase);

    ret = posenet_infer(pn_inputs, &pn_outputs);
    if (ret != APP_ERR_OK) {
        LogError << "PoseEstNetInference failed, ret=" << ret << ".";
        return ret;
    }
    // postprocess  posenet output data
    MxBase::TensorBase &pn_output = pn_outputs[0];
    ret = pn_output.ToHost();
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Tensor deploy to host failed.";
        return ret;
    }

    ret = flip_(pn_input, &pn_output);
    if (ret != APP_ERR_OK) {
        LogError << "Flip failed, ret=" << ret << ".";
        return ret;
    }
    std::vector<cv::Point2f> preds;
    std::vector<float> maxvals;

    get_posenet_pred(pn_output, &preds, &maxvals, center, scale);
    cv::Mat mt_img_input;
    cv::Mat vkpts;

    get_multitask_input(mt_rgb_input, pn_output, preds, maxvals, &mt_img_input,
                            &vkpts);
    ret =
        mat_to_tensorbase(mt_img_input, &mt_img_tensorBase, mt_img_input_shape_);
    if (ret != APP_ERR_OK) {
        LogError << "CVMatToTensorBase failed, ret=" << ret << ".";
        return ret;
    }
    ret = mat_to_tensorbase(vkpts, &mt_vkpts_tensorBase, mt_kpt_input_shape_);
    if (ret != APP_ERR_OK) {
        LogError << "CVMatToTensorBase failed, ret=" << ret << ".";
        return ret;
    }
    mt_inputs.push_back(mt_img_tensorBase);
    mt_inputs.push_back(mt_vkpts_tensorBase);

    ret = multitask_infer(mt_inputs, &mt_outputs);
    if (ret != APP_ERR_OK) {
        LogError << "MultiTaskNetInference failed, ret=" << ret << ".";
        return ret;
    }

    ret = save_result(imgPath, mt_outputs);
    if (ret != APP_ERR_OK) {
        LogError << "Save infer results into file failed. ret = " << ret << ".";
        return ret;
    }

    return APP_ERR_OK;
}

APP_ERROR PAMTRIClassifyOpencv::save_result(
    const std::vector<std::string> &imgPath,
    const std::vector<MxBase::TensorBase> &mt_outputs) {
    MxBase::TensorBase colors = mt_outputs[1];
    MxBase::TensorBase types = mt_outputs[2];
    APP_ERROR ret;
    ret = colors.ToHost();
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Tensor deploy to host failed.";
        return ret;
    }
    ret = types.ToHost();
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Tensor deploy to host failed.";
        return ret;
    }
    std::vector<uint32_t> colors_shape = colors.GetShape();
    std::vector<uint32_t> types_shape = types.GetShape();
    auto colors_p =
        reinterpret_cast<float(*)[colors_shape[1]]>(colors.GetBuffer());
    auto types_p = reinterpret_cast<float(*)[types_shape[1]]>(types.GetBuffer());
    std::string resultStr;
    for (uint32_t b = 0; b < colors_shape[0]; b++) {
        std::string fileName = imgPath[b].substr(imgPath[b].find_last_of("/") + 1);

        float *start = colors_p[b];
        float *max_color_conf = std::max_element(colors_p[b], colors_p[b + 1]);
        size_t color = max_color_conf - start + 1;
        start = types_p[b];
        float *max_type_conf = std::max_element(types_p[b], types_p[b + 1]);
        size_t type = max_type_conf - start + 1;
        resultStr += fileName + std::to_string(b) +
                    " color:" + std::to_string(color) +
                    " type:" + std::to_string(type);
    }
    result_file_ << resultStr << std::endl;

    return APP_ERR_OK;
}

PAMTRIClassifyOpencv::PAMTRIClassifyOpencv(const InitParam &initParam) {
    segment_aware_ = initParam.segmentAware;
    heatmap_aware_ = initParam.heatmapAware;
    batch_size_ = initParam.batchSize;
}
