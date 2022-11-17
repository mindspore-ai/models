/*
 * Copyright(C) 2021. Huawei Technologies Co.,Ltd.
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

#ifndef OPENPOSEPOSTPROCESS_MXPIOPENPOSEPOSTPROCESS_H
#define OPENPOSEPOSTPROCESS_MXPIOPENPOSEPOSTPROCESS_H
#include <vector>
#include <map>
#include <memory>
#include <string>
#include "MxTools/PluginToolkit/base/MxPluginGenerator.h"
#include "MxTools/PluginToolkit/base/MxPluginBase.h"
#include "MxTools/PluginToolkit/metadata/MxpiMetadataManager.h"
#include "MxTools/Proto/MxpiDataType.pb.h"
#include "MxBase/ErrorCode/ErrorCode.h"
#include "mxpiOpenposeProto.pb.h"
#include "opencv2/opencv.hpp"

/**
* @api
* @brief Definition of MxpiOpenposePostProcess class.
*/

namespace  MxPlugins {
struct PartPair {
    float score;
    int partIdx1;
    int partIdx2;
    int idx1;
    int idx2;
    std::vector<float> coord1;
    std::vector<float> coord2;
    float score1;
    float score2;
};

class MxpiOpenposePostProcess : public MxTools::MxPluginBase {
 public:
    MxpiOpenposePostProcess() = default;

    ~MxpiOpenposePostProcess() = default;
    /**
      * @brief Initialize configure parameter.
      * @param config_param_map
      * @return APP_ERROR
      */
    APP_ERROR Init(std::map<std::string, std::shared_ptr<void>> &config_param_map) override;

    /**
      * @brief DeInitialize configure parameter.
      * @return APP_ERROR
      */
    APP_ERROR DeInit() override;

    /**
      * @brief Process the data of MxpiBuffer.
      * @param mxpi_buffer
      * @return APP_ERROR
      */
    APP_ERROR Process(std::vector<MxTools::MxpiBuffer*> &mxpi_buffer) override;

    /**
      * @brief Definition the parameter of configure properties.
      * @return std::vector<std::shared_ptr<void>>
      */
    static std::vector<std::shared_ptr<void>> DefineProperties();

    /**
      * Overall process to generate all person skeleton information
      * @param image_decoder_visionListSptr - Source MxpiVisionList containing vision data about input and aligned image 
      * @param src_mxpi_tensor_package - Source MxpiTensorPackage containing heatmap data
      * @param dst_mxpi_person_list - Target MxpiPersonList containing detection result list
      * @return APP_ERROR
      */
    APP_ERROR GeneratePersonList(const MxTools::MxpiVisionList image_decoder_visionListSptr,
        const MxTools::MxpiTensorPackageList src_mxpi_tensor_package,
        mxpiopenposeproto::MxpiPersonList *dst_mxpi_person_list);

    /**
      * @brief Resize output heatmaps to the size of the origin image
      * @param keypoint_heatmap - Keypoint heatmap, each channel of the heatmap is stored as a Mat
      * @param paf_heatmap - PAF heatmap, each channel of the heatmap is stored as a Mat
      * @param vision_infos - Vision infos of origin image and aligned image
      * @return APP_ERROR
      */
    APP_ERROR ResizeHeatmaps(const std::vector<int> &vision_infos, std::vector<cv::Mat> *keypoint_heatmap,
        std::vector<cv::Mat > *paf_heatmap);

    /**
      * @brief Extract candidate keypoints from output heatmap
      * @param keypoint_heatmap - Keypoint heatmap stored in vector
      * @param coor - Keep coor for candidate keypoints by category
      * @param coor_score - Keep coor score for candidate keypoints by category
      * @return APP_ERROR
      */
    APP_ERROR ExtractKeypoints(const std::vector<cv::Mat> &keypoint_heatmap,
        std::vector<std::vector<cv::Point> > *coor, std::vector<std::vector<float> > *coor_score);

    /**
      * @brief Group keypoints to skeletons and assemble them to person
      * @param paf_heatmap - PAF heatmap
      * @param coor - Coordinates of all the candidate keypoints
      * @param coor_score - Corresponding score of coordinates
      * @param person_list - Target vector to store person, each person is stored as a vector of skeletons
      * @return APP_ERROR
      */
    APP_ERROR GroupKeypoints(const std::vector<cv::Mat>& paf_heatmap,
        const std::vector<std::vector<cv::Point> > &coor, const std::vector<std::vector<float> > &coor_score,
        std::vector<std::vector<PartPair> > *person_list);

    /**
      * @brief Calculate expected confidence of each possible skeleton and choose candidates
      * @param part_idx - Index of skeleton in kPoseBodyPartSkeletons
      * @param coor - Candidate positions of endpoints
      * @param coor_score - Corresponding score of coor
      * @param paf_heatmap - PAF heatmap
      * @param connections - Target vector that collects candidate skeletons
      * @return APP_ERROR
      */
    APP_ERROR ScoreSkeletons(const int part_idx, const std::vector<std::vector<cv::Point> > &coor,
        const std::vector<std::vector<float> > &coor_score, const std::vector<cv::Mat> &paf_heatmap,
        std::vector<PartPair> *connections);

    /**
      * @brief Compute expected confidence for each candidate skeleton
      * @param endpoints - Coordinates of the two end points of a skeleton
      * @param paf_x - PAF heatmap of x coordinate
      * @param paf_y - PAF heatmap of y coordinate
      * @return result - Keep confidence information of this skeleton in the form:
      * [confidence score, number of successfully hit sub points]
      */
    std::vector<float> OneSkeletonScore(const cv::Mat &paf_x, const cv::Mat &paf_y,
        const std::vector<cv::Point> &endpoints);

    /**
      * @brief Remove duplicate skeletons
      * @param src - Source vector that stores skeletons to be processed
      * @param dst - Target vector that collects filter skeletons
      * @return APP_ERROR
      */
    APP_ERROR ConntectionNms(std::vector<PartPair> *src, std::vector<PartPair> *dst);

    /**
      * @brief Merge a skeleton to an existed person
      * @param person_list - Currently existed person list
      * @param current_pair - Skeleton to be merged
      * @return True if merged successfully, otherwise false
      */
    bool MergeSkeletonToPerson(std::vector<std::vector<PartPair> > *person_list, PartPair current_pair);

    /**
      * @brief Calculate score of a person according to its skeletons
      * @param person - Target person
      * @return Score value
      */
    float PersonScore(const std::vector<PartPair> &person);

    /**
      * @brief Prepare output in the format of MxpiPersonList
      * @param person_list - Source data in the format of std::vector<std::vector<PartPair> >
      * @param dst_mxpi_person_list - Target data in the format of MxpiPersonList
      * @return
      */
    APP_ERROR GenerateMxpiOutput(const std::vector<std::vector<PartPair> > &person_list,
        mxpiopenposeproto::MxpiPersonList *dst_mxpi_person_list);

 private:
    APP_ERROR SetMxpiErrorInfo(const std::string &plugin_name,
        const MxTools::MxpiErrorInfo &mxpi_error_info, MxTools::MxpiBuffer *buffer);
    std::string parentName_;
    std::string imageDecoderName_;
    std::uint32_t inputHeight_;
    std::uint32_t inputWidth_;
    std::ostringstream ErrorInfo_;
};
}  // namespace MxPlugins
#endif  // OPENPOSEPOSTPROCESS_MXPIOPENPOSEPOSTPROCESS_H
