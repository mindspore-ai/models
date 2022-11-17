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
#ifndef OPENPOSEPOSTPROCESS_OPENPOSEPOSTPROCESS_H
#define OPENPOSEPOSTPROCESS_OPENPOSEPOSTPROCESS_H
#include <vector>
#include "MxTools/PluginToolkit/base/MxPluginGenerator.h"
#include "MxTools/PluginToolkit/base/MxPluginBase.h"
#include "MxTools/PluginToolkit/metadata/MxpiMetadataManager.h"
#include "MxTools/Proto/MxpiDataType.pb.h"
#include "MxBase/ErrorCode/ErrorCode.h"
#include "MxBase/PostProcessBases/PostProcessBase.h"
#include "MxBase/PostProcessBases/ObjectPostProcessBase.h"
#include "mxpiOpenposeProto.pb.h"
#include "opencv4/opencv2/opencv.hpp"

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

namespace  MxBase {
class OpenPoseMindsporePost : public ObjectPostProcessBase {
 public:
    OpenPoseMindsporePost() = default;

    ~OpenPoseMindsporePost() = default;

    OpenPoseMindsporePost(const OpenPoseMindsporePost &other) = default;

    OpenPoseMindsporePost &operator=(const OpenPoseMindsporePost &other);

    APP_ERROR selfProcess(const std::vector<TensorBase> &tensors,
        const std::vector<int> &vision_infos, std::vector<std::vector<PartPair> > *person_list);

    void GeneratePersonList(const std::vector<TensorBase> &tensors,
        const std::vector<int> &vision_infos, std::vector<std::vector<PartPair> > *person_list);

    void ResizeHeatmaps(const std::vector<int> &vision_infos, std::vector<cv::Mat> *keypoint_heatmap,
        std::vector<cv::Mat > *paf_heatmap);

    void ExtractKeypoints(const std::vector<cv::Mat> &keypoint_heatmap,
        std::vector<std::vector<cv::Point> > *coor, std::vector<std::vector<float> > *coor_score);

    void GroupKeypoints(const std::vector<cv::Mat>& paf_heatmap, const std::vector<std::vector<cv::Point> > &coor,
        const std::vector<std::vector<float> > &coor_score, std::vector<std::vector<PartPair> > *person_list);

    void ScoreSkeletons(const int part_idx, const std::vector<std::vector<cv::Point> > &coor,
        const std::vector<std::vector<float> > &coor_score, const std::vector<cv::Mat> &paf_heatmap,
        std::vector<PartPair> *connections);

    std::vector<float> OneSkeletonScore(const cv::Mat &paf_x, const cv::Mat &paf_y,
        const std::vector<cv::Point> &endpoints);

    void ConntectionNms(std::vector<PartPair> *src, std::vector<PartPair> *dst);

    bool MergeSkeletonToPerson(std::vector<std::vector<PartPair> > *person_list, PartPair current_pair);

    float PersonScore(const std::vector<PartPair> &person);
};
#endif  // OpenPose_MINSPORE_PORT_H
}  // namespace MxBase
