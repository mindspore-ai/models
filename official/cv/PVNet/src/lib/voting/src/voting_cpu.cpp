/*Copyright 2021 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
============================================================================
*/

#include <random>     // random
#include <algorithm>  // shuffle
#include <ctime>
#include "include/voting_cpu.h"

namespace {
    inline uint32_t RandomUInt32Range(uint32_t end, uint32_t start = 0) {
        std::random_device rd;
        std::mt19937 engine(rd());
        std::uniform_int_distribution<uint32_t> dist(start, end);
        return dist(engine);
    }

    void RandomShuffle(std::vector<uint32_t> &array) {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(array.begin(), array.end(), g);
    }

    int ArgmaxArray(const std::vector<uint32_t>& array, int start = 0) {
        uint32_t maxClass = start;
        uint32_t maxClassCount = array[start];
        for (uint32_t i = start + 1; i < array.size(); i++) {  // start form class 2
            if (maxClassCount < array[i]) {
                maxClassCount = array[i];
                maxClass = i;
            }
        }
        return maxClass;
    }

    int ArgmaxClassId(const std::vector<const float* >& maskMaps, uint32_t max, uint32_t index) {
        auto maxClassValue = maskMaps[0][index];
        int32_t maxClassId = 0;

        for (uint32_t k = 1; k < max; k++) {
            if (maxClassValue < maskMaps[k][index]) {
                maxClassValue = maskMaps[k][index];
                maxClassId = k;
            }
        }
        return maxClassId;
    }
}  // namespace

int32_t VotingProcess::Init(const ModelShape &outputShape, int32_t classNum, int32_t controlPointNum) {
    mClassMax = classNum;
    mControlPoints = controlPointNum;
    mNetShape = outputShape;

    mImgSize = mNetShape.H * mNetShape.W;
    mSegMask.resize(mImgSize, 0);
    mVectorMapAddr.resize(mControlPoints * 2);

    if (mSegMask.size() < mImgSize || mVectorMapAddr.size() < mControlPoints * 2) {
        LOGI("alloc memory fail");
        return -1;
    }

    mInitial = true;
    return 0;
}

int32_t VotingProcess::PostProcess(const std::vector<float> &data, int32_t &classId, std::vector<float> &box2D) {
    if (!mInitial) {
        LOGE("not initialize");
        return -1;
    }

    if (box2D.size() != mControlPoints * 2) {
        LOGE("error shape: points.length:%zu ", box2D.size());
        return -1;
    }

    if (data.size() != (mImgSize * mNetShape.C)) {
        LOGE("buffer length:%zu   mOutputWxH:%u  C:%u", data.size(), mImgSize, mNetShape.C);
        return -1;
    }

    const float *vectorMapAddr = data.data() + mClassMax * mImgSize;
    for (uint32_t k = 0; k < mControlPoints; k++) {
        mVectorMapAddr[2 * k] = vectorMapAddr + mImgSize * k * 2;
        mVectorMapAddr[2 * k + 1] = vectorMapAddr + mImgSize * (k * 2 + 1);
    }

    int32_t ret = FindClassAndMask(data, classId);
    if (ret != 0) {
        LOGW("ProcessMaskAndMap fail");
        return -1;
    }

    for (uint32_t i = 0; i < static_cast<uint32_t>(mControlPoints); i++) {
        Vector2f keypoint = SelectVoteKeyPoint(i);
        box2D[i * 2 + 0] = keypoint.x;
        box2D[i * 2 + 1] = keypoint.y;
    }

    return 0;
}

int32_t VotingProcess::FindClassAndMask(const std::vector<float> &data, int32_t &classId) {
    std::vector<const float* > maskMaps(mClassMax);
    if (maskMaps.size() < static_cast<size_t>(mClassMax)) {
        return -1;
    }

    for (uint32_t i = 0; i < mClassMax; i++) {
        maskMaps[i] = data.data() + mImgSize * i;
    }

    // get seg mask
    std::vector<uint32_t> classCount(mClassMax, 0);  // 0- 10
    for (uint32_t i = 0; i < mImgSize; i++) {
        int32_t maxClassId = ArgmaxClassId(maskMaps, mClassMax, i);
        mSegMask[i] = maxClassId;
        classCount[maxClassId]++;
    }

    // find class which's counts in mSegMask is biggest
    int32_t maxClass = ArgmaxArray(classCount, 1);  // default class 1
    uint32_t maxClassCount = classCount[maxClass];

    // set mSegMask other class pixel label to 0(background)
    mPixel2DInMask.assign(maxClassCount, 0);
    classId = maxClass;

    uint32_t cnt = 0;
    for (uint32_t i = 0; i < static_cast<uint32_t>(mImgSize); i++) {
        if (mSegMask[i] == maxClass) {
            mPixel2DInMask[cnt++] = i;
        }
    }

    if (maxClassCount < mOptions.minNum) {
        LOGE("foreground class:%d  nums :%u  not satisfy", maxClass, maxClassCount);
        return -1;
    }

    if (mPixel2DInMask.size() > static_cast<size_t>(mOptions.maxNum)) {
        RandomShuffle(mPixel2DInMask);
        mPixel2DInMask.erase(mPixel2DInMask.begin() + mOptions.maxNum, mPixel2DInMask.end());
        std::sort(mPixel2DInMask.begin(), mPixel2DInMask.end());
    }

    LOGI("mask pixel count: %zu, maxClassCount:%u", mPixel2DInMask.size(), maxClassCount);
    return 0;
}

Vector2f VotingProcess::SelectVoteKeyPoint(uint32_t index) const {
    Vector2f keypoint(-1.0f, -1.0f);
    std::vector<uint32_t> bestInlierIndex;

    for (uint32_t i = 0; i < mOptions.maxIter; i++) {
        Vector2f hypo = GetHypoPoint(index);
        std::vector<uint32_t> inlierIndex;

        for (unsigned int pointIndex : mPixel2DInMask) {
            const Vector2f v = (hypo - PixelIndex2XY(pointIndex)).Normalized();
            const Vector2f voteVec = GetVoteVec(pointIndex, index).Normalized();
            float dot = v.Dot(voteVec);

            if (dot > mOptions.inlierThresh) {
                inlierIndex.push_back(pointIndex);
            }
        }

        if (inlierIndex.size() > bestInlierIndex.size()) {
            bestInlierIndex.swap(inlierIndex);
            keypoint = hypo;
        }

        float currentRatio = static_cast<float>(inlierIndex.size()) / mPixel2DInMask.size();
        float conf = 1.0f - pow((1 - currentRatio * currentRatio), i + 1);
        if (conf > mOptions.confidence) {
            break;
        }
    }

    if (bestInlierIndex.size() > 20) {  // inliners count > 20, we refine the key point with inliners
        Vector2f refinePoint;
        if (RefinePosition(index, bestInlierIndex, refinePoint) == 0) {
            keypoint = refinePoint;
        }
    }
    return keypoint;
}

int32_t VotingProcess::RefinePosition(uint32_t kpIndex, const std::vector<uint32_t> &inliers,
                                      Vector2f &keypoint) const {
    Matrix2f matrixAtA(0.0f, 0.0f, 0.0f, 0.0f);
    Vector2f matrixAtB(0.0f, 0.0f);

    for (int32_t index : inliers) {
        Vector2f normal = GetNormal(index, kpIndex);
        Vector2f votePoint = PixelIndex2XY(index);

        matrixAtA += Matrix2f::FromAxB(normal, normal);
        matrixAtB[0] += normal.x * normal.Dot(votePoint);
        matrixAtB[1] += normal.y * normal.Dot(votePoint);
    }

    if (std::fabs(matrixAtA.DetA()) < FLT_EPSILON) {
        LOGE("matrixAtA det is too small");
        return -1;
    }

    keypoint = matrixAtA.Inverse() * matrixAtB;
    return 0;
}

Vector2f VotingProcess::GetHypoPoint(uint32_t kpIndex) const {
    Vector2f output(-1.0, -1.0);
    size_t pointCount = mPixel2DInMask.size();
    uint32_t base = mPixel2DInMask[RandomUInt32Range(pointCount - 1, 0)];  // random from [0, pointIdx.size() - 1]
    Vector2f normal0 = GetNormal(base, kpIndex);
    auto dotX = normal0.Dot(PixelIndex2XY(base));

    const uint32_t maxTryTimes = 3;
    for (uint32_t i = 0; i < maxTryTimes; i++) {
        uint32_t other = mPixel2DInMask[RandomUInt32Range(pointCount - 1, 0)];
        Vector2f normal1 = GetNormal(other, kpIndex);
        Matrix2f matrixA(normal0, normal1);
        auto dotY = normal1.Dot(PixelIndex2XY(other));

        if (std::fabs(matrixA.DetA()) > FLT_EPSILON) {
            Vector2f dotVector(dotX, dotY);
            output = matrixA.Inverse() * dotVector;
            break;
        }
    }
    return output;
}
