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

#ifndef MXBASE_BRDNET_H
#define MXBASE_BRDNET_H

#include <memory>
#include <utility>
#include <vector>
#include <string>
#include <map>
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"

extern std::vector<double> g_inferCost;

struct InitParam {
    uint32_t deviceId;
    std::string modelPath;
    std::string outputDataPath;
};

class R2P1D {
 public:
    APP_ERROR Init(const InitParam &initParam);
    APP_ERROR DeInit();
    APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> *outputs);
    APP_ERROR Process(const std::string &inferPath, const std::string &fileName);

 protected:
    APP_ERROR ReadTensorFromFile(const std::string &file, float *data);
    APP_ERROR ReadInputTensor(const std::string &fileName, std::vector<MxBase::TensorBase> *inputs);
    APP_ERROR WriteResult(const std::string &imageFile, std::vector<MxBase::TensorBase> outputs);

 private:
    std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
    MxBase::ModelDesc modelDesc_ = {};
    uint32_t deviceId_ = 0;
    std::string outputDataPath_ = "./result";
    std::vector<uint32_t> inputDataShape_ = {1, 3, 16, 112, 112};
    uint32_t inputDataSize_ = 602112;
    std::vector<std::string> ucf101_label_names = {"ApplyEyeMakeup", "ApplyLipstick", "Archery", \
                      "BabyCrawling", "BalanceBeam", \
                      "BandMarching", "BaseballPitch", "Basketball", "BasketballDunk", "BenchPress", \
                      "Biking", "Billiards", "BlowDryHair", "BlowingCandles", "BodyWeightSquats", \
                      "Bowling", "BoxingPunchingBag", "BoxingSpeedBag", "BreastStroke", "BrushingTeeth", \
                      "CleanAndJerk", "CliffDiving", "CricketBowling", "CricketShot", "CuttingInKitchen", \
                      "Diving", "Drumming", "Fencing", "FieldHockeyPenalty", "FloorGymnastics", \
                      "FrisbeeCatch", "FrontCrawl", "GolfSwing", "Haircut", "HammerThrow", \
                      "Hammering", "HandstandPushups", "HandstandWalking", "HeadMassage", "HighJump", \
                      "HorseRace", "HorseRiding", "HulaHoop", "IceDancing", "JavelinThrow", \
                      "JugglingBalls", "JumpRope", "JumpingJack", "Kayaking", "Knitting", \
                      "LongJump", "Lunges", "MilitaryParade", "Mixing", "MoppingFloor", \
                      "Nunchucks", "ParallelBars", "PizzaTossing", "PlayingCello", "PlayingDaf", \
                      "PlayingDhol", "PlayingFlute", "PlayingGuitar", "PlayingPiano", "PlayingSitar", \
                      "PlayingTabla", "PlayingViolin", "PoleVault", "PommelHorse", "PullUps", \
                      "Punch", "PushUps", "Rafting", "RockClimbingIndoor", "RopeClimbing", \
                      "Rowing", "SalsaSpin", "ShavingBeard", "Shotput", "SkateBoarding", \
                      "Skiing", "Skijet", "SkyDiving", "SoccerJuggling", "SoccerPenalty", \
                      "StillRings", "SumoWrestling", "Surfing", "Swing", "TableTennisShot", \
                      "TaiChi", "TennisSwing", "ThrowDiscus", "TrampolineJumping", "Typing", \
                      "UnevenBars", "VolleyballSpiking", "WalkingWithDog", "WallPushups", "WritingOnBoard", \
                      "YoYo"};
};

#endif
