/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include <sys/time.h>
#include <dirent.h>
#include <iostream>
#include <string>
#include <algorithm>
#include <iosfwd>
#include <vector>
#include <fstream>
#include <sstream>
#include <cstdint>
#include <random>

#include "../inc/utils.h"
#include "../inc/maxlen.h"
#include "common_inc/infer.h"
#include "third_party/src/common.h"

const std::string change_ext(const std::string& imageFile) {
    std::string homePath = "./result_Files";

    int pos = imageFile.rfind('/');
    std::string fileName(imageFile, pos + 1);
    fileName.replace(fileName.find('.'), fileName.size() - fileName.find('.'), ".pcm");
    std::string outFileName = homePath + "/" + fileName;

    return outFileName;
}

int sample_multinomial(const float *ptrProbs, std::default_random_engine &generator, int n, double corr) {
    std::vector<double> probs(n);
    for (size_t i=0; i < probs.size(); i++) {
        probs[i] = static_cast<double>(ptrProbs[i]);
    }

    corr = 1.5 * corr - 0.5;
    corr = corr > 0 ? corr : 0;

    double norm = 1e-18;
    for (size_t i=0; i < probs.size(); i++) {
        probs[i] *= pow(probs[i], corr);
        norm += probs[i];
    }

    double norm2 = 1e-8;
    for (size_t i = 0; i < probs.size(); i++) {
        probs[i] /= norm;
        probs[i] -= 0.002;
        probs[i] = probs[i] > 0 ? probs[i] : 0;
        norm2 += probs[i];
    }

    for (size_t i = 0; i < probs.size(); i++) {
        probs[i] /= norm2;
    }

    std::discrete_distribution<int> distribution(probs.begin(), probs.end());
    return distribution(generator);
}

DEFINE_string(encoder_path, "", "encoder path");
DEFINE_string(decoder_path, "", "decoder path");
DEFINE_string(dataset_path, ".", "dataset path");
DEFINE_int32(device_id, 0, "device id");
DEFINE_string(precision_mode, "preferred_fp32", "precision mode");
DEFINE_string(op_select_impl_mode, "", "op select impl mode");
DEFINE_string(aipp_path, "./aipp.cfg", "aipp path");
DEFINE_string(device_target, "Ascend310", "device target");
DEFINE_string(device_type, "CPU", "device type");

void preprocess_encoder(const std::string &file_name, std::vector<float> &features,
                        std::vector<MSTensor> &encoderInputs, std::vector<MSTensor> &inputs) {
    // Read features
    ReadFileToVector(file_name, features);
    size_t real_len = features.size() / 36;
    for (size_t i = real_len; i < MAXLEN; i++) {
        for (int j = 0; j < 36; j++) {
            features.emplace_back(0.);
        }
    }
    features.erase(features.begin()+ (MAXLEN * 36), features.end());

    std::vector<float> used_features(features.size() / 36 * 20);
    for (size_t i = 0; i < features.size() / 36; i++) {
        for (int j = 0; j < 20; j++) {
            used_features[i * 20 + j] = features[i * 36 + j];
        }
    }

    // Calculate pitch
    std::vector<int> pitch;
    for (unsigned int j = 18; j < features.size(); j+=36) {
        pitch.emplace_back(static_cast<int>(.1 + 50 * features[j] + 100));
    }

    // Create encoder inputs
    std::shared_ptr<float> sptrUsedFeatures(new float[used_features.size()], std::default_delete<float[]>());
    for (size_t i = 0; i < used_features.size(); ++i) {
        sptrUsedFeatures.get()[i] = used_features[i];
    }
    inputs.emplace_back(encoderInputs[0].Name(), encoderInputs[0].DataType(), encoderInputs[0].Shape(),
                (const void *)sptrUsedFeatures.get(), used_features.size() * sizeof(float));

    std::shared_ptr<int> sptrPitch(new int[pitch.size()], std::default_delete<int[]>());
    for (size_t i = 0; i < pitch.size(); ++i) {
        sptrPitch.get()[i] = pitch[i];
    }
    inputs.emplace_back(encoderInputs[1].Name(), encoderInputs[1].DataType(), encoderInputs[1].Shape(),
                (const void *)sptrPitch.get(), pitch.size() * sizeof(int));
}

void predict_file(const std::string &file_name, Model &encoder, std::vector<MSTensor> &encoderInputs, Model &decoder,
                  std::vector<MSTensor> &decoderInputs, struct timeval &start, struct timeval &end) {
    std::ofstream fout(change_ext(file_name), std::ios::binary);

    std::default_random_engine generator(16);

    std::vector<MSTensor> inputs;
    std::vector<MSTensor> encoder_outputs;
    std::vector<MSTensor> decoder_inputs;
    std::vector<MSTensor> outputs;

    std::cout << "Start predict input files:" << file_name << std::endl;
    std::vector<float> features;
    preprocess_encoder(file_name, features, encoderInputs, inputs);

    std::cout << "Model predict" << std::endl;
    gettimeofday(&start, NULL);
    encoder.Predict(inputs, &encoder_outputs);

    // Create decoder inputs
    std::shared_ptr<int> sptrFexc(new int[3], std::default_delete<int[]>());
    for (int i = 0; i < 3; i++) {
        sptrFexc.get()[i] = 128;
    }

    std::shared_ptr<float> sptrState1(new float[384], std::default_delete<float[]>());
    for (int i = 0; i < 384; i++) {
        sptrState1.get()[i] = 0;
    }
    MSTensor state1(decoderInputs[2].Name(),
                    decoderInputs[2].DataType(),
                    decoderInputs[2].Shape(),
                    (const void *)(sptrState1.get()),
                    384 * sizeof(float));

    std::shared_ptr<float> sptrState2(new float[16], std::default_delete<float[]>());
    for (int i = 0; i < 16; i++) {
        sptrState2.get()[i] = 0;
    }
    MSTensor state2(decoderInputs[3].Name(),
                    decoderInputs[3].DataType(),
                    decoderInputs[3].Shape(),
                    (const void *)(sptrState2.get()),
                    16 * sizeof(float));

    std::shared_ptr<const void> sptrCfeat = encoder_outputs[0].Data();
    const char *ptrCfeat = (const char *)sptrCfeat.get();

    std::vector<float> pcm(encoder_outputs[0].Shape()[1] * 160);
    for (size_t i = 0; i < pcm.size(); i++) {
        pcm[i] = 0.;
    }

    float mem = 0.;
    int16_t round_mem;
    float coef = 0.85;
    int skip = 17;
    size_t real_len = features.size() / 36;
    int end_fr = real_len > MAXLEN ? MAXLEN : real_len;
    for (int fr = 0; fr < end_fr; fr++) {
        const void *timestep_cfeat = (const void *)(ptrCfeat + fr * 128 * 2);
        for (int i = skip; i < 160; i++) {
            float pred = 0.;
            for (int j = 0; j < 16; j++) {
                pred -= features[fr * 36 + 20 + j] * pcm[fr * 160 + i - j - 1];
            }

            sptrFexc.get()[1] = lin2ulaw(pred);

            decoder_inputs.emplace_back(decoderInputs[0].Name(),
                                        decoderInputs[0].DataType(),
                                        decoderInputs[0].Shape(),
                                        (const void *)(sptrFexc.get()),
                                        3 * sizeof(int));
            decoder_inputs.emplace_back(decoderInputs[1].Name(),
                                        decoderInputs[1].DataType(),
                                        decoderInputs[1].Shape(),
                                        timestep_cfeat,
                                        128 * 2);
            decoder_inputs.emplace_back(state1);
            decoder_inputs.emplace_back(state2);
            decoder.Predict(decoder_inputs, &outputs);

            state1 = outputs[1];
            state2 = outputs[2];

            std::shared_ptr<const void> sptrProbs = outputs[0].Data();
            const float *ptrProbs = (const float *)sptrProbs.get();

            sptrFexc.get()[2] = sample_multinomial(ptrProbs, generator, 256, features[fr * 36 + 19]);

            pcm[fr * 160 + i] = pred + ulaw2lin(sptrFexc.get()[2]);
            sptrFexc.get()[0] = lin2ulaw(pcm[fr * 160 + i]);

            mem = coef*mem + pcm[fr * 160 + i];

            round_mem = (int16_t)roundf(mem);
            fout.write(reinterpret_cast<const char *>(&round_mem), sizeof(round_mem));

            decoder_inputs.clear();
        }
        skip = 0;
    }
    gettimeofday(&end, NULL);
    fout.close();
}

int main(int argc, char **argv) {
    if (!ParseCommandLineFlags(argc, argv)) {
        std::cout << "Failed to parse args" << std::endl;
        return 1;
    }
    if (RealPath(FLAGS_encoder_path).empty() || RealPath(FLAGS_decoder_path).empty()) {
      std::cout << "Invalid model" << std::endl;
      return 1;
    }

    auto ascend310_info = std::make_shared<mindspore::Ascend310DeviceInfo>();
    ascend310_info->SetDeviceID(FLAGS_device_id);
    ascend310_info->SetPrecisionMode("preferred_fp32");
    ascend310_info->SetOpSelectImplMode("high_precision");

    Status ret;
    // Build encoder
    Model encoder;
    if (!LoadModel(FLAGS_encoder_path, FLAGS_device_type, FLAGS_device_id, ascend310_info, &encoder)) {
        std::cout << "Failed to load model " << FLAGS_encoder_path << ", device id: " << FLAGS_device_id
                  << ", device type: " << FLAGS_device_type;
        return 1;
    }
    std::cout << "Encoder built successfully" << std::endl;

    // Build decoder
    Model decoder;
    if (!LoadModel(FLAGS_decoder_path, FLAGS_device_type, FLAGS_device_id, ascend310_info, &decoder)) {
        std::cout << "Failed to load model " << FLAGS_decoder_path << ", device id: " << FLAGS_device_id
                  << ", device type: " << FLAGS_device_type;
        return 1;
    }
    std::cout << "Decoder built successfully" << std::endl;


    std::vector<MSTensor> encoderInputs = encoder.GetInputs();
    std::vector<MSTensor> decoderInputs = decoder.GetInputs();


    auto all_files = GetAllFiles(FLAGS_dataset_path);
    if (all_files.empty()) {
        std::cout << "ERROR: no input data." << std::endl;
        return 1;
    }

    std::map<double, double> costTime_map;
    size_t size = all_files.size();

    for (size_t file_n = 0; file_n < size; ++file_n) {
        struct timeval start;
        struct timeval end;
        double startTime_ms;
        double endTime_ms;

        predict_file(all_files[file_n], encoder, encoderInputs, decoder, decoderInputs, start, end);

        startTime_ms = (1.0 * start.tv_sec * 1000000 + start.tv_usec) / 1000;
        endTime_ms = (1.0 * end.tv_sec * 1000000 + end.tv_usec) / 1000;
        costTime_map.insert(std::pair<double, double>(startTime_ms, endTime_ms));
    }
    double average = 0.0;
    int infer_cnt = 0;

    for (auto iter = costTime_map.begin(); iter != costTime_map.end(); iter++) {
        double diff = 0.0;
        diff = iter->second - iter->first;
        average += diff;
        infer_cnt++;
    }

    average = average / infer_cnt;

    std::stringstream timeCost;
    timeCost << "NN inference cost average time: "<< average << " ms of infer_count " << infer_cnt << std::endl;
    std::cout << "NN inference cost average time: "<< average << "ms of infer_count " << infer_cnt << std::endl;
    std::string file_name = "./time_Result" + std::string("/test_perform_static.txt");
    std::ofstream file_stream(file_name.c_str(), std::ios::trunc);
    file_stream << timeCost.str();
    file_stream.close();
    costTime_map.clear();
    return 0;
}
