// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "onnxruntime_cxx_api.h"
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace isaaclab
{

class Algorithms
{
public:
    virtual ~Algorithms() = default;

    virtual std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs) = 0;

    std::vector<float> get_action()
    {
        std::lock_guard<std::mutex> lock(act_mtx_);
        return action;
    }

    std::vector<float> action;
protected:
    std::mutex act_mtx_;
};

class OrtRunner : public Algorithms
{
public:
    OrtRunner(std::string model_path)
    {
        // Init Model
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "onnx_model");
        session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);

        session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);

        for (size_t i = 0; i < session->GetInputCount(); ++i)
        {
            auto shape = session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            if (shape[0] <= 0)
            {
                shape[0] = 1;
            }
            size_t size = 1;
            for (const auto dim : shape)
            {
                size *= static_cast<size_t>(dim);
            }
            input_shapes.push_back(std::move(shape));
            input_sizes.push_back(size);

            auto input_name = session->GetInputNameAllocated(i, allocator);
            input_names.push_back(input_name.get());
            input_name_ptrs.push_back(std::move(input_name));
        }

        output_shape = session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (output_shape[0] <= 0)
        {
            output_shape[0] = 1;
        }
        auto output_name = session->GetOutputNameAllocated(0, allocator);
        output_names.push_back(output_name.get());
        output_name_ptrs.push_back(std::move(output_name));

        action.resize(static_cast<size_t>(output_shape[1]));
    }

    std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs) override
    {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        std::vector<Ort::Value> input_tensors;
        input_tensors.reserve(input_names.size());
        for (size_t i = 0; i < input_names.size(); ++i)
        {
            auto& input_data = obs.at(input_names[i]);
            input_tensors.push_back(Ort::Value::CreateTensor<float>(
                memory_info,
                input_data.data(),
                input_sizes[i],
                input_shapes[i].data(),
                input_shapes[i].size()));
        }

        auto output_tensor = session->Run(
            Ort::RunOptions{nullptr},
            input_names.data(),
            input_tensors.data(),
            input_tensors.size(),
            output_names.data(),
            1);

        auto* floatarr = output_tensor.front().GetTensorMutableData<float>();
        std::lock_guard<std::mutex> lock(act_mtx_);
        std::memcpy(action.data(), floatarr, action.size() * sizeof(float));
        return action;
    }

private:
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<Ort::AllocatedStringPtr> input_name_ptrs;
    std::vector<Ort::AllocatedStringPtr> output_name_ptrs;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;

    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<size_t> input_sizes;
    std::vector<int64_t> output_shape;
};
};
