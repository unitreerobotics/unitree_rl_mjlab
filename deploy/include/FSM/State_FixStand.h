// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <chrono>
#include "FSMState.h"
#include "LinearInterpolator.h"

class State_FixStand : public FSMState
{
public:
    State_FixStand(int state, std::string state_string = "FixStand")
    : FSMState(state, state_string)
    {
        auto cfg = param::config["FSM"][state_string];
        ts_ = cfg["ts"].as<std::vector<float>>();
        qs_ = cfg["qs"].as<std::vector<std::vector<float>>>();
        kp_ = cfg["kp"].as<std::vector<float>>();
        kd_ = cfg["kd"].as<std::vector<float>>();
        assert(ts_.size() == qs_.size());
    }

    void enter() override
    {
        std::vector<float> q0(kp_.size());
        {
            std::lock_guard<std::mutex> state_lock(lowstate->mutex_);
            for (size_t i = 0; i < kp_.size(); ++i)
            {
                q0[i] = lowstate->msg_.motor_state()[i].q();
            }
        }
        qs_[0] = q0;

        lowcmd->lock();
        for (size_t i = 0; i < kp_.size(); ++i)
        {
            auto& motor = lowcmd->msg_.motor_cmd()[i];
            motor.q() = q0[i];
            motor.kp() = kp_[i];
            motor.kd() = kd_[i];
            motor.dq() = 0.0f;
            motor.tau() = 0.0f;
        }
        lowcmd->unlock();

        t0_ = std::chrono::steady_clock::now();
    }

    void run() override
    {
        const float t = std::chrono::duration<float>(std::chrono::steady_clock::now() - t0_).count();
        const auto q = linear_interpolate(t, ts_, qs_);
        for (size_t i = 0; i < q.size(); ++i)
        {
            lowcmd->msg_.motor_cmd()[i].q() = q[i];
        }
    }

private:
    std::chrono::steady_clock::time_point t0_;
    std::vector<float> ts_;
    std::vector<std::vector<float>> qs_;
    std::vector<float> kp_;
    std::vector<float> kd_;
};

REGISTER_FSM(State_FixStand)
