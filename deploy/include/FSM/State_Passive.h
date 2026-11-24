// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"

class State_Passive : public FSMState
{
public:
    State_Passive(int state, std::string state_string = "Passive")
    : FSMState(state, state_string)
    {
        auto cfg = param::config["FSM"][state_string];
        kd_ = cfg["kd"].as<std::vector<float>>();
        auto motor_mode = cfg["mode"];
        if (motor_mode.IsDefined())
        {
            auto values = motor_mode.as<std::vector<int>>();
            lowcmd->lock();
            for (size_t i = 0; i < values.size(); ++i)
            {
                lowcmd->msg_.motor_cmd()[i].mode() = values[i];
            }
            lowcmd->unlock();
        }
    }

    void enter() override
    {
        lowcmd->lock();
        for (size_t i = 0; i < kd_.size(); ++i)
        {
            auto& motor = lowcmd->msg_.motor_cmd()[i];
            motor.kp() = 0.0f;
            motor.kd() = kd_[i];
            motor.dq() = 0.0f;
            motor.tau() = 0.0f;
        }
        lowcmd->unlock();
    }

    void run() override
    {
        std::lock_guard<std::mutex> lock(lowstate->mutex_);
        for (size_t i = 0; i < lowcmd->msg_.motor_cmd().size(); ++i)
        {
            lowcmd->msg_.motor_cmd()[i].q() = lowstate->msg_.motor_state()[i].q();
        }
    }

private:
    std::vector<float> kd_;
};

REGISTER_FSM(State_Passive)
