// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <atomic>

#include "FSMState.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"

class State_RLBase : public FSMState
{
public:
    State_RLBase(int state_mode, std::string state_string);

    ~State_RLBase() override
    {
        exit();
    }

    void enter() override
    {
        lowcmd->lock();
        for (size_t i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
        {
            const auto motor_id = env->robot->data.joint_ids_map[i];
            auto& motor_cmd = lowcmd->msg_.motor_cmd()[motor_id];
            motor_cmd.kp() = env->robot->data.joint_stiffness[i];
            motor_cmd.kd() = env->robot->data.joint_damping[i];
            motor_cmd.dq() = 0.0f;
            motor_cmd.tau() = 0.0f;
        }
        lowcmd->unlock();

        // Reset env synchronously before starting policy thread so run() never sees stale actions.
        env->reset();

        policy_thread_running = true;
        policy_thread = std::thread([this]()
        {
            using clock = std::chrono::steady_clock;
            const std::chrono::duration<double> desired_duration(env->step_dt);
            const auto dt = std::chrono::duration_cast<clock::duration>(desired_duration);

            auto sleep_till = clock::now() + dt;
            while (policy_thread_running)
            {
                env->step();
                std::this_thread::sleep_until(sleep_till);
                sleep_till += dt;
            }
        });
    }

    void run() override;

    void exit() override
    {
        policy_thread_running = false;
        if (policy_thread.joinable())
        {
            policy_thread.join();
        }
    }

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;

    std::thread policy_thread;
    std::atomic<bool> policy_thread_running{false};
};

REGISTER_FSM(State_RLBase)
