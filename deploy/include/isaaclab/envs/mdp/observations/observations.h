// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "isaaclab/envs/manager_based_rl_env.h"

namespace isaaclab
{
namespace mdp
{

REGISTER_OBSERVATION(base_ang_vel)
{
    auto & asset = env->robot;
    auto & data = asset->data.root_ang_vel_b;
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(projected_gravity)
{
    auto & asset = env->robot;
    auto & data = asset->data.projected_gravity_b;
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(joint_pos)
{
    auto & asset = env->robot;
    std::vector<float> data;

    std::vector<int> joint_ids;
    try {
        joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();
    } catch(const std::exception& e) {
    }

    if(joint_ids.empty())
    {
        data.resize(asset->data.joint_pos.size());
        for(size_t i = 0; i < asset->data.joint_pos.size(); ++i)
        {
            data[i] = asset->data.joint_pos[i];
        }
    }
    else
    {
        data.resize(joint_ids.size());
        for(size_t i = 0; i < joint_ids.size(); ++i)
        {
            data[i] = asset->data.joint_pos[joint_ids[i]];
        }
    }

    return data;
}

REGISTER_OBSERVATION(joint_pos_rel)
{
    auto & asset = env->robot;
    std::vector<float> data;

    data.resize(asset->data.joint_pos.size());
    for(size_t i = 0; i < asset->data.joint_pos.size(); ++i) {
        data[i] = asset->data.joint_pos[i] - asset->data.default_joint_pos[i];
    }

    try {
        std::vector<int> joint_ids;
        joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();
        if(!joint_ids.empty()) {
            std::vector<float> tmp_data;
            tmp_data.resize(joint_ids.size());
            for(size_t i = 0; i < joint_ids.size(); ++i){
                tmp_data[i] = data[joint_ids[i]];
            }
            data = tmp_data;
        }
    } catch(const std::exception& e) {

    }

    return data;
}

REGISTER_OBSERVATION(joint_vel_rel)
{
    auto & asset = env->robot;
    auto data = asset->data.joint_vel;

    try {
        const std::vector<int> joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();

        if(!joint_ids.empty()) {
            data.resize(joint_ids.size());
            for(size_t i = 0; i < joint_ids.size(); ++i) {
                data[i] = asset->data.joint_vel[joint_ids[i]];
            }
        }
    } catch(const std::exception& e) {
    }
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(last_action)
{
    auto data = env->action_manager->action();
    return std::vector<float>(data.data(), data.data() + data.size());
};

inline float scale_velocity_axis(float value, const YAML::Node& range)
{
    const float clamped = std::clamp(value, -1.0f, 1.0f);
    return clamped >= 0.0f
        ? clamped * range[1].as<float>()
        : -clamped * range[0].as<float>();
}

inline bool uses_keyboard_velocity(const YAML::Node& obs_cfg)
{
    if (obs_cfg["keyboard_velocity_commands"])
    {
        return true;
    }
    for (auto it = obs_cfg.begin(); it != obs_cfg.end(); ++it)
    {
        if (it->second.IsMap() && it->second["keyboard_velocity_commands"])
        {
            return true;
        }
    }
    return false;
}

REGISTER_OBSERVATION(velocity_commands)
{
    auto& joystick = env->robot->data.joystick;
    const auto ranges = env->cfg["commands"]["base_velocity"]["ranges"];
    std::vector<float> obs = {
        scale_velocity_axis(joystick->ly(), ranges["lin_vel_x"]),
        scale_velocity_axis(-joystick->lx(), ranges["lin_vel_y"]),
        scale_velocity_axis(-joystick->rx(), ranges["ang_vel_z"]),
    };

    if (std::hypot(obs[0], obs[1]) + std::abs(obs[2]) <= 0.1f)
    {
        return {0.0f, 0.0f, 0.0f};
    }
    return obs;
}

REGISTER_OBSERVATION(gait_phase)
{
    const float period = params["period"].as<float>();
    const float t = static_cast<float>(env->episode_length.load()) * env->step_dt;
    const float phase = std::fmod(t, period) / period;
    env->global_phase = phase;

    const auto cmd = uses_keyboard_velocity(env->cfg["observations"])
        ? observations_map().at("keyboard_velocity_commands")(env, params)
        : isaaclab::mdp::velocity_commands(env, params);

    if (std::hypot(cmd[0], cmd[1]) + std::abs(cmd[2]) <= 0.1f)
    {
        return {0.0f, 0.0f};
    }

    const float angle = phase * 2.0f * static_cast<float>(M_PI);
    return {std::sin(angle), std::cos(angle)};
}

}
}
