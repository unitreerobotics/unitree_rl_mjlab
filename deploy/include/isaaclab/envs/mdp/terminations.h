#pragma once

#include <algorithm>
#include <cmath>

#include "isaaclab/envs/manager_based_rl_env.h"

namespace isaaclab
{
namespace mdp
{

inline bool bad_orientation(ManagerBasedRLEnv* env, float limit_angle = 1.0)
{
    const auto & gravity_b = env->robot->data.projected_gravity_b;
    // Tilt angle between the base z-axis and world up; clamp to keep acos defined.
    const float cos_tilt = std::clamp(-gravity_b[2], -1.0f, 1.0f);
    return std::acos(cos_tilt) > limit_angle;
}

}
}
