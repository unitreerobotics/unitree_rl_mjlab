#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include <algorithm>

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    if (env->cfg["hybrid_control"])
    {
        use_hybrid_control_ = true;
        const auto hybrid_cfg = env->cfg["hybrid_control"];
        leg_joint_ids_ = hybrid_cfg["leg_joint_ids"].as<std::vector<int>>();
        wheel_joint_ids_ = hybrid_cfg["wheel_joint_ids"].as<std::vector<int>>();

        wheel_command_mode_ = hybrid_cfg["wheel_command_mode"].as<std::string>("velocity");
        wheel_kd_ = hybrid_cfg["wheel_kd"].as<float>(2.0f);
        wheel_vel_limit_ = hybrid_cfg["wheel_vel_limit"].as<float>(40.0f);
        wheel_tau_limit_ = hybrid_cfg["wheel_tau_limit"].as<float>(12.0f);
        wheel_tau_ff_ = hybrid_cfg["wheel_tau_ff"].as<float>(0.0f);

        spdlog::info(
            "Hybrid wheel-leg control enabled: legs={}, wheels={}, wheel_mode={}",
            leg_joint_ids_.size(),
            wheel_joint_ids_.size(),
            wheel_command_mode_
        );
    }

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_RLBase::run()
{
    const auto action = env->action_manager->processed_actions();

    if (!use_hybrid_control_)
    {
        const size_t n = std::min(action.size(), env->robot->data.joint_ids_map.size());
        for(size_t i = 0; i < n; ++i) {
            lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
        }
        return;
    }

    const size_t expected_dim = leg_joint_ids_.size() + wheel_joint_ids_.size();
    if (action.size() != expected_dim)
    {
        spdlog::error(
            "Hybrid action dimension mismatch: got {}, expected {}",
            action.size(),
            expected_dim
        );
        return;
    }

    size_t cursor = 0;

    for (const int joint_id : leg_joint_ids_)
    {
        if (joint_id < 0 || joint_id >= static_cast<int>(env->robot->data.joint_ids_map.size()))
        {
            spdlog::error("Invalid leg joint_id {} in hybrid_control.leg_joint_ids", joint_id);
            return;
        }

        const int motor_id = env->robot->data.joint_ids_map[joint_id];
        auto & motor = lowcmd->msg_.motor_cmd()[motor_id];
        motor.q() = action[cursor++];
        motor.dq() = 0.0f;
        motor.tau() = 0.0f;
    }

    for (const int joint_id : wheel_joint_ids_)
    {
        if (joint_id < 0 || joint_id >= static_cast<int>(env->robot->data.joint_ids_map.size()))
        {
            spdlog::error("Invalid wheel joint_id {} in hybrid_control.wheel_joint_ids", joint_id);
            return;
        }

        const int motor_id = env->robot->data.joint_ids_map[joint_id];
        auto & motor = lowcmd->msg_.motor_cmd()[motor_id];

        motor.q() = lowstate->msg_.motor_state()[motor_id].q();
        motor.kp() = 0.0f;

        if (wheel_command_mode_ == "torque")
        {
            const float target_tau = std::clamp(action[cursor++], -wheel_tau_limit_, wheel_tau_limit_);
            motor.kd() = 0.0f;
            motor.dq() = 0.0f;
            motor.tau() = target_tau;
        }
        else
        {
            const float target_dq = std::clamp(action[cursor++], -wheel_vel_limit_, wheel_vel_limit_);
            const float sign = (target_dq > 0.0f) - (target_dq < 0.0f);
            motor.kd() = wheel_kd_;
            motor.dq() = target_dq;
            motor.tau() = std::clamp(wheel_tau_ff_ * sign, -wheel_tau_limit_, wheel_tau_limit_);
        }
    }
}
