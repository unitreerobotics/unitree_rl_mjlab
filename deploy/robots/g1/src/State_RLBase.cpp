#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include <unordered_map>

namespace isaaclab
{
// keyboard velocity commands example
// change "velocity_commands" observation name in policy deploy.yaml to "keyboard_velocity_commands"
REGISTER_OBSERVATION(keyboard_velocity_commands)
{
    static const std::unordered_map<std::string, std::array<float, 3>> key_commands = {
        {"w", { 1.0f,  0.0f,  0.0f}},
        {"s", {-1.0f,  0.0f,  0.0f}},
        {"a", { 0.0f,  1.0f,  0.0f}},
        {"d", { 0.0f, -1.0f,  0.0f}},
        {"q", { 0.0f,  0.0f,  1.0f}},
        {"e", { 0.0f,  0.0f, -1.0f}},
    };

    const auto it = key_commands.find(FSMState::keyboard->key());
    if (it == key_commands.end())
    {
        return {0.0f, 0.0f, 0.0f};
    }

    const auto ranges = env->cfg["commands"]["base_velocity"]["ranges"];
    return {
        mdp::scale_velocity_axis(it->second[0], ranges["lin_vel_x"]),
        mdp::scale_velocity_axis(it->second[1], ranges["lin_vel_y"]),
        mdp::scale_velocity_axis(it->second[2], ranges["ang_vel_z"]),
    };
}

}

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

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_RLBase::run()
{
    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}
