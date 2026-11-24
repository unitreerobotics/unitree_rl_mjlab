#include "State_Mimic.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

static Eigen::Quaternionf init_quat;
std::shared_ptr<State_Mimic::MotionLoader_> State_Mimic::motion = nullptr;


Eigen::Quaternionf robot_quat_w(isaaclab::ManagerBasedRLEnv* env)
{
    using G1Type = unitree::BaseArticulation<LowState_t::SharedPtr>;
    auto* robot = dynamic_cast<G1Type*>(env->robot.get());

    const auto root_quat = env->robot->data.root_quat_w;
    float waist_yaw = 0.0f;
    {
        std::lock_guard<std::mutex> lock(robot->lowstate->mutex_);
        waist_yaw = robot->lowstate->msg_.motor_state()[12].q();
    }

    return root_quat * Eigen::AngleAxisf(waist_yaw, Eigen::Vector3f::UnitZ());
}

Eigen::Quaternionf motion_anchor_quat_w(const std::shared_ptr<State_Mimic::MotionLoader_>& loader)
{
    const auto root_quat = loader->root_quaternion();
    const auto joint_pos = loader->joint_pos();
    return root_quat * Eigen::AngleAxisf(joint_pos[12], Eigen::Vector3f::UnitZ());
}


namespace isaaclab
{
namespace mdp
{

REGISTER_OBSERVATION(motion_command)
{
    auto loader = State_Mimic::motion;
    const auto motion_joint_pos = loader->joint_pos();
    const auto motion_joint_vel = loader->joint_vel();

    std::vector<float> data;
    data.reserve(motion_joint_pos.size() + motion_joint_vel.size());
    data.insert(data.end(), motion_joint_pos.data(), motion_joint_pos.data() + motion_joint_pos.size());
    data.insert(data.end(), motion_joint_vel.data(), motion_joint_vel.data() + motion_joint_vel.size());
    return data;
}

REGISTER_OBSERVATION(motion_anchor_ori_b)
{
    auto loader = State_Mimic::motion;
    const auto real_quat_w = robot_quat_w(env);
    const auto ref_quat_w = motion_anchor_quat_w(loader);

    const auto rel_quat = (init_quat * ref_quat_w).conjugate() * real_quat_w;
    const Eigen::Matrix3f rot = rel_quat.toRotationMatrix().transpose();

    Eigen::Matrix<float, 6, 1> data;
    data << rot(0, 0), rot(0, 1), rot(1, 0), rot(1, 1), rot(2, 0), rot(2, 1);
    return std::vector<float>(data.data(), data.data() + data.size());
}

}
}


State_Mimic::State_Mimic(int state_mode, std::string state_string)
: FSMState(state_mode, state_string)
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());
    auto articulation = std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate);

    std::filesystem::path motion_file = cfg["motion_file"].as<std::string>();
    if (!motion_file.is_absolute())
    {
        motion_file = param::proj_dir / motion_file;
    }

    motion_ = std::make_shared<MotionLoader_>(motion_file.string());
    spdlog::info(
        "Loaded motion file '{}' with duration {:.2f}s (fps={:.1f})",
        motion_file.stem().string(),
        motion_->duration,
        1.0f / motion_->dt);
    motion = motion_;

    time_range_[0] = cfg["time_start"]
        ? std::clamp(cfg["time_start"].as<float>(), 0.0f, motion_->duration)
        : 0.0f;
    time_range_[1] = cfg["time_end"]
        ? std::clamp(cfg["time_end"].as<float>(), 0.0f, motion_->duration)
        : motion_->duration;
    const std::string end_state = cfg["end_state"]
        ? cfg["end_state"].as<std::string>()
        : "Velocity";

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        articulation);
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    registered_checks.emplace_back(
        [this]() -> bool { return (env->episode_length.load() * env->step_dt + time_range_[0]) >= time_range_[1]; },
        FSMStringMap.right.at(end_state));
    registered_checks.emplace_back(
        [this]() -> bool { return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
        FSMStringMap.right.at("Passive"));
}

void State_Mimic::enter()
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

    motion = motion_;
    env->robot->update();
    motion->reset(env->robot->data, time_range_[0]);
    const auto ref_yaw = isaaclab::yawQuaternion(motion_anchor_quat_w(motion)).toRotationMatrix();
    const auto robot_yaw = isaaclab::yawQuaternion(robot_quat_w(env.get())).toRotationMatrix();
    init_quat = robot_yaw * ref_yaw.transpose();
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
            motion->update(env->episode_length.load() * env->step_dt + time_range_[0]);
            env->step();

            std::this_thread::sleep_until(sleep_till);
            sleep_till += dt;
        }
    });
}

void State_Mimic::run()
{
    const auto action = env->action_manager->processed_actions();
    for (size_t i = 0; i < env->robot->data.joint_ids_map.size(); ++i)
    {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}
