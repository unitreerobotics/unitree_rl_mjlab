#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include "FSM/State_RLBase.h"
#include <cnpy.h>


class State_Mimic : public FSMState
{
public:
    State_Mimic(int state_mode, std::string state_string);

    ~State_Mimic() override
    {
        exit();
    }

    void enter() override;
    void run() override;
    void exit() override
    {
        policy_thread_running = false;
        if (policy_thread.joinable())
        {
            policy_thread.join();
        }
    }

    class MotionLoader_;

    static std::shared_ptr<MotionLoader_> motion; // for obs computation

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;
    std::shared_ptr<MotionLoader_> motion_; // for saving

    std::thread policy_thread;
    std::atomic<bool> policy_thread_running{false};
    std::array<float, 2> time_range_;
};


class State_Mimic::MotionLoader_
{
public:
    explicit MotionLoader_(const std::string& motion_file)
    : dt(1.0f / 50.0f)
    {
        load_data_from_npz(motion_file);
        num_frames = static_cast<int>(dof_positions.size());
        duration = num_frames * dt;
        update(0.0f);
    }

    void load_data_from_npz(const std::string& motion_file)
    {
        cnpy::npz_t npz_data = cnpy::npz_load(motion_file);

        const auto& body_pos_w  = npz_data.at("body_pos_w");   // [frame, body_id, 3]
        const auto& body_quat_w = npz_data.at("body_quat_w");  // [frame, body_id, 4]
        const auto& joint_pos   = npz_data.at("joint_pos");    // [frame, dof]
        const auto& joint_vel   = npz_data.at("joint_vel");    // [frame, dof]

        auto fps_it = npz_data.find("fps");
        if (fps_it != npz_data.end())
        {
            const auto& fps_arr = fps_it->second;
            const float fps = (fps_arr.word_size == 8)
                ? static_cast<float>(*fps_arr.data<double>())
                : *fps_arr.data<float>();
            dt = 1.0f / fps;
        }

        const size_t num_frames_npz   = body_pos_w.shape[0];
        const size_t body_stride_pos  = body_pos_w.shape[1] * body_pos_w.shape[2];
        const size_t body_stride_quat = body_quat_w.shape[1] * body_quat_w.shape[2];
        const size_t num_dofs         = joint_pos.shape[1];

        root_positions.clear();
        root_quaternions.clear();
        dof_positions.clear();
        dof_velocities.clear();
        root_positions.reserve(num_frames_npz);
        root_quaternions.reserve(num_frames_npz);
        dof_positions.reserve(num_frames_npz);
        dof_velocities.reserve(num_frames_npz);

        const float* pos_ptr  = body_pos_w.data<float>();
        const float* quat_ptr = body_quat_w.data<float>();
        const float* jpos_ptr = joint_pos.data<float>();
        const float* jvel_ptr = joint_vel.data<float>();

        for (size_t i = 0; i < num_frames_npz; ++i)
        {
            root_positions.emplace_back(Eigen::Vector3f::Map(pos_ptr + i * body_stride_pos));
            const float* q = quat_ptr + i * body_stride_quat;
            root_quaternions.emplace_back(q[0], q[1], q[2], q[3]); // w, x, y, z
            dof_positions.emplace_back(Eigen::VectorXf::Map(jpos_ptr + i * num_dofs, num_dofs));
            dof_velocities.emplace_back(Eigen::VectorXf::Map(jvel_ptr + i * num_dofs, num_dofs));
        }
    }

    void update(float time)
    {
        const float phase = std::clamp(time, 0.0f, duration);
        frame = std::clamp(static_cast<int>(std::floor(phase / dt)), 0, num_frames - 1);
    }

    void reset(const isaaclab::ArticulationData& data, float t = 0.0f)
    {
        update(t);
        auto init_to_anchor = isaaclab::yawQuaternion(root_quaternion()).toRotationMatrix();
        auto world_to_anchor = isaaclab::yawQuaternion(data.root_quat_w).toRotationMatrix();
        world_to_init_ = world_to_anchor * init_to_anchor.transpose();
    }

    Eigen::VectorXf root_position() const { return root_positions[frame]; }
    Eigen::Quaternionf root_quaternion() const { return root_quaternions[frame]; }
    Eigen::VectorXf joint_pos() const { return dof_positions[frame]; }
    Eigen::VectorXf joint_vel() const { return dof_velocities[frame]; }

    float dt;
    int num_frames = 0;
    float duration = 0.0f;

    int frame = 0;
    std::vector<Eigen::VectorXf> root_positions;
    std::vector<Eigen::Quaternionf> root_quaternions;
    std::vector<Eigen::VectorXf> dof_positions;
    std::vector<Eigen::VectorXf> dof_velocities;
    Eigen::Matrix3f world_to_init_;
};


REGISTER_FSM(State_Mimic)
