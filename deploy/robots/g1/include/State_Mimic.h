#pragma once

#include <atomic>
#include "FSM/State_RLBase.h"
#include <cnpy.h>


class State_Mimic : public FSMState
{
public:
    State_Mimic(int state_mode, std::string state_string);

    void enter();
    void run();
    void exit()
    {
        policy_thread_running = false;
        if (policy_thread.joinable()) {
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
    MotionLoader_(std::string motion_file)
    : dt(1.0f / 50.0f)
    {
        load_data_from_npz(motion_file);
        num_frames = dof_positions.size();
        duration = num_frames * dt;

        update(0.0f);
    }

    void load_data_from_npz(const std::string& motion_file)
    {
        cnpy::npz_t npz_data = cnpy::npz_load(motion_file);

        auto body_pos_w  = npz_data["body_pos_w"];   // [frame, body_id, 3]
        auto body_quat_w = npz_data["body_quat_w"];  // [frame, body_id, 4]
        auto joint_pos   = npz_data["joint_pos"];    // [frame, dof]
        auto joint_vel   = npz_data["joint_vel"];    // [frame, dof]

        root_positions.clear();
        root_quaternions.clear();
        dof_positions.clear();
        dof_velocities.clear();

        const size_t num_frames_npz = body_pos_w.shape[0];

        for (size_t i = 0; i < num_frames_npz; i++)
        {
            const size_t body_stride_pos  = body_pos_w.shape[1] * body_pos_w.shape[2];
            const size_t body_stride_quat = body_quat_w.shape[1] * body_quat_w.shape[2];

            Eigen::Vector3f root_pos = Eigen::Vector3f::Map(body_pos_w.data<float>() + i * body_stride_pos);
            root_positions.push_back(root_pos);

            Eigen::Quaternionf quat(
                body_quat_w.data<float>()[i * body_stride_quat + 0], // w
                body_quat_w.data<float>()[i * body_stride_quat + 1], // x
                body_quat_w.data<float>()[i * body_stride_quat + 2], // y
                body_quat_w.data<float>()[i * body_stride_quat + 3]  // z
            );
            root_quaternions.push_back(quat);

            Eigen::VectorXf joint_position(joint_pos.shape[1]);
            for (int j = 0; j < joint_pos.shape[1]; j++) {
                joint_position[j] = joint_pos.data<float>()[i * joint_pos.shape[1] + j];
            }

            Eigen::VectorXf joint_velocity(joint_vel.shape[1]);
            for (int j = 0; j < joint_vel.shape[1]; j++) {
                joint_velocity[j] = joint_vel.data<float>()[i * joint_vel.shape[1] + j];
            }

            dof_positions.push_back(joint_position);
            dof_velocities.push_back(joint_velocity);
        }
    }

    void update(float time)
    {
        float phase = std::clamp(time, 0.0f, duration);
        float f = phase / dt;
        frame = static_cast<int>(std::floor(f));
        frame = std::min(frame, num_frames - 1);
    }

    void reset(const isaaclab::ArticulationData & data, float t = 0.0f)
    {
        update(t);
        auto init_to_anchor = isaaclab::yawQuaternion(this->root_quaternion()).toRotationMatrix();
        auto world_to_anchor = isaaclab::yawQuaternion(data.root_quat_w).toRotationMatrix();
        world_to_init_ = world_to_anchor * init_to_anchor.transpose();
    }

    Eigen::VectorXf root_position() {
        return root_positions[frame];
    }
    Eigen::Quaternionf root_quaternion() {
        return root_quaternions[frame];
    }
    Eigen::VectorXf joint_pos() {
        return dof_positions[frame];
    }
    Eigen::VectorXf joint_vel() {
        return dof_velocities[frame];
    }

    float dt;
    int num_frames;
    float duration;

    int frame;
    std::vector<Eigen::VectorXf> root_positions;
    std::vector<Eigen::Quaternionf> root_quaternions;
    std::vector<Eigen::VectorXf> dof_positions;
    std::vector<Eigen::VectorXf> dof_velocities;
    Eigen::Matrix3f world_to_init_;
};


REGISTER_FSM(State_Mimic)
