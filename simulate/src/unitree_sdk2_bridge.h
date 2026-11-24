#pragma once

#include <mujoco/mujoco.h>

#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/dds_wrapper/robots/go2/go2.h>
#include <unitree/dds_wrapper/robots/g1/g1.h>
#include <unitree/idl/hg/BmsState_.hpp>
#include <unitree/idl/hg/IMUState_.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>

#include "param.h"
#include "physics_joystick.h"

#define MOTOR_SENSOR_NUM 3

class UnitreeSDK2BridgeBase
{
public:
    UnitreeSDK2BridgeBase(mjModel* model, mjData* data, std::recursive_mutex* sim_mtx = nullptr)
    : mj_model_(model), mj_data_(data), sim_mtx_(sim_mtx)
    {
        _check_sensor();
        if (param::config.print_scene_information == 1) {
            printSceneInformation();
        }
        initJoystick();
    }

    virtual ~UnitreeSDK2BridgeBase() = default;

    virtual void start() {}
    virtual void stop() {}

    virtual void setModelData(mjModel* model, mjData* data)
    {
        auto lock = lockSim();
        mj_model_ = model;
        mj_data_ = data;
        resetSensorIndices();
        _check_sensor();
    }

    void printSceneInformation()
    {
        auto printObjects = [this](const char* title, int count, int type, auto getIndex) {
            std::cout << "<<------------- " << title << " ------------->> " << std::endl;
            for (int i = 0; i < count; ++i) {
                const char* name = mj_id2name(mj_model_, type, i);
                if (name) {
                    std::cout << title << "_index: " << getIndex(i) << ", name: " << name;
                    if (type == mjOBJ_SENSOR) {
                        std::cout << ", dim: " << mj_model_->sensor_dim[i];
                    }
                    std::cout << std::endl;
                }
            }
            std::cout << std::endl;
        };

        printObjects("Link", mj_model_->nbody, mjOBJ_BODY, [](int i) { return i; });
        printObjects("Joint", mj_model_->njnt, mjOBJ_JOINT, [](int i) { return i; });
        printObjects("Actuator", mj_model_->nu, mjOBJ_ACTUATOR, [](int i) { return i; });

        int sensorIndex = 0;
        printObjects("Sensor", mj_model_->nsensor, mjOBJ_SENSOR, [&](int i) {
            int currentIndex = sensorIndex;
            sensorIndex += mj_model_->sensor_dim[i];
            return currentIndex;
        });
    }

protected:
    std::unique_lock<std::recursive_mutex> lockSim() const
    {
        return sim_mtx_
            ? std::unique_lock<std::recursive_mutex>(*sim_mtx_)
            : std::unique_lock<std::recursive_mutex>();
    }

    template <typename ImuMsg>
    void writeImuState(ImuMsg& imu, int quat_adr, int gyro_adr, int acc_adr) const
    {
        if (quat_adr >= 0) {
            const double w = mj_data_->sensordata[quat_adr + 0];
            const double x = mj_data_->sensordata[quat_adr + 1];
            const double y = mj_data_->sensordata[quat_adr + 2];
            const double z = mj_data_->sensordata[quat_adr + 3];

            imu.quaternion()[0] = w;
            imu.quaternion()[1] = x;
            imu.quaternion()[2] = y;
            imu.quaternion()[3] = z;

            imu.rpy()[0] = std::atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y));
            imu.rpy()[1] = std::asin(std::clamp(2.0 * (w * y - z * x), -1.0, 1.0));
            imu.rpy()[2] = std::atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
        }

        if (gyro_adr >= 0) {
            imu.gyroscope()[0] = mj_data_->sensordata[gyro_adr + 0];
            imu.gyroscope()[1] = mj_data_->sensordata[gyro_adr + 1];
            imu.gyroscope()[2] = mj_data_->sensordata[gyro_adr + 2];
        }

        if (acc_adr >= 0) {
            imu.accelerometer()[0] = mj_data_->sensordata[acc_adr + 0];
            imu.accelerometer()[1] = mj_data_->sensordata[acc_adr + 1];
            imu.accelerometer()[2] = mj_data_->sensordata[acc_adr + 2];
        }
    }

    void resetSensorIndices()
    {
        num_motor_ = 0;
        dim_motor_sensor_ = 0;
        imu_quat_adr_ = -1;
        imu_gyro_adr_ = -1;
        imu_acc_adr_ = -1;
        frame_pos_adr_ = -1;
        frame_vel_adr_ = -1;
        secondary_imu_quat_adr_ = -1;
        secondary_imu_gyro_adr_ = -1;
        secondary_imu_acc_adr_ = -1;
    }

    void initJoystick()
    {
        if (param::config.use_joystick != 1) {
            return;
        }
        const auto& type = param::config.joystick_type;
        const auto& dev  = param::config.joystick_device;
        const int bits   = param::config.joystick_bits;

        if (type == "xbox") {
            joystick = std::make_shared<XBoxJoystick>(dev, bits);
        } else if (type == "switch") {
            joystick = std::make_shared<SwitchJoystick>(dev, bits);
        } else {
            std::cerr << "Error: Unsupported joystick type '" << type << "'." << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    void _check_sensor()
    {
        num_motor_ = mj_model_->nu;
        dim_motor_sensor_ = MOTOR_SENSOR_NUM * num_motor_;

        auto findAdr = [this](const char* name) -> int {
            const int id = mj_name2id(mj_model_, mjOBJ_SENSOR, name);
            return id >= 0 ? mj_model_->sensor_adr[id] : -1;
        };

        imu_quat_adr_           = findAdr("imu_quat");
        imu_gyro_adr_           = findAdr("imu_gyro");
        imu_acc_adr_            = findAdr("imu_acc");
        frame_pos_adr_          = findAdr("frame_pos");
        frame_vel_adr_          = findAdr("frame_vel");
        secondary_imu_quat_adr_ = findAdr("secondary_imu_quat");
        secondary_imu_gyro_adr_ = findAdr("secondary_imu_gyro");
        secondary_imu_acc_adr_  = findAdr("secondary_imu_acc");
    }

    int num_motor_ = 0;
    int dim_motor_sensor_ = 0;

    mjModel* mj_model_ = nullptr;
    mjData* mj_data_ = nullptr;
    std::recursive_mutex* sim_mtx_ = nullptr;

    // Sensor data indices
    int imu_quat_adr_ = -1;
    int imu_gyro_adr_ = -1;
    int imu_acc_adr_ = -1;
    int frame_pos_adr_ = -1;
    int frame_vel_adr_ = -1;

    int secondary_imu_quat_adr_ = -1;
    int secondary_imu_gyro_adr_ = -1;
    int secondary_imu_acc_adr_ = -1;

    std::shared_ptr<unitree::common::UnitreeJoystick> joystick = nullptr;
};

template <typename LowCmd_t, typename LowState_t>
class RobotBridge : public UnitreeSDK2BridgeBase
{
    using HighState_t = unitree::robot::go2::publisher::SportModeState;
    using WirelessController_t = unitree::robot::go2::publisher::WirelessController;

public:
    RobotBridge(mjModel* model, mjData* data, std::recursive_mutex* sim_mtx = nullptr)
    : UnitreeSDK2BridgeBase(model, data, sim_mtx)
    {
        lowcmd = std::make_shared<LowCmd_t>("rt/lowcmd");
        lowstate = std::make_unique<LowState_t>();
        highstate = std::make_unique<HighState_t>();
        wireless_controller = std::make_unique<WirelessController_t>();
    }

    ~RobotBridge() override
    {
        stop();
    }

    void start() override
    {
        thread_ = std::make_shared<unitree::common::RecurrentThread>(
            "unitree_bridge", UT_CPU_ID_NONE, 1000, [this]() { this->run(); });
    }

    void stop() override
    {
        thread_.reset();
    }

    virtual void run()
    {
        auto sim_lock = lockSim();

        if (joystick) {
            joystick->update();
        }

        // lowcmd
        {
            std::lock_guard<std::mutex> lock(lowcmd->mutex_);
            for (int i = 0; i < num_motor_; ++i) {
                const auto& m = lowcmd->msg_.motor_cmd()[i];
                mj_data_->ctrl[i] =
                    m.tau() +
                    m.kp() * (m.q() - mj_data_->sensordata[i]) +
                    m.kd() * (m.dq() - mj_data_->sensordata[i + num_motor_]);
            }
        }

        // lowstate
        if (lowstate->trylock()) {
            for (int i = 0; i < num_motor_; ++i) {
                lowstate->msg_.motor_state()[i].q() = mj_data_->sensordata[i];
                lowstate->msg_.motor_state()[i].dq() = mj_data_->sensordata[i + num_motor_];
                lowstate->msg_.motor_state()[i].tau_est() = mj_data_->sensordata[i + 2 * num_motor_];
            }

            writeImuState(lowstate->msg_.imu_state(), imu_quat_adr_, imu_gyro_adr_, imu_acc_adr_);

            if (joystick) {
                const unitree::common::REMOTE_DATA_RX key = joystick->combine();
                std::memcpy(
                    &lowstate->msg_.wireless_remote()[0],
                    &key,
                    sizeof(unitree::common::REMOTE_DATA_RX));
            }

            lowstate->msg_.tick() = std::round(mj_data_->time / 1e-3);
            lowstate->unlockAndPublish();
        }

        // highstate
        if (highstate->trylock()) {
            if (frame_pos_adr_ >= 0) {
                highstate->msg_.position()[0] = mj_data_->sensordata[frame_pos_adr_ + 0];
                highstate->msg_.position()[1] = mj_data_->sensordata[frame_pos_adr_ + 1];
                highstate->msg_.position()[2] = mj_data_->sensordata[frame_pos_adr_ + 2];
            }
            if (frame_vel_adr_ >= 0) {
                highstate->msg_.velocity()[0] = mj_data_->sensordata[frame_vel_adr_ + 0];
                highstate->msg_.velocity()[1] = mj_data_->sensordata[frame_vel_adr_ + 1];
                highstate->msg_.velocity()[2] = mj_data_->sensordata[frame_vel_adr_ + 2];
            }
            highstate->unlockAndPublish();
        }

        // wireless_controller
        if (joystick && wireless_controller->trylock()) {
            wireless_controller->msg_.lx() = joystick->lx();
            wireless_controller->msg_.ly() = joystick->ly();
            wireless_controller->msg_.rx() = joystick->rx();
            wireless_controller->msg_.ry() = joystick->ry();
            wireless_controller->msg_.keys() = joystick->combine().RF_RX.btn.value;
            wireless_controller->unlockAndPublish();
        }
    }

    std::unique_ptr<HighState_t> highstate;
    std::unique_ptr<WirelessController_t> wireless_controller;
    std::shared_ptr<LowCmd_t> lowcmd;
    std::unique_ptr<LowState_t> lowstate;

private:
    unitree::common::RecurrentThreadPtr thread_;
};

using Go2Bridge = RobotBridge<unitree::robot::go2::subscription::LowCmd, unitree::robot::go2::publisher::LowState>;

class G1Bridge : public RobotBridge<unitree::robot::g1::subscription::LowCmd, unitree::robot::g1::publisher::LowState>
{
public:
    G1Bridge(mjModel* model, mjData* data, std::recursive_mutex* sim_mtx = nullptr)
    : RobotBridge(model, data, sim_mtx)
    {
        if (param::config.robot.find("g1") != std::string::npos) {
            const auto scene = param::config.robot_scene.filename().string();
            lowstate->msg_.mode_machine() = (scene.find("23") != std::string::npos) ? 4 : 5;
        }

        bmsstate = std::make_unique<BmsState_t>("rt/lf/bmsstate");
        bmsstate->msg_.soc() = 100;

        secondary_imustate = std::make_unique<IMUState_t>("rt/secondary_imu");
    }

    ~G1Bridge() override
    {
        stop();
    }

    void run() override
    {
        auto sim_lock = lockSim();

        RobotBridge::run();

        if (secondary_imustate->trylock()) {
            writeImuState(
                secondary_imustate->msg_,
                secondary_imu_quat_adr_,
                secondary_imu_gyro_adr_,
                secondary_imu_acc_adr_);
            secondary_imustate->unlockAndPublish();
        }

        // In practice, bmsstate is sent at a low frequency; here it is sent with the main loop.
        if (bmsstate->trylock()) {
            bmsstate->unlockAndPublish();
        }
    }

    using BmsState_t = unitree::robot::RealTimePublisher<unitree_hg::msg::dds_::BmsState_>;
    using IMUState_t = unitree::robot::RealTimePublisher<unitree_hg::msg::dds_::IMUState_>;
    std::unique_ptr<BmsState_t> bmsstate;
    std::unique_ptr<IMUState_t> secondary_imustate;
};
