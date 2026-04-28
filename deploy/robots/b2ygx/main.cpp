#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"
#include "LocalJoystick.h"
#include "unitree/robot/b2/motion_switcher/motion_switcher_client.hpp"

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = nullptr;

namespace
{

std::string query_service_name(const std::string& form, const std::string& name)
{
    if(form == "0")
    {
        if(name == "normal") return "sport_mode";
        if(name == "ai") return "ai_sport";
        if(name == "advanced") return "advanced_sport";
    }
    else
    {
        if(name == "ai-w") return "wheeled_sport(go2W)";
        if(name == "normal-w") return "wheeled_sport(b2W)";
    }
    return "";
}

bool query_motion_status(unitree::robot::b2::MotionSwitcherClient& msc)
{
    std::string robot_form;
    std::string motion_name;
    int32_t ret = msc.CheckMode(robot_form, motion_name);
    if(ret == 0) {
        spdlog::info("CheckMode succeeded.");
    } else {
        spdlog::warn("CheckMode failed. Error code: {}", ret);
    }

    if(motion_name.empty())
    {
        spdlog::info("The motion control-related service is deactivated.");
        return false;
    }

    std::string service_name = query_service_name(robot_form, motion_name);
    spdlog::warn("Service: {} is active.", service_name.empty() ? motion_name : service_name);
    return true;
}

void release_motion_control_service()
{
    unitree::robot::b2::MotionSwitcherClient msc;
    msc.SetTimeout(10.0f);
    msc.Init();

    while(query_motion_status(msc))
    {
        spdlog::info("Try to deactivate the motion control-related service.");
        int32_t ret = msc.ReleaseMode();
        if(ret == 0) {
            spdlog::info("ReleaseMode succeeded.");
        } else {
            spdlog::warn("ReleaseMode failed. Error code: {}", ret);
        }
        sleep(5);
    }
}

} // namespace

void init_fsm_state()
{
    auto joystick_cfg = param::config["Joystick"];
    if(joystick_cfg && joystick_cfg["enabled"].as<bool>(false))
    {
        const auto type = joystick_cfg["type"].as<std::string>("xbox");
        const auto device = joystick_cfg["device"].as<std::string>("/dev/input/js0");
        const auto bits = joystick_cfg["bits"].as<int>(16);
        if(type == "xbox") {
            FSMState::local_joystick = std::make_shared<XBoxJoystick>(device, bits);
        } else if(type == "switch") {
            FSMState::local_joystick = std::make_shared<SwitchJoystick>(device, bits);
        } else {
            spdlog::warn("Unsupported local joystick type: {}", type);
        }
        if(FSMState::local_joystick) {
            spdlog::info("Using local {} joystick from {}", type, device);
        }
    }

    auto lowcmd_sub = std::make_shared<unitree::robot::go2::subscription::LowCmd>();
    usleep(0.2 * 1e6);
    if(!lowcmd_sub->isTimeout())
    {
        spdlog::critical("The other process is using the lowcmd channel, please close it first.");
        unitree::robot::go2::shutdown();
        // exit(0);
    }
    FSMState::lowcmd = std::make_unique<LowCmd_t>();
    FSMState::lowstate = std::make_shared<LowState_t>();
    spdlog::info("Waiting for connection to robot...");
    FSMState::lowstate->wait_for_connection();
    spdlog::info("Connected to robot.");
}

int main(int argc, char** argv)
{
    // Load parameters
    auto vm = param::helper(argc, argv);

    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     B2YGX Controller \n";

    // Unitree DDS Config
    unitree::robot::ChannelFactory::Instance()->Init(0, vm["network"].as<std::string>());

    release_motion_control_service();

    init_fsm_state();

    // Initialize FSM
    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
    fsm->start();

    std::cout << "Press [L2 + Up] to enter FixStand mode.\n";
    std::cout << "And then press [R2 + A] to start controlling the robot.\n";

    while (true)
    {
        sleep(1);
    }
    
    return 0;
}
