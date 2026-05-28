#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = std::make_shared<Keyboard>();

void init_fsm_state()
{
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
    std::cout << "     Go2-W Controller \n";

    // Unitree DDS Config
    unitree::robot::ChannelFactory::Instance()->Init(0, vm["network"].as<std::string>());

    init_fsm_state();

    // Initialize FSM
    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
    fsm->start();

    if(param::start_mode == "gamepad")
    {
        std::cout << "Press [L2 + up] to enter FixStand mode.\n";
        std::cout << "And then press [R2 + A] to start controlling the robot.\n";
    }
    else if(param::start_mode == "keyboard")
    {
        std::cout << "Keyboard mode: Press [2] for FixStand, [3] for Velocity, [1] for Passive.\n";
    }
    else if(param::start_mode == "auto")
    {
        std::cout << "Auto mode: Passive -> FixStand (1s) -> Velocity (3s). Press [0] for emergency stop.\n";
    }

    if(param::command_mode == "patrol")
    {
        std::cout << "Command: patrol (forward 2m, backward 2m at 0.5 m/s)\n";
    }
    else if(param::command_mode == "fixed")
    {
        std::cout << "Command: fixed ["
                  << param::command_override[0] << ", "
                  << param::command_override[1] << ", "
                  << param::command_override[2] << "]\n";
    }

    while (true)
    {
        sleep(1);
    }
    
    return 0;
}
