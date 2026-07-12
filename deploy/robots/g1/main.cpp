#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"
#include "State_Mimic.h"

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = std::make_shared<Keyboard>();

void init_fsm_state()
{
    auto lowcmd_sub = std::make_shared<unitree::robot::g1::subscription::LowCmd>();
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

    // Use the keyboard as a virtual joystick when `--keyboard` is given.
    FSMState::use_keyboard = vm.count("keyboard") > 0;

    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     G1-29dof Controller \n";

    // Unitree DDS Config
    unitree::robot::ChannelFactory::Instance()->Init(0, vm["network"].as<std::string>());

    init_fsm_state();

    FSMState::lowcmd->msg_.mode_machine() = 5; // 29dof
    if(!FSMState::lowcmd->check_mode_machine(FSMState::lowstate)) {
        spdlog::critical("Unmatched robot type.");
        exit(-1);
    }

    // Initialize FSM
    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
    fsm->start();

    if(FSMState::use_keyboard)
    {
        std::cout << "\n--- Keyboard control enabled (virtual joystick) ---\n";
        std::cout << "  Mode:  [2] FixStand   [3] Velocity(walk)   [4] Mimic(dance)   [1] Passive(stop)\n";
        std::cout << "  Move:  [w/s] forward/back   [a/d] strafe   [q/e] turn   [space] stop\n";
        std::cout << "  Flow:  press 2 to stand, then 3 to walk, then use w/a/s/d/q/e.\n\n";
    }
    else
    {
        std::cout << "Press [L2 + Up] to enter FixStand mode.\n";
        std::cout << "And then press [R2 + A] to start controlling the robot.\n";
        std::cout << "And then press [R1 + A/B/Y/X] to control the robot dance.\n";
    }

    while (true)
    {
        sleep(1);
    }
    
    return 0;
}

