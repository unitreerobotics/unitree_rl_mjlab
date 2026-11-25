#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = nullptr;

void init_fsm_state()
{
    auto lowcmd_sub = std::make_shared<unitree::robot::g1::subscription::LowCmd>();
    lowcmd_sub->set_timeout_ms(200);
    usleep(300000);
    if (!lowcmd_sub->isTimeout())
    {
        spdlog::warn("Another process is using the lowcmd channel; attempting to release onboard motion service...");
        unitree::robot::go2::shutdown();
        usleep(300000);
        if (!lowcmd_sub->isTimeout())
        {
            spdlog::critical("The other process is still using the lowcmd channel, please close it first.");
            exit(EXIT_FAILURE);
        }
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
    std::cout << "     H2 Controller \n";

    // Unitree DDS Config
    unitree::robot::ChannelFactory::Instance()->Init(0, vm["network"].as<std::string>());

    init_fsm_state();

    // H2 echoes the machine mode reported by the robot and commands joints in PR mode.
    {
        std::lock_guard<std::mutex> lock(FSMState::lowstate->mutex_);
        FSMState::lowcmd->msg_.mode_machine() = FSMState::lowstate->msg_.mode_machine();
    }
    FSMState::lowcmd->msg_.mode_pr() = 0;

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
