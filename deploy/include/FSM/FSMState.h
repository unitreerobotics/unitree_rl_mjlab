#pragma once

#include <algorithm>

#include "Types.h"
#include "param.h"
#include "FSM/BaseState.h"
#include "isaaclab/devices/keyboard/keyboard.h"
#include "unitree_joystick_dsl.hpp"

class FSMState : public BaseState
{
public:
    FSMState(int state, std::string state_string) 
    : BaseState(state, state_string) 
    {
        spdlog::info("Initializing State_{} ...", state_string);

        auto transitions = param::config["FSM"][state_string]["transitions"];

        if(transitions)
        {
            auto transition_map = transitions.as<std::map<std::string, std::string>>();

            for(auto it = transition_map.begin(); it != transition_map.end(); ++it)
            {
                std::string target_fsm = it->first;
                if(!FSMStringMap.right.count(target_fsm))
                {
                    spdlog::warn("FSM State_'{}' not found in FSMStringMap!", target_fsm);
                    continue;
                }

                int fsm_id = FSMStringMap.right.at(target_fsm);

                std::string condition = it->second;
                unitree::common::dsl::Parser p(condition);
                auto ast = p.Parse();
                auto func = unitree::common::dsl::Compile(*ast);
                registered_checks.emplace_back(
                    std::make_pair(
                        [func]()->bool{ return func(FSMState::lowstate->joystick); },
                        fsm_id
                    )
                );
            }
        }

        // register for all states
        registered_checks.emplace_back(
            std::make_pair(
                []()->bool{ return lowstate->isTimeout(); },
                FSMStringMap.right.at("Passive")
            )
        );
    }

    void pre_run()
    {
        lowstate->update();
        if(keyboard) keyboard->update();
        if(use_keyboard && keyboard) keyboard_control();
    }

    void post_run()
    {
        lowcmd->unlockAndPublish();
    }

    static std::unique_ptr<LowCmd_t> lowcmd;
    static std::shared_ptr<LowState_t> lowstate;
    static std::shared_ptr<Keyboard> keyboard;

    // Enabled by the `--keyboard` flag. When true, the keyboard drives a
    // virtual joystick so the robot can be controlled without a gamepad.
    inline static bool use_keyboard = false;

    /**
     * @brief Translate keyboard input into `lowstate->joystick`, so that both the
     * velocity commands and the FSM transitions (which all read the joystick)
     * behave exactly as if a real gamepad were connected.
     *
     * Controls:
     *   Locomotion (latched velocity setpoints, tap to adjust):
     *     w / s : forward / backward       (vx)
     *     a / d : strafe left / right      (vy)
     *     q / e : turn left / right        (yaw)
     *     space / x : stop (zero velocity)
     *   Mode switches (single press, emulate the gamepad combos):
     *     1 : Passive   (LT + B)   -- damping / soft stop
     *     2 : FixStand  (LT + up)  -- stand up
     *     3 : Velocity  (RT + A)   -- start the walking policy
     *     4 : Mimic     (RB + A)   -- dance
     */
    void keyboard_control()
    {
        auto & js = lowstate->joystick;

        // Keyboard input is digital: disable the analog smoothing so a key press
        // maps to the axis/trigger value immediately (and releases cleanly).
        js.lx.smooth = js.ly.smooth = js.rx.smooth = 1.f;
        js.LT.smooth = js.RT.smooth = 1.f;

        // Latched velocity setpoints, in joystick-axis units [-1, 1].
        static float sp_ly = 0.f;  // + forward
        static float sp_lx = 0.f;  // + strafe right
        static float sp_rx = 0.f;  // + turn right
        constexpr float kStep = 0.2f;

        const std::string k = keyboard->key();

        // React only on the key-down edge (avoids terminal auto-repeat quirks).
        if(keyboard->on_pressed)
        {
            bool cmd_changed = true;
            if(k == "w")      sp_ly = std::clamp(sp_ly + kStep, -1.f, 1.f);
            else if(k == "s") sp_ly = std::clamp(sp_ly - kStep, -1.f, 1.f);
            else if(k == "d") sp_lx = std::clamp(sp_lx + kStep, -1.f, 1.f);
            else if(k == "a") sp_lx = std::clamp(sp_lx - kStep, -1.f, 1.f);
            else if(k == "e") sp_rx = std::clamp(sp_rx + kStep, -1.f, 1.f);
            else if(k == "q") sp_rx = std::clamp(sp_rx - kStep, -1.f, 1.f);
            else if(k == " " || k == "x") { sp_ly = sp_lx = sp_rx = 0.f; }
            else cmd_changed = false;

            if(cmd_changed)
            {
                // Observation mapping: vx = ly, vy = -lx, wz = -rx.
                spdlog::info("[keyboard] cmd  vx={:+.2f}  vy={:+.2f}  wz={:+.2f}",
                             sp_ly, -sp_lx, -sp_rx);
            }

            // FSM mode switches: press the corresponding virtual button combo for
            // this single frame, and reset the velocity for a safe hand-off.
            if(k == "1")      { js.LT(1.f); js.B(1);  sp_ly = sp_lx = sp_rx = 0.f; }
            else if(k == "2") { js.LT(1.f); js.up(1); sp_ly = sp_lx = sp_rx = 0.f; }
            else if(k == "3") { js.RT(1.f); js.A(1);  sp_ly = sp_lx = sp_rx = 0.f; }
            else if(k == "4") { js.RB(1);   js.A(1);  sp_ly = sp_lx = sp_rx = 0.f; }
        }

        // Drive the velocity axes continuously from the latched setpoints.
        js.lx(sp_lx);
        js.ly(sp_ly);
        js.rx(sp_rx);
    }
};