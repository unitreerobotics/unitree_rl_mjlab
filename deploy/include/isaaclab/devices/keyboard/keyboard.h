#pragma once

#include <atomic>
#include <chrono>
#include <deque>
#include <iostream>
#include <string>
#include <sys/select.h>
#include <termios.h>
#include <thread>
#include <unistd.h>
#include <vector>


/**
 * @brief Maintain a keyboard reading thread.
 * And get the latest key value.
 */
class Keyboard
{
public:
  Keyboard()
  {
    tcgetattr( fileno( stdin ), &_oldSettings );
    _newSettings = _oldSettings;
    _oldSettings.c_lflag |= ( ICANON |  ECHO);
    _newSettings.c_lflag &= (~ICANON & ~ECHO);

    _startKey();

    _thread_running  = true;
    _readThread = std::thread([this] {
      while (_thread_running) {
        if (_running) {
          _read();
        } else {
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
      }
    });
  }

  ~Keyboard()
  {
    _thread_running = false;
    if (_readThread.joinable()) {
      _readThread.join();
    }
    _pauseKey();
  }

  void update()
  {
    if(_key != _last_key)
    {
      on_pressed = _key != "";
      on_released = _key == "";
    }
    else
    {
      on_pressed = false;
      on_released = false;
    }

    _last_key = _key;
  }

  /**
   * @brief Get the current key value
   *
   * @return std::string
   */
  std::string key() const { return _key; };

  /**
   * @brief Get the String object from keyboard
   *
   * @param slogan Used to prompt the user for input
   * @return std::string
   */
  std::string getString(std::string slogan)
  {
    // Stop reading keyboard value
    _running = false;
    _pauseKey();

    std::string stringtemp;
    std::cout << slogan << std::endl;// prompt
    std::getline(std::cin, stringtemp);

    // Restart reading keyboard value
    _startKey();
    _running = true;

    return stringtemp;
  }

  /**
   * flags; available after update()
   */
  bool on_pressed = false;
  bool on_released = false;

  private:
  std::atomic<bool> _thread_running{false};
  std::atomic<bool> _running{false};
  std::thread _readThread;

  void _read()
  {
    if(_running)
    {
      FD_ZERO(&_fd_set);
      FD_SET( fileno(stdin), &_fd_set);

      _tv.tv_sec = 0;
      _tv.tv_usec = 80000;

      if(select(fileno(stdin)+1, &_fd_set, NULL, NULL, &_tv))
      {
        // Read the key value into _c
        int res = read( fileno(stdin), &_c, 1 );

        // Parser the key value
        if(_c != '\033') {
          // This is a normal key
          _key = _c;
        }else{
          // This is a special key
          int m = read(fileno(stdin), &_c, 1);
          if(_c == '[')
          {
            m = read(fileno(stdin), &_c, 1);
            switch (_c)
            {
            case 'A': _key = "up";    break;
            case 'B': _key = "down";  break;
            case 'C': _key = "right"; break;
            case 'D': _key = "left";  break;
            default:  _key = "";      break;
            }
          }
        }
      }else{
        _key = "";
      }
      // std::cout << "key: "<< key() << std::endl;
    }
  }

  /**
   * @brief Restore keyboard default settings.
   */
  void _pauseKey()
  {
    tcsetattr( fileno( stdin ), TCSANOW, &_oldSettings );
    _running = false;
  }

  /**
   * @brief Disable canonical mode and echoing of input characters.
   */
  void _startKey()
  {
    tcsetattr( fileno( stdin ), TCSANOW, &_newSettings );
    _running = true;
  }

  fd_set _fd_set;
  char _c = '\0';
  std::string _key, _last_key;

  termios _oldSettings, _newSettings;
  timeval _tv;
};
