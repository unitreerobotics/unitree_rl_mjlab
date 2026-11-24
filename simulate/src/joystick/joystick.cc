// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Copyright Drew Noakes 2013-2016

#include "joystick.h"

Joystick::Joystick()
{
  openPath("/dev/input/js0");
}

Joystick::Joystick(int joystickNumber)
{
  std::stringstream sstm;
  sstm << "/dev/input/js" << joystickNumber;
  openPath(sstm.str());
}

Joystick::Joystick(std::string devicePath)
{
  openPath(devicePath);
}

Joystick::Joystick(std::string devicePath, bool blocking)
{
  openPath(devicePath, blocking);
}

Joystick::Joystick(Joystick &&other) noexcept
  : _fd(other._fd), event_(other.event_)
{
  std::copy(std::begin(other.button_), std::end(other.button_), std::begin(button_));
  std::copy(std::begin(other.axis_), std::end(other.axis_), std::begin(axis_));
  other._fd = -1;
}

Joystick &Joystick::operator=(Joystick &&other) noexcept
{
  if (this != &other)
  {
    if (_fd >= 0)
    {
      close(_fd);
    }
    _fd = other._fd;
    event_ = other.event_;
    std::copy(std::begin(other.button_), std::end(other.button_), std::begin(button_));
    std::copy(std::begin(other.axis_), std::end(other.axis_), std::begin(axis_));
    other._fd = -1;
  }
  return *this;
}

void Joystick::openPath(std::string devicePath, bool blocking)
{
  // Open the device using either blocking or non-blocking
  _fd = open(devicePath.c_str(), blocking ? O_RDONLY : O_RDONLY | O_NONBLOCK);
}

bool Joystick::sample(JoystickEvent *event)
{
  ssize_t bytes = read(_fd, event, sizeof(*event));

  if (bytes == -1)
    return false;

  // NOTE if this condition is not met, we're probably out of sync and this
  // Joystick instance is likely unusable
  return bytes == static_cast<ssize_t>(sizeof(*event));
}

bool Joystick::isFound()
{
  return _fd >= 0;
}

void Joystick::getState()
{
  while (sample(&event_))
  {
    if (event_.isButton())
    {
      if (event_.number < kMaxButtons)
      {
        button_[event_.number] = event_.value;
      }
    }
    else if (event_.isAxis())
    {
      if (event_.number < kMaxAxes)
      {
        axis_[event_.number] = event_.value;
      }
    }
  }
}

Joystick::~Joystick()
{
  if (_fd >= 0)
  {
    close(_fd);
    _fd = -1;
  }
}

std::ostream &operator<<(std::ostream &os, const JoystickEvent &e)
{
  os << "type=" << static_cast<int>(e.type)
     << " number=" << static_cast<int>(e.number)
     << " value=" << static_cast<int>(e.value);
  return os;
}
