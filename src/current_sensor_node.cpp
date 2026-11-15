// Copyright 2026 WiscoHumanoids
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include "ads7128.hpp"

namespace wisco
{
namespace
{

// ---------------------------------------------------------------------------
// Hardcoded sensor / ADC configuration
// ---------------------------------------------------------------------------

// The ADS7128 uses AVDD as its reference, and AVDD is the shared 3.3 V sensor rail.
constexpr double kAdcRefVolts = 3.3;

// ACS758 (bidirectional) sensitivity, as specified by the team.
// NOTE: the datasheet sensitivity is ratiometric to VCC and quoted at 5 V.
// If this value is the 5 V figure, the 3.3 V-rail equivalent is
// 13.2 * 3.3 / 5.0 = ~8.71 mV/A. See the accompanying notes.
constexpr double kSensitivityMvPerA = 13.2;

constexpr double kAmpsPerCount =
  (kAdcRefVolts / Ads7128::kFullScaleCounts) * 1000.0 / kSensitivityMvPerA;

struct SensorChannel
{
  const char * name;     // printed label (PLACEHOLDER net names, edit to match the PDB)
  uint8_t adc_channel;   // ADS7128 input index: AIN<adc_channel>
  double zero_counts;    // hardcoded zero-current offset, in ADC counts
};

// Ideal zero-current code is 2048 (VCC/2 on both the sensor and the ADC reference).
// Replace each with the value printed by this node (print_raw_counts:=true) with
// the rails unloaded.
constexpr SensorChannel kSensors[] = {
  {"Main_curr", 0, 2048.0},        // main branch, after bulk capacitors
  {"Buck24_curr", 1, 2048.0},      // after 24 V buck, before the compute rails
  {"Actuators_curr", 2, 2048.0},   // all actuator/servo rails combined
  {"Act1_curr", 3, 2048.0},        // actuator rail 1
  {"Act2_curr", 4, 2048.0},        // actuator rail 2
  {"Act3_curr", 5, 2048.0},        // actuator rail 3
  {"Act4_curr", 6, 2048.0},        // actuator rail 4
};

}  // namespace

/// Polls the ACS758 current sensors through an ADS7128 on I2C and printf()s amperes.
///
/// Parameters:
///   i2c_device       (string, "/dev/i2c-1")  i2c-dev node the ADC is on
///   i2c_address      (int,    16)            7-bit address (0x10 = ADDR pin unconnected)
///   poll_period_ms   (int,    100)           time between sweeps of all 7 channels
///   print_raw_counts (bool,   false)         also print the raw ADC code per channel
///
/// Raw single-conversion samples only: there is deliberately no averaging or
/// filtering, in software or in the ADC.
class CurrentSensorNode : public rclcpp::Node
{
public:
  explicit CurrentSensorNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : Node("current_sensor_node", options)
  {
    const auto i2c_device = this->declare_parameter<std::string>("i2c_device", "/dev/i2c-1");
    const auto i2c_address = this->declare_parameter<int64_t>("i2c_address", 0x10);
    const auto poll_period_ms = this->declare_parameter<int64_t>("poll_period_ms", 100);
    print_raw_ = this->declare_parameter<bool>("print_raw_counts", false);

    if (i2c_address < 0x08 || i2c_address > 0x77) {
      throw std::invalid_argument("i2c_address must be a valid 7-bit address (0x08..0x77)");
    }
    if (poll_period_ms < 1) {
      throw std::invalid_argument("poll_period_ms must be >= 1");
    }

    if (!adc_.open(i2c_device, static_cast<uint8_t>(i2c_address))) {
      throw std::runtime_error("ADS7128 init failed on " + i2c_device + ": " + adc_.last_error());
    }

    RCLCPP_INFO(
      this->get_logger(), "ADS7128 ready on %s @ 0x%02x, polling %zu sensors every %ld ms.",
      i2c_device.c_str(), static_cast<unsigned>(i2c_address), sizeof(kSensors) / sizeof(kSensors[0]),
      static_cast<long>(poll_period_ms));

    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(poll_period_ms), [this]() {poll();});
  }

private:
  void poll()
  {
    const double t =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
    std::printf("[%9.3f]", t);

    for (const auto & sensor : kSensors) {
      uint16_t counts = 0;
      if (!adc_.read_raw(sensor.adc_channel, counts)) {
        std::printf(" %s=ERR", sensor.name);
        RCLCPP_ERROR_THROTTLE(
          this->get_logger(), *this->get_clock(), 2000, "ADS7128 read failed on %s (AIN%d): %s",
          sensor.name, static_cast<int>(sensor.adc_channel), adc_.last_error().c_str());
        continue;
      }
      const double amps = (static_cast<double>(counts) - sensor.zero_counts) * kAmpsPerCount;
      if (print_raw_) {
        std::printf(" %s=%+8.3fA(%4u)", sensor.name, amps, static_cast<unsigned>(counts));
      } else {
        std::printf(" %s=%+8.3fA", sensor.name, amps);
      }
    }
    std::printf("\n");
  }

  Ads7128 adc_;
  bool print_raw_{false};
  std::chrono::steady_clock::time_point start_{std::chrono::steady_clock::now()};
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace wisco

int main(int argc, char * argv[])
{
  // Under `ros2 launch` stdout is a pipe and would otherwise be block-buffered,
  // so the printf() lines would show up in large delayed chunks.
  std::setvbuf(stdout, nullptr, _IOLBF, 0);

  rclcpp::init(argc, argv);
  int rc = 0;
  try {
    rclcpp::spin(std::make_shared<wisco::CurrentSensorNode>());
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("current_sensor_node"), "%s", e.what());
    rc = 1;
  }
  rclcpp::shutdown();
  return rc;
}
