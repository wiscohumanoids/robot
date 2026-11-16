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

#ifndef ADS7128_HPP_
#define ADS7128_HPP_

#include <fcntl.h>
#include <linux/i2c-dev.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>

namespace wisco
{

/// Minimal userspace driver for the TI ADS7128 (8-channel, 12-bit, I2C ADC).
///
/// Talks to the chip through the Linux i2c-dev interface (/dev/i2c-N) and uses
/// the ADC's manual mode: every read_raw() selects one channel (MANUAL_CHID),
/// then a plain 2-byte read starts a conversion (the chip stretches SCL for
/// ~1 us) and returns the result. One conversion per call: no averaging
/// (OSR_CFG left at reset), no CRC, no filtering of any kind.
class Ads7128
{
public:
  static constexpr int kNumChannels = 8;
  static constexpr uint16_t kFullScaleCounts = 4096;  // 12-bit converter

  Ads7128() = default;
  ~Ads7128() {close_device();}
  Ads7128(const Ads7128 &) = delete;
  Ads7128 & operator=(const Ads7128 &) = delete;

  /// Open the bus, bind the slave address, then reset and offset-calibrate the ADC.
  bool open(const std::string & device, uint8_t address)
  {
    close_device();
    fd_ = ::open(device.c_str(), O_RDWR);
    if (fd_ < 0) {
      return fail_errno("open " + device);
    }
    if (::ioctl(fd_, I2C_SLAVE, static_cast<unsigned long>(address)) < 0) {
      fail_errno("bind I2C address (EBUSY = a kernel driver already owns it)");
      close_device();
      return false;
    }
    if (!initialise()) {
      close_device();
      return false;
    }
    return true;
  }

  /// Run one conversion on AIN<channel>. counts is the raw 12-bit code, 0..4095.
  bool read_raw(uint8_t channel, uint16_t & counts)
  {
    if (fd_ < 0 || channel >= kNumChannels) {
      return fail_msg("read_raw: device not open or channel out of range");
    }
    // MANUAL_CHID lives in CHANNEL_SEL[3:0]; ZCD_CHID[7:4] stays 0.
    if (!write_reg(kRegChannelSel, channel)) {
      return false;
    }
    // The read frame's address ACK starts the conversion. Data is MSB-first
    // D11..D4, then D3..D0 in the upper nibble of the second byte.
    uint8_t rx[2] = {0, 0};
    const ssize_t n = ::read(fd_, rx, sizeof(rx));
    if (n < 0) {
      return fail_errno("conversion read");
    }
    if (n != static_cast<ssize_t>(sizeof(rx))) {
      return fail_msg("conversion read: short read");
    }
    counts = static_cast<uint16_t>((rx[0] << 4) | (rx[1] >> 4));
    return true;
  }

  const std::string & last_error() const {return error_;}

private:
  // Command opcodes (datasheet Table 9).
  static constexpr uint8_t kOpRegRead = 0x10;
  static constexpr uint8_t kOpRegWrite = 0x08;

  // Registers.
  static constexpr uint8_t kRegSystemStatus = 0x00;
  static constexpr uint8_t kRegGeneralCfg = 0x01;
  static constexpr uint8_t kRegChannelSel = 0x11;

  // SYSTEM_STATUS bits.
  static constexpr uint8_t kStatusRsvd = 0x80;         // always reads 1
  static constexpr uint8_t kStatusCrcErrFuse = 0x04;   // power-up config CRC failed

  // GENERAL_CFG bits.
  static constexpr uint8_t kGenCfgRst = 0x01;
  static constexpr uint8_t kGenCfgCal = 0x02;

  bool initialise()
  {
    uint8_t status = 0;
    if (!read_reg(kRegSystemStatus, status)) {
      return false;
    }
    if (!(status & kStatusRsvd)) {
      return fail_msg("SYSTEM_STATUS bit 7 reads 0: device at this address is not an ADS7128");
    }

    // Soft reset to a known state: all pins analog, manual mode, CRC off, no averaging.
    if (!write_reg(kRegGeneralCfg, kGenCfgRst)) {
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));  // tRST max is 5 ms

    if (!read_reg(kRegSystemStatus, status)) {
      return false;
    }
    if (!(status & kStatusRsvd)) {
      return fail_msg("no valid SYSTEM_STATUS after reset");
    }
    if (status & kStatusCrcErrFuse) {
      return fail_msg("ADS7128 reports a power-up configuration CRC error (CRC_ERR_FUSE)");
    }

    // ADC offset calibration; the chip clears the CAL bit when it is done.
    if (!write_reg(kRegGeneralCfg, kGenCfgCal)) {
      return false;
    }
    for (int i = 0; i < 100; ++i) {
      uint8_t cfg = 0;
      if (!read_reg(kRegGeneralCfg, cfg)) {
        return false;
      }
      if (!(cfg & kGenCfgCal)) {
        return true;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return fail_msg("ADC offset calibration did not complete");
  }

  bool write_reg(uint8_t reg, uint8_t value)
  {
    const uint8_t tx[3] = {kOpRegWrite, reg, value};
    const ssize_t n = ::write(fd_, tx, sizeof(tx));
    if (n < 0) {
      return fail_errno("register write");
    }
    if (n != static_cast<ssize_t>(sizeof(tx))) {
      return fail_msg("register write: short write");
    }
    return true;
  }

  bool read_reg(uint8_t reg, uint8_t & value)
  {
    const uint8_t tx[2] = {kOpRegRead, reg};
    ssize_t n = ::write(fd_, tx, sizeof(tx));
    if (n < 0) {
      return fail_errno("register read (address phase)");
    }
    if (n != static_cast<ssize_t>(sizeof(tx))) {
      return fail_msg("register read: short write");
    }
    n = ::read(fd_, &value, 1);
    if (n < 0) {
      return fail_errno("register read (data phase)");
    }
    if (n != 1) {
      return fail_msg("register read: short read");
    }
    return true;
  }

  void close_device()
  {
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
  }

  bool fail_errno(const std::string & what)
  {
    error_ = what + ": " + std::strerror(errno);
    return false;
  }

  bool fail_msg(const std::string & what)
  {
    error_ = what;
    return false;
  }

  int fd_{-1};
  std::string error_;
};

}  // namespace wisco

#endif  // ADS7128_HPP_
