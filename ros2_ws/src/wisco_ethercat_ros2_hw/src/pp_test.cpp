// One-drive bench test using ethercat_driver_ros2's master and generic PDO backend.
#include "pp_motion.hpp"

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <ethercat_generic_plugins/generic_ec_slave.hpp>
#include <ethercat_interface/ec_master.hpp>

#include <algorithm>
#include <charconv>
#include <csignal>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <thread>

namespace
{
using namespace pp_test;
using Slave = ethercat_generic_plugins::GenericEcSlave;
volatile std::sig_atomic_t interrupted = 0;
void interrupt(int) { interrupted = 1; }

std::string hex(uint32_t value)
{
  std::ostringstream out;
  out << "0x" << std::hex << value;
  return out.str();
}

uint32_t decode(const uint8_t * bytes, size_t size)
{
  switch (size) {
    case 1: return EC_READ_U8(bytes);
    case 2: return EC_READ_U16(bytes);
    case 4: return EC_READ_U32(bytes);
    default: throw std::runtime_error("Unexpected SDO width");
  }
}

class Bench : public ethercat_interface::EcMaster
{
public:
  explicit Bench(uint16_t position) : position_(position)
  {
    if (!master_) { throw std::runtime_error("Could not acquire IgH master 0. Stop other launches and check master access."); }
    ec_slave_info_t info{};
    if (ecrt_master_get_slave(master_, position_, &info) ||
      info.vendor_id != 0x00202008 || info.product_code != 0 || info.revision_number != 0x00010000)
    {
      throw std::runtime_error("No matching MT_Device at bus position " + std::to_string(position_));
    }
    uint8_t version[128]{};
    uint32_t abort = 0;
    size_t length = 0;
    std::cout << "MT_Device at bus position " << position_;
    if (read_before_activation(0x100a, 0, version, sizeof(version), length, abort) == 0) {
      const auto end = std::find(version, version + length, 0);
      std::cout << ", software version " << std::string(version, end);
    }
    std::cout << ".\n";
    const uint16_t status = upload(0x6041, 0, 2);
    if ((status & 4) || (status & 8)) {
      throw std::runtime_error("Drive must start disabled and fault-free, 0x6041=" + hex(status));
    }
    std::unordered_map<std::string, std::string> interfaces{
      {"slave_config", ament_index_cpp::get_package_share_directory("wisco_ethercat_ros2_hw") +
        "/config/pp_test.yaml"}};
    const std::vector<std::string> outputs{"control", "target", "zero", "kp", "kd", "mode",
      "velocity", "acceleration", "deceleration", "quick_stop"};
    const std::vector<std::string> inputs{"status", "actual", "error", "mode"};
    for (size_t i = 0; i < outputs.size(); ++i) {
      interfaces["command_interface/" + outputs[i]] = std::to_string(i);
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
      interfaces["state_interface/" + inputs[i]] = std::to_string(i);
    }
    command_.resize(outputs.size());
    state_.resize(inputs.size());
    if (!slave_.setupSlave(interfaces, &state_, &command_)) {
      throw std::runtime_error("Cannot load PP PDO configuration");
    }
    // Compare the live CoE mapping descriptors before asking IgH to configure the drive.
    for (size_t i = 0; i < slave_.syncSize(); ++i) {
      const auto & sm = slave_.syncs()[i];
      for (unsigned j = 0; j < sm.n_pdos; ++j) {
        const auto & pdo = sm.pdos[j];
        expect(pdo.index, 0, upload(pdo.index, 0, 1), pdo.n_entries);
        for (unsigned k = 0; k < pdo.n_entries; ++k) {
          const auto & entry = pdo.entries[k];
          const uint32_t descriptor = (uint32_t(entry.index) << 16) |
            (uint32_t(entry.subindex) << 8) | entry.bit_length;
          expect(pdo.index, k + 1, upload(pdo.index, k + 1, 4), descriptor);
        }
      }
    }
    command_[Target] = static_cast<int32_t>(upload(0x6064, 0, 4));
    command_[Kp] = static_cast<int32_t>(upload(0x2000, 0, 4));
    command_[Kd] = static_cast<int32_t>(upload(0x2001, 0, 4));
    command_[Mode] = 1;
    for (size_t i = 0; i < 4; ++i) {
      command_[Velocity + i] = upload(profile_indices[i], 0, 4);
    }
    original_ = command_;
    setCtrlFrequency(1000.0 / cycle_period.count());
    addSlave(0, position_, &slave_);
    if (!domain_info_.count(0) || !domain_info_.at(0)->domain || slave_info_.size() != 1) {
      throw std::runtime_error("Failed to register the PP domain");
    }
    mailbox_ = request(4);
    cap_mailbox_ = request(2);
    if (!activate()) { throw std::runtime_error("Could not activate PP domain"); }
  }

  void test(bool move)
  {
    const auto startup_deadline = Clock::now() + 30s;
    unsigned stable = 0;
    while (stable < 30) {
      check_interrupt();
      cycle();
      stable = healthy() && feedback().disabled() ? stable + 1 : 0;
      if (Clock::now() >= startup_deadline) {
        throw std::runtime_error("Timed out waiting for OP, complete WKC and switch-on disabled");
      }
    }
    expect_live(0x1c12, 0, 1, 2);
    expect_live(0x1c12, 1, 2, 0x1601);
    expect_live(0x1c12, 2, 2, 0x1602);
    expect_live(0x1c13, 0, 1, 1);
    expect_live(0x1c13, 1, 2, 0x1a00);

    // Exercise the profile channels while disabled and preserve quick-stop deceleration.
    command_[Velocity] = speed;
    command_[Acceleration] = acceleration;
    command_[Deceleration] = acceleration - 1;
    command_[QuickStop] = original_[QuickStop];  // retain the drive's quick-stop setting
    for (int i = 0; i < 20; ++i) { checked_cycle(); }
    for (size_t i = 0; i < 4; ++i) {
      expect_live(profile_indices[i], 0, 4, command_[Velocity + i]);
    }
    if (!feedback().disabled() || feedback().error) {
      throw std::runtime_error("Mapping check lost the disabled, fault-free drive state");
    }
    std::cout << "PASS mapping: Rx 0x1601 + 0x1602, Tx 0x1a00, OP, complete WKC "
              << domain_info_.at(0)->domain_state.working_counter
              << ", profile values read back, torque disabled. Mode display "
              << feedback().mode << ".\n";
    if (!move) { return; }

    original_cap_ = read_live(cap_mailbox_, 0x6072, 0, 2);
    const uint16_t cap = original_cap_ == 0 ? 100 : std::min<uint16_t>(original_cap_, 100);
    if (cap != original_cap_) {
      restore_cap_ = true;
      write_cap(cap);
    }
    expect(0x6072, 0, read_live(cap_mailbox_, 0x6072, 0, 2), cap);
    if (feedback().mode != 1 || !feedback().disabled()) {
      throw std::runtime_error("Expected PP mode display 1 and switch-on disabled before enable");
    }
    Motion motion(feedback().actual, Clock::now());
    motion_active_ = true;
    do {
      command_[Control] = motion.control;
      command_[Target] = motion.target;
      checked_cycle();
      motion.update(feedback(), Clock::now());
    } while (motion.phase != Motion::done);
    const int32_t reached = feedback().actual;
    confirm_disabled();
    std::cout << "PP target reached: start " << motion.start << ", target " << motion.target
              << ", actual " << reached << " counts. Bit 12 acknowledgement edge "
              << (motion.ack_edge ? "observed" : "not observed") << ".\n";
  }

  void finish()
  {
    cleaning_ = true;
    confirm_disabled();
    for (size_t i = 0; i < 4; ++i) { command_[Velocity + i] = original_[Velocity + i]; }
    for (int i = 0; i < 20; ++i) { checked_cycle(); }
    if (restore_cap_) {
      write_cap(original_cap_);
      expect(0x6072, 0, read_live(cap_mailbox_, 0x6072, 0, 2), original_cap_);
    }
    for (size_t i = 0; i < 4; ++i) {
      expect_live(profile_indices[i], 0, 4, original_[Velocity + i]);
    }
    std::cout << "Drive disabled. Original profile restored.\n";
  }

  std::string diagnostic() const
  {
    auto f = feedback();
    ec_slave_config_state_t slave{};
    ecrt_slave_config_state(slave_info_.at(0).config, &slave);
    return "AL=" + hex(slave.al_state) + " WKC=" +
      std::to_string(domain_info_.at(0)->domain_state.working_counter) +
      " status=" + hex(f.status) + " error=" + hex(f.error) + " mode=" +
      std::to_string(f.mode) + " actual=" + std::to_string(f.actual) +
      " target=" + std::to_string(static_cast<int32_t>(command_[Target]));
  }

private:
  enum {Control, Target, Zero, Kp, Kd, Mode, Velocity, Acceleration, Deceleration, QuickStop};
  static constexpr uint16_t profile_indices[4] = {0x6081, 0x6083, 0x6084, 0x6085};
  static constexpr auto cycle_period = 10ms;
  uint16_t position_;
  Slave slave_;
  std::vector<double> state_, command_, original_;
  ec_sdo_request_t * mailbox_ = nullptr;
  ec_sdo_request_t * cap_mailbox_ = nullptr;
  uint16_t original_cap_ = 0;
  bool restore_cap_ = false;
  bool motion_active_ = false;
  bool cleaning_ = false;
  Clock::time_point next_ = Clock::now();

  Feedback feedback() const
  {
    return {static_cast<uint16_t>(state_[0]), static_cast<int32_t>(state_[1]),
      static_cast<int>(state_[3]), static_cast<uint16_t>(state_[2])};
  }
  int read_before_activation(uint16_t index, uint8_t sub, uint8_t *bytes, size_t capacity,
    size_t &length, uint32_t &abort)
  {
    for (int attempt = 1; ; ++attempt) {
      check_interrupt();
      length = 0;
      abort = 0;
      const int result = ecrt_master_sdo_upload(master_, position_, index, sub, bytes,
        capacity, &length, &abort);
      if (result == 0 || abort != 0 || attempt == 3) { return result; }
      std::cerr << "Position " << position_ << ": no SDO response value for " << hex(index)
                << ":" << unsigned(sub) << " (rc=" << result << "), retrying.\n";
      std::this_thread::sleep_for(250ms * attempt);
    }
  }
  uint32_t upload(uint16_t index, uint8_t sub, size_t size)
  {
    alignas(uint32_t) uint8_t bytes[4]{};
    size_t actual = 0;
    uint32_t abort = 0;
    const int result = read_before_activation(index, sub, bytes, sizeof(bytes), actual, abort);
    if (result != 0) {
      throw std::runtime_error("Pre-activation SDO transfer failed at " + hex(index) + ":" +
        std::to_string(sub) + " (rc=" + std::to_string(result) + ", abort=" + hex(abort) + ")");
    }
    if (actual != size) {
      throw std::runtime_error("Wrong SDO width at " + hex(index) + ":" + std::to_string(sub) +
        ": expected " + std::to_string(size) + " bytes, received " + std::to_string(actual));
    }
    return decode(bytes, size);
  }
  static void expect(uint16_t index, uint8_t sub, uint32_t actual, uint32_t wanted)
  {
    if (actual != wanted) {
      throw std::runtime_error("SDO " + hex(index) + ":" + std::to_string(sub) +
        " expected " + hex(wanted) + ", read " + hex(actual));
    }
  }
  void check_interrupt() const
  {
    if (interrupted && !cleaning_) { throw std::runtime_error("Interrupted"); }
  }
  bool healthy() const
  {
    ec_slave_config_state_t slave{};
    ecrt_slave_config_state(slave_info_.at(0).config, &slave);
    auto & domain = domain_info_.at(0)->domain_state;
    return slave.online && slave.operational && slave.al_state == 8 &&
           domain.wc_state == EC_WC_COMPLETE && domain.working_counter > 0;
  }
  void cycle()
  {
    next_ = std::max(next_ + cycle_period, Clock::now());
    std::this_thread::sleep_until(next_);
    update();
  }
  void checked_cycle()
  {
    check_interrupt();
    cycle();
    if (!healthy()) { throw std::runtime_error("Lost slave OP or complete domain WKC"); }
    if (!motion_active_ && !feedback().disabled()) {
      throw std::runtime_error("Drive left switch-on disabled during the passive check");
    }
  }
  void confirm_disabled()
  {
    command_[Control] = 0;
    const auto deadline = Clock::now() + 2s;
    unsigned stable = 0;
    while (stable < 20) {
      cycle();
      stable = healthy() && feedback().disabled() ? stable + 1 : 0;
      if (Clock::now() >= deadline) {
        throw std::runtime_error("Could not confirm switch-on disabled; check the drive on the bench");
      }
    }
    motion_active_ = false;
  }
  ec_sdo_request_t * request(size_t size)
  {
    auto * req = ecrt_slave_config_create_sdo_request(slave_info_.at(0).config, 0x6041, 0, size);
    if (!req) { throw std::runtime_error("Failed to allocate cyclic SDO request"); }
    ecrt_sdo_request_timeout(req, 1000);
    return req;
  }
  void wait_sdo(ec_sdo_request_t * req, uint16_t index)
  {
    const auto deadline = Clock::now() + 2s;
    for (;;) {
      checked_cycle();
      auto state = ecrt_sdo_request_state(req);
      if (state == EC_REQUEST_SUCCESS) { return; }
      if (state != EC_REQUEST_BUSY || Clock::now() >= deadline) {
        throw std::runtime_error("Cyclic SDO transfer failed at " + hex(index));
      }
    }
  }
  uint32_t read_live(ec_sdo_request_t * req, uint16_t index, uint8_t sub, size_t size)
  {
    if (ecrt_sdo_request_state(req) == EC_REQUEST_BUSY) { wait_sdo(req, index); }
    if (ecrt_sdo_request_index(req, index, sub) || ecrt_sdo_request_read(req)) {
      throw std::runtime_error("Could not schedule SDO read at " + hex(index));
    }
    wait_sdo(req, index);
    if (ecrt_sdo_request_data_size(req) != size) {
      throw std::runtime_error("Wrong SDO width at " + hex(index));
    }
    return decode(ecrt_sdo_request_data(req), size);
  }
  void expect_live(uint16_t index, uint8_t sub, size_t size, uint32_t value)
  {
    expect(index, sub, read_live(mailbox_, index, sub, size), value);
  }
  void write_cap(uint16_t value)
  {
    if (ecrt_sdo_request_state(cap_mailbox_) == EC_REQUEST_BUSY) { wait_sdo(cap_mailbox_, 0x6072); }
    if (ecrt_sdo_request_index(cap_mailbox_, 0x6072, 0)) {
      throw std::runtime_error("Could not address current limit 0x6072");
    }
    EC_WRITE_U16(ecrt_sdo_request_data(cap_mailbox_), value);
    if (ecrt_sdo_request_write(cap_mailbox_)) {
      throw std::runtime_error("Could not schedule current limit write");
    }
    wait_sdo(cap_mailbox_, 0x6072);
  }
};
std::vector<uint16_t> connected_positions()
{
  std::unique_ptr<ec_master_t, decltype(&ecrt_release_master)> master(
    ecrt_open_master(0), ecrt_release_master);
  ec_master_info_t info{};
  if (!master || ecrt_master(master.get(), &info) || !info.link_up || info.scan_busy ||
    !info.slave_count || info.slave_count > 65536u) {
    throw std::runtime_error("Cannot enumerate a ready EtherCAT ring on master 0");
  }
  std::vector<uint16_t> positions;
  for (unsigned pos = 0; pos < info.slave_count; ++pos) {
    ec_slave_info_t drive{};
    if (ecrt_master_get_slave(master.get(), pos, &drive) ||
      drive.vendor_id != 0x00202008 || drive.product_code != 0 || drive.revision_number != 0x00010000)
    {
      throw std::runtime_error("Unexpected device at position " + std::to_string(pos));
    }
    positions.push_back(pos);
  }
  return positions;
}

int run_position(uint16_t position, bool move)
{
  try {
    Bench bench(position);
    std::string failure;
    try {
      bench.test(move);
    } catch (const std::exception & error) {
      failure = std::string(error.what()) + " (" + bench.diagnostic() + ")";
    }
    // Send disable before writing diagnostics to a potentially blocked terminal.
    try {
      bench.finish();
    } catch (const std::exception & error) {
      if (!failure.empty()) { std::cerr << "FAIL: " << failure << '\n'; }
      throw;
    }
    if (!failure.empty()) { std::cerr << "FAIL: " << failure << '\n'; }
    if (interrupted) { return 130; }
    if (!failure.empty()) { return 1; }
    if (move) { std::cout << "PASS PP motion.\n"; }
    return 0;
  } catch (const std::exception & error) {
    std::cerr << "pp_test position " << position << ": " << error.what() << '\n';
    return 1;
  }
}
}  // namespace

int main(int argc, char ** argv)
{
  const char * usage = "Usage: ros2 run wisco_ethercat_ros2_hw pp_test (--position N | --all) [--move]\n";
  if (argc == 2 && std::string(argv[1]) == "--help") { std::cout << usage; return 0; }
  uint16_t position = 0;
  bool have_position = false, all = false, move = false;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--position" && !have_position && i + 1 < argc) {
      std::string number = argv[++i];
      auto end = number.data() + number.size();
      auto parsed = std::from_chars(number.data(), end, position);
      if (parsed.ec != std::errc{} || parsed.ptr != end) { std::cerr << usage; return 2; }
      have_position = true;
    } else if (arg == "--all" && !all) {
      all = true;
    } else if (arg == "--move" && !move) {
      move = true;
    } else { std::cerr << usage; return 2; }
  }
  if (have_position == all) { std::cerr << usage; return 2; }
  std::vector<uint16_t> positions;
  try {
    positions = all ? connected_positions() : std::vector<uint16_t>{position};
  } catch (const std::exception & error) {
    std::cerr << "pp_test: " << error.what() << '\n';
    return 1;
  }
  if (move) {
    std::cout << "Selected bus positions:";
    for (auto selected : positions) { std::cout << ' ' << selected; }
    std::cout << ". Each will move +" << step_degrees << " encoder degrees, one drive at a time. "
              << "Support the arm throughout and check clearance at each joint. Type MOVE to proceed: "
              << std::flush;
    std::string reply;
    if (!std::getline(std::cin, reply) || reply != "MOVE" || interrupted) {
      std::cerr << "Cancelled before acquiring the bus.\n"; return 2;
    }
  }
  std::signal(SIGINT, interrupt);
  std::signal(SIGTERM, interrupt);
  std::vector<uint16_t> failed_positions;
  for (uint16_t selected : positions) {
    if (interrupted) { return 130; }
    const int code = run_position(selected, move);
    if (interrupted) { return 130; }
    if (code != 0 && all && !move) {
      failed_positions.push_back(selected);
    } else if (code != 0) {
      std::cerr << "Stopped at bus position " << selected << ".\n";
      return code;
    }
  }
  if (all && !move) {
    std::cout << "Mapping passed on " << positions.size() - failed_positions.size()
              << " of " << positions.size() << " drives.\n";
    if (!failed_positions.empty()) {
      std::cerr << "Failed bus positions:";
      for (auto pos : failed_positions) { std::cerr << ' ' << pos; }
      std::cerr << '\n';
      return 1;
    }
  }
  return 0;
}
