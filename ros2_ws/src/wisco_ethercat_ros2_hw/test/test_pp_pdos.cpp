#include <gtest/gtest.h>
#include <ethercat_generic_plugins/generic_ec_slave.hpp>
#include <array>
#include <vector>

TEST(PPPdoMapping, GenericRosDriverExchangesVendorPackedFrames)
{
  ethercat_generic_plugins::GenericEcSlave slave;
  std::vector<double> states(4);
  std::vector<double> commands{0x1f, -32001, 0, -12, 15, 1, 728, 2913, 2912, 8000};
  std::unordered_map<std::string, std::string> interfaces{{"slave_config", PP_TEST_CONFIG}};
  size_t i = 0;
  for (const auto name : {"control", "target", "zero", "kp", "kd", "mode", "velocity",
    "acceleration", "deceleration", "quick_stop"})
  {
    interfaces["command_interface/" + std::string(name)] = std::to_string(i++);
  }
  i = 0;
  for (const auto name : {"status", "actual", "error", "mode"}) {
    interfaces["state_interface/" + std::string(name)] = std::to_string(i++);
  }
  ASSERT_TRUE(slave.setupSlave(interfaces, &states, &commands));
  const auto & sm2 = slave.syncs()[2];
  const auto & sm3 = slave.syncs()[3];
  ASSERT_EQ(sm2.index, 2);
  ASSERT_EQ(sm2.n_pdos, 2u);
  ASSERT_EQ(sm2.pdos[0].index, 0x1601);
  ASSERT_EQ(sm2.pdos[1].index, 0x1602);
  ASSERT_EQ(sm3.index, 3);
  ASSERT_EQ(sm3.n_pdos, 1u);
  ASSERT_EQ(sm3.pdos[0].index, 0x1a00);

  // IgH registers the byte offsets for each entry in SM2, then SM3.
  std::array<uint8_t, 54> frame{};
  EC_WRITE_U16(&frame[38], 0x1637);
  EC_WRITE_S32(&frame[40], -32111);
  EC_WRITE_U16(&frame[50], 0xff01);
  EC_WRITE_S8(&frame[52], 1);
  size_t offset = 0, entry = 0;
  for (const auto * sm : {&sm2, &sm3}) {
    for (unsigned p = 0; p < sm->n_pdos; ++p) {
      for (unsigned n = 0; n < sm->pdos[p].n_entries; ++n) {
        const auto width = sm->pdos[p].entries[n].bit_length;
        ASSERT_EQ(width % 8, 0);
        ASSERT_LE(offset + width / 8, frame.size());
        slave.processData(entry++, &frame[offset]);
        offset += width / 8;
      }
    }
  }
  EXPECT_EQ(offset, frame.size());
  EXPECT_EQ(EC_READ_U16(&frame[0]), 0x1f);
  EXPECT_EQ(EC_READ_S32(&frame[2]), -32001);
  EXPECT_EQ(EC_READ_S32(&frame[12]), -12);
  EXPECT_EQ(EC_READ_S32(&frame[16]), 15);
  EXPECT_EQ(frame[20], 1);
  EXPECT_EQ(EC_READ_U32(&frame[22]), 728u);
  EXPECT_EQ(EC_READ_U32(&frame[26]), 2913u);
  EXPECT_EQ(EC_READ_U32(&frame[30]), 2912u);
  EXPECT_EQ(EC_READ_U32(&frame[34]), 8000u);
  EXPECT_EQ(states, (std::vector<double>{0x1637, -32111, 0xff01, 1}));
}
