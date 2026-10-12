#include <gtest/gtest.h>

#include "state/Propagator.h"
#include "state/State.h"
#include "types/IMU.h"
#include "update/UpdaterMSCKF.h"
#include "update/UpdaterSLAM.h"
#include "utils/sensor_data.h"

namespace {
TEST(FastPropagation, NonzeroOffsetsUseImuTargetAcrossCacheRebuilds) {
  for (const double offset : {-0.0052691525299192535, 0.02, 0.0}) {
    auto state = std::make_shared<ov_msckf::State>(ov_msckf::StateOptions{});
    state->_timestamp = 10.0;
    Eigen::Matrix<double, 1, 1> time_offset;
    time_offset << offset;
    state->_calib_dt_CAMtoIMU->set_value(time_offset);
    Eigen::Matrix<double, 16, 1> value = state->_imu->value();
    value.segment<3>(7) << 2.0, 0.0, 0.0;
    state->_imu->set_value(value);
    ov_msckf::Propagator propagator(ov_msckf::NoiseManager{}, 9.81);
    for (int i = 0; i <= 400; ++i) {
      ov_core::ImuData measurement;
      measurement.timestamp = 9.9 + i * 0.001;
      measurement.wm.setZero();
      measurement.am << 0.0, 0.0, 9.81;
      propagator.feed_imu(measurement);
    }
    Eigen::Matrix<double, 13, 1> predicted;
    Eigen::Matrix<double, 12, 12> covariance;
    for (const double target : {10.10, 10.15}) {
      ASSERT_TRUE(propagator.fast_state_propagate(state, target, predicted, covariance));
      EXPECT_NEAR(predicted(4), 2.0 * (target - 10.0 - offset), 1e-9);
      EXPECT_NEAR(predicted(6), 0.0, 1e-9);
      EXPECT_NEAR(predicted(7), 2.0, 1e-9);
      EXPECT_TRUE(covariance.allFinite());
    }
    state->_timestamp = 10.05;
    value(4) = 3.0;
    state->_imu->set_value(value);
    propagator.invalidate_cache();
    ASSERT_TRUE(propagator.fast_state_propagate(state, 10.16, predicted, covariance));
    EXPECT_NEAR(predicted(4), 3.0 + 2.0 * (10.16 - 10.05 - offset), 1e-9);
    EXPECT_FALSE(propagator.fast_state_propagate(state, 10.16, predicted, covariance));
    EXPECT_FALSE(propagator.fast_state_propagate(state, 10.15, predicted, covariance));
    ASSERT_TRUE(propagator.fast_state_propagate(state, 10.17, predicted, covariance));
    EXPECT_NEAR(predicted(4), 3.0 + 2.0 * (10.17 - 10.05 - offset), 1e-9);
  }
}

TEST(VisualFusion, EmptyFeatureUpdatesDoNotReportFusion) {
  auto state = std::make_shared<ov_msckf::State>(ov_msckf::StateOptions{});
  ov_msckf::UpdaterOptions options;
  ov_core::FeatureInitializerOptions initialization;
  ov_msckf::UpdaterMSCKF msckf(options, initialization);
  ov_msckf::UpdaterSLAM slam(options, options, initialization);
  std::vector<std::shared_ptr<ov_core::Feature>> features;
  EXPECT_FALSE(msckf.update(state, features));
  EXPECT_FALSE(slam.update(state, features));
}
} // namespace
