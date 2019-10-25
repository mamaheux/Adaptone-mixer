#include <Mixer/Configuration/UniformizationConfiguration.h>

#include <gtest/gtest.h>

using namespace adaptone;

TEST(UniformizationConfigurationTests, constructor_shouldSetTheAttributes)
{
    UniformizationConfiguration configuration(Properties(
    {
        { "uniformization.network.discovery_endpoint", "192.168.1.255:5000" },
        { "uniformization.network.discovery_timeout_ms", "1000" },
        { "uniformization.network.discovery_trial_count", "5" },

        { "uniformization.network.tcp_connection_port", "5001"},
        { "uniformization.network.udp_receiving_port", "5002"},
        { "uniformization.network.probe_timeout_ms", "2000"},

        { "uniformization.routine_ir_sweep_f1", "1" },
        { "uniformization.routine_ir_sweep_f2", "2" },
        { "uniformization.routine_ir_sweep_t", "3" },
        { "uniformization.routine_ir_sweep_max_delay", "0.5" },

        { "uniformization.speed_of_sound", "343" },

        { "uniformization.auto_position_alpha", "1.0" },
        { "uniformization.auto_position_epsilon_total_distance_error", "5e-5" },
        { "uniformization.auto_position_epsilon_delta_total_distance_error", "1e-7" },
        { "uniformization.auto_position_distance_relative_error", "0" },
        { "uniformization.auto_position_iteration_count", "10000" },
        { "uniformization.auto_position_thermal_iteration_count", "200" },
        { "uniformization.auto_position_try_count", "50" },
        { "uniformization.auto_position_count_threshold", "10" }
    }));

    EXPECT_EQ(configuration.discoveryEndpoint().ipAddress(), "192.168.1.255");
    EXPECT_EQ(configuration.discoveryEndpoint().port(), 5000);
    EXPECT_EQ(configuration.discoveryTimeoutMs(), 1000);
    EXPECT_EQ(configuration.discoveryTrialCount(), 5);

    EXPECT_EQ(configuration.tcpConnectionPort(), 5001);
    EXPECT_EQ(configuration.udpReceivingPort(), 5002);
    EXPECT_EQ(configuration.probeTimeoutMs(), 2000);

    EXPECT_EQ(configuration.routineIRSweepF1(), 1);
    EXPECT_EQ(configuration.routineIRSweepF2(), 2);
    EXPECT_EQ(configuration.routineIRSweepT(), 3);

    EXPECT_DOUBLE_EQ(configuration.routineIRSweepMaxDelay(), 0.5);
    EXPECT_DOUBLE_EQ(configuration.speedOfSound(), 343);

    EXPECT_DOUBLE_EQ(configuration.autoPositionAlpha(), 1.0);
    EXPECT_DOUBLE_EQ(configuration.autoPositionEpsilonTotalDistanceError(), 5e-5);
    EXPECT_DOUBLE_EQ(configuration.autoPositionEpsilonDeltaTotalDistanceError(), 1e-7);
    EXPECT_DOUBLE_EQ(configuration.autoPositionDistanceRelativeError(), 0.0);
    EXPECT_EQ(configuration.autoPositionIterationCount(), 10000);
    EXPECT_EQ(configuration.autoPositionThermalIterationCount(), 200);
    EXPECT_EQ(configuration.autoPositionTryCount(), 50);
    EXPECT_EQ(configuration.autoPositionCountThreshold(), 10);
}
