#include <Mixer/Configuration/Configuration.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

TEST(ConfigurationTests, constructor_shouldInitializeSubConfigurations)
{
    Configuration configuration(Properties(
    {
        { "logger.level", "information" },
        { "logger.type", "console" },
        { "logger.filename", "log.txt" },

        { "audio.frame_sample_count", "32" },
        { "audio.sample_frequency", "48000" },
        { "audio.input_channel_count", "16" },
        { "audio.output_channel_count", "14" },
        { "audio.processing_data_type", "double" },
        { "audio.eq.center_frequencies", "[10, 20]" },
        { "audio.max_output_delay", "8192" },
        { "audio.analysis.sound_level_length", "4096" },
        { "audio.analysis.spectrum.fft_length", "2048" },
        { "audio.analysis.spectrum.point_count_per_decade", "10" },
        { "audio.headphone_channel_indexes", "[12, 13]"},

        { "audio.input.type", "raw_file" },
        { "audio.input.format", "signed_8" },
        { "audio.input.filename", "input.raw" },
        { "audio.input.looping", "false" },

        { "audio.output.type", "raw_file" },
        { "audio.output.format", "signed_16" },
        { "audio.output.filename", "output.raw" },
        { "audio.output.hardware_delay", "0.017" },

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
        { "uniformization.auto_position_count_threshold", "10" },

        { "web_socket.endpoint", "^/echo/?$" },
        { "web_socket.port", "8080" }
    }));

    EXPECT_EQ(configuration.logger().level(), Logger::Level::Information);
    EXPECT_EQ(configuration.logger().type(), LoggerConfiguration::Type::Console);
    EXPECT_EQ(configuration.logger().filename(), "");

    EXPECT_EQ(configuration.audio().frameSampleCount(), 32);
    EXPECT_EQ(configuration.audio().sampleFrequency(), 48000);
    EXPECT_EQ(configuration.audio().inputChannelCount(), 16);
    EXPECT_EQ(configuration.audio().outputChannelCount(), 14);
    EXPECT_EQ(configuration.audio().processingDataType(), ProcessingDataType::Double);
    EXPECT_EQ(configuration.audio().eqCenterFrequencies(), vector<double>({ 10, 20 }));
    EXPECT_EQ(configuration.audio().maxOutputDelay(), 8192);
    EXPECT_EQ(configuration.audio().soundLevelLength(), 4096);
    EXPECT_EQ(configuration.audio().spectrumAnalysisFftLength(), 2048);
    EXPECT_EQ(configuration.audio().spectrumAnalysisPointCountPerDecade(), 10);
    EXPECT_EQ(configuration.audio().headphoneChannelIndexes(), vector<size_t>({ 12, 13 }));

    EXPECT_EQ(configuration.audioInput().type(), AudioInputConfiguration::Type::RawFile);
    EXPECT_EQ(configuration.audioInput().format(), PcmAudioFrameFormat::Signed8);
    EXPECT_EQ(configuration.audioInput().filename(), "input.raw");
    EXPECT_EQ(configuration.audioInput().looping(), false);

    EXPECT_EQ(configuration.audioOutput().type(), AudioOutputConfiguration::Type::RawFile);
    EXPECT_EQ(configuration.audioOutput().format(), PcmAudioFrameFormat::Signed16);
    EXPECT_EQ(configuration.audioOutput().filename(), "output.raw");

    EXPECT_EQ(configuration.uniformization().discoveryEndpoint().ipAddress(), "192.168.1.255");
    EXPECT_EQ(configuration.uniformization().discoveryEndpoint().port(), 5000);
    EXPECT_EQ(configuration.uniformization().discoveryTimeoutMs(), 1000);
    EXPECT_EQ(configuration.uniformization().discoveryTrialCount(), 5);
    EXPECT_EQ(configuration.uniformization().tcpConnectionPort(), 5001);
    EXPECT_EQ(configuration.uniformization().udpReceivingPort(), 5002);
    EXPECT_EQ(configuration.uniformization().probeTimeoutMs(), 2000);
    EXPECT_EQ(configuration.uniformization().routineIRSweepF1(), 1);
    EXPECT_EQ(configuration.uniformization().routineIRSweepF2(), 2);
    EXPECT_EQ(configuration.uniformization().routineIRSweepT(), 3);
    EXPECT_DOUBLE_EQ(configuration.uniformization().speedOfSound(), 343);
    EXPECT_DOUBLE_EQ(configuration.uniformization().autoPositionAlpha(), 1.0);
    EXPECT_DOUBLE_EQ(configuration.uniformization().autoPositionEpsilonTotalDistanceError(), 5e-5);
    EXPECT_DOUBLE_EQ(configuration.uniformization().autoPositionEpsilonDeltaTotalDistanceError(), 1e-7);
    EXPECT_DOUBLE_EQ(configuration.uniformization().autoPositionDistanceRelativeError(), 0.0);
    EXPECT_EQ(configuration.uniformization().autoPositionIterationCount(), 10000);
    EXPECT_EQ(configuration.uniformization().autoPositionThermalIterationCount(), 200);
    EXPECT_EQ(configuration.uniformization().autoPositionTryCount(), 50);
    EXPECT_EQ(configuration.uniformization().autoPositionCountThreshold(), 10);
    EXPECT_DOUBLE_EQ(configuration.uniformization().routineIRSweepMaxDelay(), 0.5);

    EXPECT_EQ(configuration.webSocket().endpoint(), "^/echo/?$");
    EXPECT_EQ(configuration.webSocket().port(), 8080);

    SignalProcessorParameters signalProcessorParameters = configuration.toSignalProcessorParameters();
    EXPECT_EQ(signalProcessorParameters.processingDataType(), ProcessingDataType::Double);
    EXPECT_EQ(signalProcessorParameters.frameSampleCount(), 32);
    EXPECT_EQ(signalProcessorParameters.sampleFrequency(), 48000);
    EXPECT_EQ(signalProcessorParameters.inputChannelCount(), 16);
    EXPECT_EQ(signalProcessorParameters.outputChannelCount(), 14);
    EXPECT_EQ(signalProcessorParameters.inputFormat(), PcmAudioFrameFormat::Signed8);
    EXPECT_EQ(signalProcessorParameters.outputFormat(), PcmAudioFrameFormat::Signed16);
    EXPECT_EQ(signalProcessorParameters.eqCenterFrequencies(), vector<double>({ 10, 20 }));
    EXPECT_EQ(signalProcessorParameters.maxOutputDelay(), 8192);
    EXPECT_EQ(signalProcessorParameters.soundLevelLength(), 4096);

    UniformizationServiceParameters uniformizationServiceParameters = configuration.toUniformizationServiceParameters();
    EXPECT_EQ(uniformizationServiceParameters.discoveryEndpoint().ipAddress(), "192.168.1.255");
    EXPECT_EQ(uniformizationServiceParameters.discoveryEndpoint().port(), 5000);
    EXPECT_EQ(uniformizationServiceParameters.discoveryTimeoutMs(), 1000);
    EXPECT_EQ(uniformizationServiceParameters.discoveryTrialCount(), 5);
    EXPECT_EQ(uniformizationServiceParameters.tcpConnectionPort(), 5001);
    EXPECT_EQ(uniformizationServiceParameters.udpReceivingPort(), 5002);
    EXPECT_EQ(uniformizationServiceParameters.probeTimeoutMs(), 2000);
    EXPECT_EQ(uniformizationServiceParameters.sampleFrequency(), 48000);
    EXPECT_EQ(uniformizationServiceParameters.sweepDuration(), 3);
    EXPECT_DOUBLE_EQ(uniformizationServiceParameters.sweepMaxDelay(), 0.5);
    EXPECT_DOUBLE_EQ(uniformizationServiceParameters.outputHardwareDelay(), 0.017);
    EXPECT_EQ(uniformizationServiceParameters.speedOfSound(), 343);
    EXPECT_DOUBLE_EQ(uniformizationServiceParameters.autoPositionAlpha(), 1.0);
    EXPECT_DOUBLE_EQ(uniformizationServiceParameters.autoPositionEpsilonTotalDistanceError(), 5e-5);
    EXPECT_DOUBLE_EQ(uniformizationServiceParameters.autoPositionEpsilonDeltaTotalDistanceError(), 1e-7);
    EXPECT_DOUBLE_EQ(uniformizationServiceParameters.autoPositionDistanceRelativeError(), 0.0);
    EXPECT_EQ(uniformizationServiceParameters.autoPositionIterationCount(), 10000);
    EXPECT_EQ(uniformizationServiceParameters.autoPositionThermalIterationCount(), 200);
    EXPECT_EQ(uniformizationServiceParameters.autoPositionTryCount(), 50);
    EXPECT_EQ(uniformizationServiceParameters.autoPositionCountThreshold(), 10);
    EXPECT_EQ(uniformizationServiceParameters.eqCenterFrequencies(), vector<double>({ 10, 20 }));
    EXPECT_EQ(uniformizationServiceParameters.format(), PcmAudioFrameFormat::Signed16);

}
