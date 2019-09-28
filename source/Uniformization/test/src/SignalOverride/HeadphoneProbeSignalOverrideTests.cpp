#include <Uniformization/SignalOverride/HeadphoneProbeSignalOverride.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

TEST(HeadphoneProbeSignalOverrideTests, override_shouldWriteProbeData)
{
    constexpr PcmAudioFrameFormat Format = PcmAudioFrameFormat::Unsigned8;
    constexpr size_t ChannelCount = 4;
    constexpr size_t FrameSampleCount = 32;
    const vector<size_t> HeadphoneChannelIndexes{ 2, 3 };
    constexpr size_t ProbeId = 0;

    HeadphoneProbeSignalOverride signalOverride(Format, ChannelCount, FrameSampleCount, HeadphoneChannelIndexes);
    PcmAudioFrame frame(PcmAudioFrameFormat::Unsigned8, ChannelCount, FrameSampleCount);
    for (size_t i = 0; i < frame.size(); i++)
    {
        frame[i] = i;
    }

    signalOverride.setCurrentProbeId(ProbeId);
    for (uint8_t value = 0; value < numeric_limits<uint8_t>::max(); value++)
    {
        vector<uint8_t> data(FrameSampleCount, value);

        signalOverride.writeData(ProbeSoundDataMessage(0, 0, 0, 0, 0, 0, data.data(), data.size()), ProbeId);

        const PcmAudioFrame& overridenFrame = signalOverride.override(frame);
        for (size_t i = 0; i < frame.size(); i++)
        {
            if (i % ChannelCount == 0 || i % ChannelCount == 1)
            {
                EXPECT_EQ(overridenFrame[i], i);
            }
            else
            {
                EXPECT_EQ(overridenFrame[i], value);
            }
        }
    }
}

TEST(HeadphoneProbeSignalOverrideTests, override_wrongProbeId_shouldDoNothing)
{
    constexpr PcmAudioFrameFormat Format = PcmAudioFrameFormat::Unsigned8;
    constexpr size_t ChannelCount = 4;
    constexpr size_t FrameSampleCount = 32;
    const vector<size_t> HeadphoneChannelIndexes{ 2, 3 };
    constexpr size_t ProbeId = 0;
    constexpr uint8_t Value = 1;

    HeadphoneProbeSignalOverride signalOverride(Format, ChannelCount, FrameSampleCount, HeadphoneChannelIndexes);
    PcmAudioFrame frame(PcmAudioFrameFormat::Unsigned8, ChannelCount, FrameSampleCount);
    for (size_t i = 0; i < frame.size(); i++)
    {
        frame[i] = i;
    }

    signalOverride.setCurrentProbeId(ProbeId);
    vector<uint8_t> data(FrameSampleCount, Value);
    signalOverride.writeData(ProbeSoundDataMessage(0, 0, 0, 0, 0, 0, data.data(), data.size()), ProbeId);

    data = vector<uint8_t>(FrameSampleCount, 0);
    signalOverride.writeData(ProbeSoundDataMessage(0, 0, 0, 0, 0, 0, data.data(), data.size()), ProbeId + 1);

    const PcmAudioFrame& overridenFrame = signalOverride.override(frame);
    for (size_t i = 0; i < frame.size(); i++)
    {
        if (i % ChannelCount == 0 || i % ChannelCount == 1)
        {
            EXPECT_EQ(overridenFrame[i], i);
        }
        else
        {
            EXPECT_EQ(overridenFrame[i], Value);
        }
    }
}
