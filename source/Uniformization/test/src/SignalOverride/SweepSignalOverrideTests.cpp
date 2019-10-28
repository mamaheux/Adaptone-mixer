#include <Uniformization/SignalOverride/SweepSignalOverride.h>

#include <Uniformization/Math.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace arma;

TEST(SweepSignalOverrideTests, override_shouldOverrideWithSweepFrameData)
{
    constexpr size_t LoopCount = 2;

    constexpr PcmAudioFrameFormat Format = PcmAudioFrameFormat::Double;

    constexpr size_t Fs = 44100;
    constexpr double F1 = 20;
    constexpr double F2 = 10000;
    constexpr double Period = 1;

    constexpr size_t ChannelCount = 4;
    constexpr size_t FrameSampleCount = 256;
    constexpr size_t OutputChannelIndex = 2;

    SweepSignalOverride signalOverride(Format, Fs, ChannelCount,
        FrameSampleCount, F1, F2, Period);

    vec chirp = logSinChirp<vec>(F1, F2, Period, Fs);
    constexpr size_t ExtraFrameCount = 1; //needed to verify that the sweepSignalOverride stop outputting chirp frames
    size_t frameCount = floor(chirp.size() / static_cast<float>(FrameSampleCount)) + ExtraFrameCount;

    for (size_t n = 0; n < LoopCount; n++)
    {
        signalOverride.startSweep(OutputChannelIndex);

        for (size_t k = 0; k < frameCount; k++)
        {
            PcmAudioFrame frame(Format, ChannelCount, FrameSampleCount);

            for (size_t i = 0; i < frame.size(); i++)
            {
                frame[i] = i + 1;
            }

            const PcmAudioFrame& overridenPcmAudioFrame = signalOverride.override(frame);
            AudioFrame<double> overridenAudioFrame(const_cast<PcmAudioFrame&>(overridenPcmAudioFrame));

            vec sweepFrameData;
            if (signalOverride.isSweepActive())
            {
                sweepFrameData = chirp(span(k * FrameSampleCount, (k + 1) * FrameSampleCount));
            }
            else
            {
                sweepFrameData = zeros<vec>(FrameSampleCount);
            }

            for (size_t i = 0; i < FrameSampleCount; i++)
            {
                EXPECT_DOUBLE_EQ(overridenAudioFrame[OutputChannelIndex * FrameSampleCount + i], sweepFrameData(i));
            }

            for (size_t j = 0; j < ChannelCount; j++)
            {
                if (j % ChannelCount != OutputChannelIndex)
                {
                    for (size_t i = 0; i < FrameSampleCount; i++)
                    {
                        EXPECT_DOUBLE_EQ(overridenAudioFrame[j * FrameSampleCount + i], 0);
                    }
                }
            }
        }
    }
}
