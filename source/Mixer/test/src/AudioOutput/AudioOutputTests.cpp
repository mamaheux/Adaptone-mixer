#include <Mixer/AudioOutput/AudioOutput.h>

#include <Utils/Exception/NotSupportedException.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

class DummyAudioOutput : public AudioOutput
{
public:
    DummyAudioOutput() : AudioOutput(PcmAudioFrameFormat::Signed8, 1, 2)
    {
    }

    ~DummyAudioOutput() override
    {
    }

    void write(const PcmAudioFrame& frame) override
    {
    }

    PcmAudioFrameFormat format() const
    {
        return m_format;
    }

    size_t channelCount() const
    {
        return m_channelCount;
    }

    size_t frameSampleCount() const
    {
        return m_frameSampleCount;
    }
};

TEST(AudioOutputTests, constructor_shouldInitializeTheParameters)
{
    DummyAudioOutput output;
    EXPECT_EQ(output.format(), PcmAudioFrameFormat::Signed8);
    EXPECT_EQ(output.channelCount(), 1);
    EXPECT_EQ(output.frameSampleCount(), 2);
}
