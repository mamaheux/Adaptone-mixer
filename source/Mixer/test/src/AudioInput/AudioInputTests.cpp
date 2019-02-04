#include <Mixer/AudioInput/AudioInput.h>

#include <Utils/Exception/NotSupportedException.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

class DummyAudioInput : public AudioInput
{
public:
    DummyAudioInput() : AudioInput(PcmAudioFrame::Format::Signed8, 1, 2)
    {
    }

    ~DummyAudioInput() override
    {
    }

    const PcmAudioFrame& read() override
    {
        return m_frame;
    }

    bool hasNext() override
    {
        return true;
    }
};

TEST(AudioInputTests, constructor_shouldInitializeTheFrame)
{
    DummyAudioInput input;
    EXPECT_EQ(input.read().format(), PcmAudioFrame::Format::Signed8);
    EXPECT_EQ(input.read().channelCount(), 1);
    EXPECT_EQ(input.read().sampleCount(), 2);
}

TEST(AudioInputTests, hasGainControl_shouldReturnFalse)
{
    DummyAudioInput input;
    EXPECT_FALSE(input.hasGainControl());
}

TEST(AudioInputTests, setGain_shouldThrowNotSupportedException)
{
    DummyAudioInput input;
    EXPECT_THROW(input.setGain(0, 0), NotSupportedException);
}
