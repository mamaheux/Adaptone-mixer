#include <Mixer/AudioInput/AudioInput.h>

#include <Utils/Exception/NotSupportedException.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

class DummyAudioInput : public AudioInput
{
public:
    DummyAudioInput() : AudioInput(PcmAudioFrame::Format::Signed8, 1, 1)
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

TEST(RawFileAudioInputTests, hasGainControl_shouldReturnFalse)
{
    DummyAudioInput input;
    EXPECT_FALSE(input.hasGainControl());
}

TEST(RawFileAudioInputTests, setGain_shouldThrowNotSupportedException)
{
    DummyAudioInput input;
    EXPECT_THROW(input.setGain(0, 0), NotSupportedException);
}