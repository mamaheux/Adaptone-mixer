#include <Mixer/AudioOutput/RawFileAudioOutput.h>

#include <Utils/Exception/InvalidValueException.h>

#include <gtest/gtest.h>

#include <cstdio>

using namespace adaptone;
using namespace std;

constexpr const char* AudioOutputFilename = "output.raw";

class RawFileAudioOutputTests : public ::testing::Test
{
protected:
    void SetUp() override
    {
    }

    void TearDown() override
    {
        remove(AudioOutputFilename);
    }
};

TEST_F(RawFileAudioOutputTests, write_wrongFormat_shouldThrowInvalidValueException)
{
    PcmAudioFrame frame(PcmAudioFrame::Format::Signed8, 1, 2);
    RawFileAudioOutput output(PcmAudioFrame::Format::Signed16, 1, 2, AudioOutputFilename);

    EXPECT_THROW(output.write(frame), InvalidValueException);
}

TEST_F(RawFileAudioOutputTests, write_wrongChannelCount_shouldThrowInvalidValueException)
{
    PcmAudioFrame frame(PcmAudioFrame::Format::Signed8, 1, 2);
    RawFileAudioOutput output(PcmAudioFrame::Format::Signed8, 2, 2, AudioOutputFilename);

    EXPECT_THROW(output.write(frame), InvalidValueException);
}

TEST_F(RawFileAudioOutputTests, write_wrongFrameSampleCount_shouldThrowInvalidValueException)
{
    PcmAudioFrame frame(PcmAudioFrame::Format::Signed8, 1, 2);
    RawFileAudioOutput output(PcmAudioFrame::Format::Signed8, 1, 1, AudioOutputFilename);

    EXPECT_THROW(output.write(frame), InvalidValueException);
}

TEST_F(RawFileAudioOutputTests, write_validFrame_shouldWriteTheData)
{
    {
        PcmAudioFrame frame(PcmAudioFrame::Format::Signed8, 1, 2);
        frame[0] = 'a';
        frame[1] = 'b';

        RawFileAudioOutput output(PcmAudioFrame::Format::Signed8, 1, 2, AudioOutputFilename);
        output.write(frame);
    }

    ifstream audioOutputFileStream(AudioOutputFilename);

    string line;
    getline(audioOutputFileStream, line);
    EXPECT_EQ(line, "ab");
}
