#include <Mixer/AudioInput/RawFileAudioInput.h>

#include <Utils/Exception/NotSupportedException.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

TEST(RawFileAudioInputTests, invalidFile_shouldThrowNotSupportedException)
{
    EXPECT_THROW(RawFileAudioInput(PcmAudioFrame::Format::Signed8,
        2,
        1,
        "resources/RawFileAudioInputTests/invalid.raw",
        false),
        NotSupportedException);
}

TEST(RawFileAudioInputTests, invalidFile_noLooping_shouldReadTheFile)
{
    RawFileAudioInput input(PcmAudioFrame::Format::Signed8,
        2,
        1,
        "resources/RawFileAudioInputTests/valid.raw",
        false);

    EXPECT_TRUE(input.hasNext());
    const PcmAudioFrame* frame = &input.read();
    EXPECT_EQ(frame->data()[0], 'a');
    EXPECT_EQ(frame->data()[1], 'b');

    EXPECT_TRUE(input.hasNext());
    frame = &input.read();
    EXPECT_EQ(frame->data()[0], 'c');
    EXPECT_EQ(frame->data()[1], 'd');

    EXPECT_FALSE(input.hasNext());
}

TEST(RawFileAudioInputTests, invalidFile_looping_shouldReadTheFileInLoop)
{
    RawFileAudioInput input(PcmAudioFrame::Format::Signed8,
        2,
        1,
        "resources/RawFileAudioInputTests/valid.raw",
        true);

    EXPECT_TRUE(input.hasNext());
    const PcmAudioFrame* frame = &input.read();
    EXPECT_EQ(frame->data()[0], 'a');
    EXPECT_EQ(frame->data()[1], 'b');

    EXPECT_TRUE(input.hasNext());
    frame = &input.read();
    EXPECT_EQ(frame->data()[0], 'c');
    EXPECT_EQ(frame->data()[1], 'd');

    EXPECT_TRUE(input.hasNext());
    frame = &input.read();
    EXPECT_EQ(frame->data()[0], 'a');
    EXPECT_EQ(frame->data()[1], 'b');
}
