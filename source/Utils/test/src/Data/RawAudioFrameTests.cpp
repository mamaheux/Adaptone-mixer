#include <Utils/Data/RawAudioFrame.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

TEST(RawAudioFrameTests, construtor_shouldSetParameterAndAllocateMemory)
{
    RawAudioFrame frame(RawAudioFrame::Format::Signed24, 2, 3);
    for (size_t i = 0; i < 6; i++)
    {
        frame.data()[i] = i + 1;
    }

    EXPECT_EQ(frame.format(), RawAudioFrame::Format::Signed24);
    EXPECT_EQ(frame.channelCount(), 2);
    EXPECT_EQ(frame.sampleCount(), 3);
    EXPECT_EQ(frame.size(), 18);

    for (size_t i = 0; i < 6; i++)
    {
        EXPECT_EQ(frame.data()[i], i + 1);
    }
}

TEST(RawAudioFrameTests, copyConstrutor_shouldCopy)
{
    RawAudioFrame frame(RawAudioFrame::Format::Signed24, 2, 3);
    for (size_t i = 0; i < 6; i++)
    {
        frame.data()[i] = i + 1;
    }

    RawAudioFrame copy(frame);

    EXPECT_EQ(copy.format(), RawAudioFrame::Format::Signed24);
    EXPECT_EQ(copy.channelCount(), 2);
    EXPECT_EQ(copy.sampleCount(), 3);
    EXPECT_EQ(copy.size(), 18);
    EXPECT_NE(frame.data(), copy.data());

    for (size_t i = 0; i < 6; i++)
    {
        EXPECT_EQ(copy.data()[i], i + 1);
    }
}

TEST(RawAudioFrameTests, moveConstructor_shouldMove)
{
    RawAudioFrame frame(RawAudioFrame::Format::Signed24, 2, 3);
    for (size_t i = 0; i < 6; i++)
    {
        frame.data()[i] = i + 1;
    }

    uint8_t* data = frame.data();
    RawAudioFrame movedFrame(move(frame));

    EXPECT_EQ(movedFrame.format(), RawAudioFrame::Format::Signed24);
    EXPECT_EQ(movedFrame.channelCount(), 2);
    EXPECT_EQ(movedFrame.sampleCount(), 3);
    EXPECT_EQ(movedFrame.size(), 18);
    EXPECT_EQ(movedFrame.data(), data);

    for (size_t i = 0; i < 6; i++)
    {
        EXPECT_EQ(movedFrame.data()[i], i + 1);
    }

    EXPECT_EQ(frame.data(), nullptr);
    EXPECT_EQ(frame.channelCount(), 0);
    EXPECT_EQ(frame.sampleCount(), 0);
    EXPECT_EQ(frame.size(), 0);
}

TEST(RawAudioFrameTests, assignationOperator_shouldCopy)
{
    RawAudioFrame frame(RawAudioFrame::Format::Signed24, 2, 3);
    for (size_t i = 0; i < 6; i++)
    {
        frame.data()[i] = i + 1;
    }

    RawAudioFrame copy(RawAudioFrame::Format::Signed16, 1, 1);
    copy = frame;

    EXPECT_EQ(copy.format(), RawAudioFrame::Format::Signed24);
    EXPECT_EQ(copy.channelCount(), 2);
    EXPECT_EQ(copy.sampleCount(), 3);
    EXPECT_EQ(copy.size(), 18);
    EXPECT_NE(frame.data(), copy.data());

    for (size_t i = 0; i < 6; i++)
    {
        EXPECT_EQ(copy.data()[i], i + 1);
    }
}


TEST(RawAudioFrameTests, moveAssignationOperator_shouldCopy)
{
    RawAudioFrame frame(RawAudioFrame::Format::Signed24, 2, 3);
    for (size_t i = 0; i < 6; i++)
    {
        frame.data()[i] = i + 1;
    }

    uint8_t* data = frame.data();
    RawAudioFrame movedFrame(RawAudioFrame::Format::Signed16, 1, 1);
    movedFrame = move(frame);

    EXPECT_EQ(movedFrame.format(), RawAudioFrame::Format::Signed24);
    EXPECT_EQ(movedFrame.channelCount(), 2);
    EXPECT_EQ(movedFrame.sampleCount(), 3);
    EXPECT_EQ(movedFrame.size(), 18);
    EXPECT_EQ(movedFrame.data(), data);

    for (size_t i = 0; i < 6; i++)
    {
        EXPECT_EQ(movedFrame.data()[i], i + 1);
    }

    EXPECT_EQ(frame.data(), nullptr);
    EXPECT_EQ(frame.channelCount(), 0);
    EXPECT_EQ(frame.sampleCount(), 0);
    EXPECT_EQ(frame.size(), 0);
}