#include <Utils/Data/RawAudioFrame.h>

#include <Utils/Exception/InvalidValueException.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

TEST(RawAudioFrameTests, parseFormat_shouldReturnTheRightFormat)
{
    EXPECT_EQ(RawAudioFrame::parseFormat("signed_8"), RawAudioFrame::Format::Signed8);
    EXPECT_EQ(RawAudioFrame::parseFormat("signed_16"), RawAudioFrame::Format::Signed16);
    EXPECT_EQ(RawAudioFrame::parseFormat("signed_24"), RawAudioFrame::Format::Signed24);
    EXPECT_EQ(RawAudioFrame::parseFormat("signed_padded_24"), RawAudioFrame::Format::SignedPadded24);
    EXPECT_EQ(RawAudioFrame::parseFormat("signed_32"), RawAudioFrame::Format::Signed32);

    EXPECT_EQ(RawAudioFrame::parseFormat("unsigned_8"), RawAudioFrame::Format::Unsigned8);
    EXPECT_EQ(RawAudioFrame::parseFormat("unsigned_16"), RawAudioFrame::Format::Unsigned16);
    EXPECT_EQ(RawAudioFrame::parseFormat("unsigned_24"), RawAudioFrame::Format::Unsigned24);
    EXPECT_EQ(RawAudioFrame::parseFormat("unsigned_padded_24"), RawAudioFrame::Format::UnsignedPadded24);
    EXPECT_EQ(RawAudioFrame::parseFormat("unsigned_32"), RawAudioFrame::Format::Unsigned32);

    EXPECT_THROW(RawAudioFrame::parseFormat("unsigned_32asdasd"), InvalidValueException);
}

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
    RawAudioFrame frame(RawAudioFrame::Format::Unsigned24, 2, 3);
    for (size_t i = 0; i < 6; i++)
    {
        frame.data()[i] = i + 1;
    }

    uint8_t* data = frame.data();
    RawAudioFrame movedFrame(RawAudioFrame::Format::Signed16, 1, 1);
    movedFrame = move(frame);

    EXPECT_EQ(movedFrame.format(), RawAudioFrame::Format::Unsigned24);
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