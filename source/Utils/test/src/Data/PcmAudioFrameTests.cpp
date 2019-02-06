#include <Utils/Data/PcmAudioFrame.h>

#include <Utils/Exception/InvalidValueException.h>

#include <gtest/gtest.h>

#include <sstream>

using namespace adaptone;
using namespace std;

TEST(PcmAudioFrameTests, parseFormat_shouldReturnTheRightFormat)
{
    EXPECT_EQ(PcmAudioFrame::parseFormat("signed_8"), PcmAudioFrame::Format::Signed8);
    EXPECT_EQ(PcmAudioFrame::parseFormat("signed_16"), PcmAudioFrame::Format::Signed16);
    EXPECT_EQ(PcmAudioFrame::parseFormat("signed_24"), PcmAudioFrame::Format::Signed24);
    EXPECT_EQ(PcmAudioFrame::parseFormat("signed_padded_24"), PcmAudioFrame::Format::SignedPadded24);
    EXPECT_EQ(PcmAudioFrame::parseFormat("signed_32"), PcmAudioFrame::Format::Signed32);

    EXPECT_EQ(PcmAudioFrame::parseFormat("unsigned_8"), PcmAudioFrame::Format::Unsigned8);
    EXPECT_EQ(PcmAudioFrame::parseFormat("unsigned_16"), PcmAudioFrame::Format::Unsigned16);
    EXPECT_EQ(PcmAudioFrame::parseFormat("unsigned_24"), PcmAudioFrame::Format::Unsigned24);
    EXPECT_EQ(PcmAudioFrame::parseFormat("unsigned_padded_24"), PcmAudioFrame::Format::UnsignedPadded24);
    EXPECT_EQ(PcmAudioFrame::parseFormat("unsigned_32"), PcmAudioFrame::Format::Unsigned32);

    EXPECT_EQ(PcmAudioFrame::parseFormat("float"), PcmAudioFrame::Format::Float);
    EXPECT_EQ(PcmAudioFrame::parseFormat("double"), PcmAudioFrame::Format::Double);

    EXPECT_THROW(PcmAudioFrame::parseFormat("unsigned_32asdasd"), InvalidValueException);
}

TEST(PcmAudioFrameTests, formatSize_shouldReturnTheRightSize)
{
    EXPECT_EQ(PcmAudioFrame::formatSize(PcmAudioFrame::Format::Signed8), 1);
    EXPECT_EQ(PcmAudioFrame::formatSize(PcmAudioFrame::Format::Signed16), 2);
    EXPECT_EQ(PcmAudioFrame::formatSize(PcmAudioFrame::Format::Signed24), 3);
    EXPECT_EQ(PcmAudioFrame::formatSize(PcmAudioFrame::Format::SignedPadded24), 4);
    EXPECT_EQ(PcmAudioFrame::formatSize(PcmAudioFrame::Format::Signed32), 4);

    EXPECT_EQ(PcmAudioFrame::formatSize(PcmAudioFrame::Format::Unsigned8), 1);
    EXPECT_EQ(PcmAudioFrame::formatSize(PcmAudioFrame::Format::Unsigned16), 2);
    EXPECT_EQ(PcmAudioFrame::formatSize(PcmAudioFrame::Format::Unsigned24), 3);
    EXPECT_EQ(PcmAudioFrame::formatSize(PcmAudioFrame::Format::UnsignedPadded24), 4);
    EXPECT_EQ(PcmAudioFrame::formatSize(PcmAudioFrame::Format::Unsigned32), 4);

    EXPECT_EQ(PcmAudioFrame::formatSize(PcmAudioFrame::Format::Float), 4);
    EXPECT_EQ(PcmAudioFrame::formatSize(PcmAudioFrame::Format::Double), 8);
}

TEST(PcmAudioFrameTests, construtor_shouldSetParameterAndAllocateMemory)
{
    PcmAudioFrame frame(PcmAudioFrame::Format::Signed24, 2, 3);
    for (size_t i = 0; i < 6; i++)
    {
        frame[i] = i + 1;
    }

    EXPECT_EQ(frame.format(), PcmAudioFrame::Format::Signed24);
    EXPECT_EQ(frame.channelCount(), 2);
    EXPECT_EQ(frame.sampleCount(), 3);
    EXPECT_EQ(frame.size(), 18);

    for (size_t i = 0; i < 6; i++)
    {
        EXPECT_EQ(frame.data()[i], i + 1);
    }
}

TEST(PcmAudioFrameTests, copyConstrutor_shouldCopy)
{
    PcmAudioFrame frame(PcmAudioFrame::Format::Signed24, 2, 3);
    for (size_t i = 0; i < 6; i++)
    {
        frame[i] = i + 1;
    }

    PcmAudioFrame copy(frame);

    EXPECT_EQ(copy.format(), PcmAudioFrame::Format::Signed24);
    EXPECT_EQ(copy.channelCount(), 2);
    EXPECT_EQ(copy.sampleCount(), 3);
    EXPECT_EQ(copy.size(), 18);
    EXPECT_NE(frame.data(), copy.data());

    for (size_t i = 0; i < 6; i++)
    {
        EXPECT_EQ(copy.data()[i], i + 1);
    }
}

TEST(PcmAudioFrameTests, moveConstructor_shouldMove)
{
    PcmAudioFrame frame(PcmAudioFrame::Format::Signed24, 2, 3);
    for (size_t i = 0; i < 6; i++)
    {
        frame[i] = i + 1;
    }

    const uint8_t* data = frame.data();
    PcmAudioFrame movedFrame(move(frame));

    EXPECT_EQ(movedFrame.format(), PcmAudioFrame::Format::Signed24);
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

TEST(PcmAudioFrameTests, assignationOperator_shouldCopy)
{
    PcmAudioFrame frame(PcmAudioFrame::Format::Signed24, 2, 3);
    for (size_t i = 0; i < 6; i++)
    {
        frame[i] = i + 1;
    }

    PcmAudioFrame copy(PcmAudioFrame::Format::Signed16, 1, 1);
    copy = frame;

    EXPECT_EQ(copy.format(), PcmAudioFrame::Format::Signed24);
    EXPECT_EQ(copy.channelCount(), 2);
    EXPECT_EQ(copy.sampleCount(), 3);
    EXPECT_EQ(copy.size(), 18);
    EXPECT_NE(frame.data(), copy.data());

    for (size_t i = 0; i < 6; i++)
    {
        EXPECT_EQ(copy.data()[i], i + 1);
    }
}

TEST(PcmAudioFrameTests, moveAssignationOperator_shouldCopy)
{
    PcmAudioFrame frame(PcmAudioFrame::Format::Unsigned24, 2, 3);
    for (size_t i = 0; i < 6; i++)
    {
        frame[i] = i + 1;
    }

    const uint8_t* data = frame.data();
    PcmAudioFrame movedFrame(PcmAudioFrame::Format::Signed16, 1, 1);
    movedFrame = move(frame);

    EXPECT_EQ(movedFrame.format(), PcmAudioFrame::Format::Unsigned24);
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

TEST(PcmAudioFrameTests, extractionOperator_shouldExtractDataFromTheStream)
{
    stringstream ss;
    ss << "ab";

    PcmAudioFrame frame(PcmAudioFrame::Format::Unsigned8, 2, 1);

    ss >> frame;

    EXPECT_EQ(frame.data()[0], 'a');
    EXPECT_EQ(frame.data()[1], 'b');
}

TEST(PcmAudioFrameTests, insertionOperator_shouldInsertDataIntoTheStream)
{
    stringstream ss;
    PcmAudioFrame frame(PcmAudioFrame::Format::Unsigned8, 2, 1);
    frame[0] = 'a';
    frame[1] = 'b';

    ss << frame;

    EXPECT_EQ(ss.str(), "ab");
}
