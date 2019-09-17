#include <Utils/Data/PcmAudioFrame.h>

#include <Utils/Exception/InvalidValueException.h>

#include <gtest/gtest.h>

#include <sstream>
#include <chrono>

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

TEST(PcmAudioFrameTests, size_shouldReturnTheFrameSize)
{
    EXPECT_EQ(PcmAudioFrame::size(PcmAudioFrame::Format::Signed16, 2, 3), 12);
}

TEST(PcmAudioFrameTests, constructor_shouldSetParameterAndAllocateMemory)
{
    PcmAudioFrame frame(PcmAudioFrame::Format::Signed24, 2, 3);
    for (size_t i = 0; i < frame.size(); i++)
    {
        frame[i] = i + 1;
    }

    EXPECT_EQ(frame.format(), PcmAudioFrame::Format::Signed24);
    EXPECT_EQ(frame.channelCount(), 2);
    EXPECT_EQ(frame.sampleCount(), 3);
    EXPECT_EQ(frame.size(), 18);

    for (size_t i = 0; i < frame.size(); i++)
    {
        EXPECT_EQ(frame.data()[i], i + 1);
    }
}

TEST(PcmAudioFrameTests, copyConstructor_shouldCopy)
{
    PcmAudioFrame frame(PcmAudioFrame::Format::Signed24, 2, 3);
    for (size_t i = 0; i < frame.size(); i++)
    {
        frame[i] = i + 1;
    }

    PcmAudioFrame copy(frame);

    EXPECT_EQ(copy.format(), PcmAudioFrame::Format::Signed24);
    EXPECT_EQ(copy.channelCount(), 2);
    EXPECT_EQ(copy.sampleCount(), 3);
    EXPECT_EQ(copy.size(), 18);
    EXPECT_NE(frame.data(), copy.data());

    for (size_t i = 0; i < copy.size(); i++)
    {
        EXPECT_EQ(copy.data()[i], i + 1);
    }
}

TEST(PcmAudioFrameTests, moveConstructor_shouldMove)
{
    PcmAudioFrame frame(PcmAudioFrame::Format::Signed24, 2, 3);
    for (size_t i = 0; i < frame.size(); i++)
    {
        frame[i] = i + 1;
    }

    const uint8_t* data = frame.data();
    const PcmAudioFrame movedFrame(move(frame));

    EXPECT_EQ(movedFrame.format(), PcmAudioFrame::Format::Signed24);
    EXPECT_EQ(movedFrame.channelCount(), 2);
    EXPECT_EQ(movedFrame.sampleCount(), 3);
    EXPECT_EQ(movedFrame.size(), 18);
    EXPECT_EQ(movedFrame.data(), data);

    for (size_t i = 0; i < movedFrame.size(); i++)
    {
        EXPECT_EQ(movedFrame.data()[i], i + 1);
        EXPECT_EQ(movedFrame[i], i + 1);
    }

    EXPECT_EQ(frame.data(), nullptr);
    EXPECT_EQ(frame.channelCount(), 0);
    EXPECT_EQ(frame.sampleCount(), 0);
    EXPECT_EQ(frame.size(), 0);
}

TEST(PcmAudioFrameTests, assignationOperator_shouldCopy)
{
    PcmAudioFrame frame(PcmAudioFrame::Format::Signed24, 2, 3);
    for (size_t i = 0; i < frame.size(); i++)
    {
        frame[i] = i + 1;
    }

    PcmAudioFrame copy(PcmAudioFrame::Format::Signed16, 200, 300);
    uint8_t* oldDataBuffer = copy.data();
    copy = frame;

    EXPECT_NE(oldDataBuffer, copy.data());
    EXPECT_EQ(copy.format(), PcmAudioFrame::Format::Signed24);
    EXPECT_EQ(copy.channelCount(), 2);
    EXPECT_EQ(copy.sampleCount(), 3);
    EXPECT_EQ(copy.size(), 18);
    EXPECT_NE(frame.data(), copy.data());

    for (size_t i = 0; i < copy.size(); i++)
    {
        EXPECT_EQ(copy.data()[i], i + 1);
    }
}

TEST(PcmAudioFrameTests, assignationOperator_sameType_shouldCopyWithoutMemoryAllocation)
{
    PcmAudioFrame frame(PcmAudioFrame::Format::Signed16, 2, 3);
    for (size_t i = 0; i < frame.size(); i++)
    {
        frame[i] = i + 1;
    }

    PcmAudioFrame copy(PcmAudioFrame::Format::Signed16, 2, 3);
    uint8_t* oldDataBuffer = copy.data();
    copy = frame;

    EXPECT_EQ(oldDataBuffer, copy.data());
    EXPECT_EQ(copy.format(), PcmAudioFrame::Format::Signed16);
    EXPECT_EQ(copy.channelCount(), 2);
    EXPECT_EQ(copy.sampleCount(), 3);
    EXPECT_EQ(copy.size(), 12);
    EXPECT_NE(frame.data(), copy.data());

    for (size_t i = 0; i < copy.size(); i++)
    {
        EXPECT_EQ(copy.data()[i], i + 1);
    }
}

TEST(PcmAudioFrameTests, moveAssignationOperator_shouldCopy)
{
    PcmAudioFrame frame(PcmAudioFrame::Format::Unsigned24, 2, 3);
    for (size_t i = 0; i < frame.size(); i++)
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

    for (size_t i = 0; i < movedFrame.size(); i++)
    {
        EXPECT_EQ(movedFrame.data()[i], i + 1);
    }

    EXPECT_EQ(frame.data(), nullptr);
    EXPECT_EQ(frame.channelCount(), 0);
    EXPECT_EQ(frame.sampleCount(), 0);
    EXPECT_EQ(frame.size(), 0);
}

TEST(PcmAudioFrameTests, clear_shouldSetAllBytesTo0)
{
    PcmAudioFrame frame(PcmAudioFrame::Format::Unsigned8, 2, 3);
    for (size_t i = 0; i < frame.size(); i++)
    {
        frame[i] = i + 1;
    }
    frame.clear();

    for (size_t i = 0; i < frame.size(); i++)
    {
        EXPECT_EQ(frame.data()[i], 0);
    }
}

TEST(PcmAudioFrameTests, writeChannel_shouldWriteTheSpecifiedChannel)
{
    PcmAudioFrame frame0(PcmAudioFrame::Format::Unsigned24, 2, 2);
    PcmAudioFrame frame1(PcmAudioFrame::Format::Unsigned24, 3, 2);
    for (size_t i = 0; i < frame0.size(); i++)
    {
        frame0[i] = i + 1;
    }

    frame1.clear();
    frame1.writeChannel(1, frame0, 0);
    frame1.writeChannel(0, frame0, 1);

    EXPECT_EQ(frame1[0], 4);
    EXPECT_EQ(frame1[1], 5);
    EXPECT_EQ(frame1[2], 6);

    EXPECT_EQ(frame1[3], 1);
    EXPECT_EQ(frame1[4], 2);
    EXPECT_EQ(frame1[5], 3);

    EXPECT_EQ(frame1[6], 0);
    EXPECT_EQ(frame1[7], 0);
    EXPECT_EQ(frame1[8], 0);

    EXPECT_EQ(frame1[9], 10);
    EXPECT_EQ(frame1[10], 11);
    EXPECT_EQ(frame1[11], 12);

    EXPECT_EQ(frame1[12], 7);
    EXPECT_EQ(frame1[13], 8);
    EXPECT_EQ(frame1[14], 9);

    EXPECT_EQ(frame1[15], 0);
    EXPECT_EQ(frame1[16], 0);
    EXPECT_EQ(frame1[17], 0);
}

TEST(PcmAudioFrameTests, writeChannel_performance)
{
    PcmAudioFrame frame0(PcmAudioFrame::Format::Unsigned32, 1, 256);
    PcmAudioFrame frame1(PcmAudioFrame::Format::Unsigned32, 16, 256);
    for (size_t i = 0; i < frame0.size(); i++)
    {
        frame0[i] = i + 1;
    }

    constexpr size_t Count = 10000;

    double minElapsedTimeSeconds = 1;
    double maxElapsedTimeSeconds = 0;
    double totalElapsedTimeSeconds = 0;

    for (size_t i = 0; i < Count; i++)
    {
        auto start = chrono::system_clock::now();

        frame1.clear();
        frame1.writeChannel(0, frame0, i % frame1.channelCount());

        auto end = chrono::system_clock::now();
        chrono::duration<double> elapsedSeconds = end - start;

        totalElapsedTimeSeconds += elapsedSeconds.count();
        minElapsedTimeSeconds = elapsedSeconds.count() < minElapsedTimeSeconds ? elapsedSeconds.count() : minElapsedTimeSeconds;
        maxElapsedTimeSeconds = elapsedSeconds.count() > maxElapsedTimeSeconds ? elapsedSeconds.count() : maxElapsedTimeSeconds;
    }

    cout << "Elapsed time (avg) = " << totalElapsedTimeSeconds / Count << " s" << endl;
    cout << "Elapsed time (min) = " << minElapsedTimeSeconds << " s" << endl;
    cout << "Elapsed time (max) = " << maxElapsedTimeSeconds << " s" << endl;
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
