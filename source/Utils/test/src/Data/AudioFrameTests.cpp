#include <Utils/Data/AudioFrame.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

TEST(AudioFrameTests, construtor_shouldSetParameterAndAllocateMemory)
{
    AudioFrame<int> frame(2, 3);
    for (size_t i = 0; i < 6; i++)
    {
        frame[i] = i + 1;
    }

    EXPECT_EQ(frame.channelCount(), 2);
    EXPECT_EQ(frame.sampleCount(), 3);
    EXPECT_EQ(frame.size(), 6);
    EXPECT_EQ(frame.byteSize(), 24);

    for (size_t i = 0; i < 6; i++)
    {
        EXPECT_EQ(frame.data()[i], i + 1);
    }
}

TEST(AudioFrameTests, copyConstrutor_shouldCopy)
{
    AudioFrame<int> frame(2, 3);
    for (size_t i = 0; i < 6; i++)
    {
        frame[i] = i + 1;
    }

    AudioFrame<int> copy(frame);

    EXPECT_EQ(copy.channelCount(), 2);
    EXPECT_EQ(copy.sampleCount(), 3);
    EXPECT_EQ(copy.size(), 6);
    EXPECT_NE(frame.data(), copy.data());

    for (size_t i = 0; i < 6; i++)
    {
        EXPECT_EQ(copy.data()[i], i + 1);
    }
}

TEST(AudioFrameTests, moveConstructor_shouldMove)
{
    AudioFrame<int> frame(2, 3);
    for (size_t i = 0; i < 6; i++)
    {
        frame[i] = i + 1;
    }

    int* data = frame.data();
    AudioFrame<int> movedFrame(move(frame));

    EXPECT_EQ(movedFrame.channelCount(), 2);
    EXPECT_EQ(movedFrame.sampleCount(), 3);
    EXPECT_EQ(movedFrame.size(), 6);
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

TEST(AudioFrameTests, assignationOperator_shouldCopy)
{
    AudioFrame<int> frame(2, 3);
    for (size_t i = 0; i < 6; i++)
    {
        frame[i] = i + 1;
    }

    AudioFrame<int> copy(1, 1);
    copy = frame;

    EXPECT_EQ(copy.channelCount(), 2);
    EXPECT_EQ(copy.sampleCount(), 3);
    EXPECT_EQ(copy.size(), 6);EXPECT_EQ(frame.byteSize(), 24);
    EXPECT_NE(frame.data(), copy.data());

    for (size_t i = 0; i < 6; i++)
    {
        EXPECT_EQ(copy.data()[i], i + 1);
    }
}


TEST(AudioFrameTests, moveAssignationOperator_shouldCopy)
{
    AudioFrame<int> frame(2, 3);
    for (size_t i = 0; i < 6; i++)
    {
        frame[i] = i + 1;
    }

    int* data = frame.data();
    AudioFrame<int> movedFrame(1, 1);
    movedFrame = move(frame);

    EXPECT_EQ(movedFrame.channelCount(), 2);
    EXPECT_EQ(movedFrame.sampleCount(), 3);
    EXPECT_EQ(movedFrame.size(), 6);
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