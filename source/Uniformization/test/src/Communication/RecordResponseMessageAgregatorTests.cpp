#include <Uniformization/Communication/RecordResponseMessageAgregator.h>

#include <gtest/gtest.h>

#include <chrono>
#include <thread>

using namespace adaptone;
using namespace std;

TEST(RecordResponseMessageAgregatorTests, read_shouldReturnAudioFrames)
{
    constexpr size_t DataSize = 1;

    constexpr int TimeoutMs = 1;
    constexpr uint8_t RecordId = 0;
    constexpr uint8_t ProbeCount = 2;

    constexpr uint8_t Data0[DataSize] = { 0 };
    constexpr uint8_t Data1[DataSize] = { 127 };
    constexpr uint8_t Data2[DataSize] = { 255 };

    const uint32_t ProbeId0 = 10;
    const uint32_t ProbeId1 = 11;
    const uint32_t ProbeId2 = 12;

    RecordResponseMessageAgregator agregator(PcmAudioFrameFormat::Unsigned8);
    agregator.reset(RecordId, ProbeCount);

    agregator.agregate(RecordResponseMessage(RecordId, Data0, DataSize), ProbeId0);
    agregator.agregate(RecordResponseMessage(RecordId + 1, Data1, DataSize), ProbeId1);
    agregator.agregate(RecordResponseMessage(RecordId, Data2, DataSize), ProbeId2);

    auto result = agregator.read(TimeoutMs);

    ASSERT_EQ(result->size(), ProbeCount);
    EXPECT_EQ(result->at(ProbeId0)[0], -1);
    EXPECT_EQ(result->at(ProbeId2)[0], 1);
}

TEST(RecordResponseMessageAgregatorTests, read_multithreading_shouldReturnAudioFrames)
{
    constexpr size_t DataSize = 1;

    constexpr int TimeoutMs = 100;
    constexpr uint8_t RecordId = 0;
    constexpr uint8_t ProbeCount = 2;

    constexpr uint8_t Data0[DataSize] = { 127 };
    constexpr uint8_t Data1[DataSize] = { 0 };
    constexpr uint8_t Data2[DataSize] = { 255 };

    const uint32_t ProbeId0 = 10;
    const uint32_t ProbeId1 = 11;
    const uint32_t ProbeId2 = 12;

    RecordResponseMessageAgregator agregator(PcmAudioFrameFormat::Unsigned8);

    agregator.agregate(RecordResponseMessage(RecordId, Data0, DataSize), ProbeId0);
    agregator.reset(RecordId, ProbeCount);

    thread thread([&]()
    {
        this_thread::sleep_for(chrono::milliseconds(TimeoutMs / 10));
        agregator.agregate(RecordResponseMessage(RecordId, Data1, DataSize), ProbeId1);
        agregator.agregate(RecordResponseMessage(RecordId, Data2, DataSize), ProbeId2);
    });

    auto result = agregator.read(TimeoutMs);
    thread.join();

    ASSERT_EQ(result->size(), ProbeCount);
    EXPECT_EQ(result->at(ProbeId1)[0], -1);
    EXPECT_EQ(result->at(ProbeId2)[0], 1);
}

TEST(RecordResponseMessageAgregatorTests, read_timeout_shouldReturnNullopt)
{
    constexpr double MaxAbsElapsedMsTimeError = 1;
    constexpr int TimeoutMs = 10;
    constexpr uint8_t RecordId = 0;
    constexpr uint8_t ProbeCount = 2;

    RecordResponseMessageAgregator agregator(PcmAudioFrameFormat::Unsigned8);
    agregator.reset(RecordId, ProbeCount);

    auto start = chrono::system_clock::now();
    auto result = agregator.read(TimeoutMs);
    auto end = chrono::system_clock::now();
    chrono::duration<double> elapsedSeconds = end - start;

    EXPECT_EQ(result, nullopt);
    EXPECT_NEAR(elapsedSeconds.count() * 1000, TimeoutMs, MaxAbsElapsedMsTimeError);
}
