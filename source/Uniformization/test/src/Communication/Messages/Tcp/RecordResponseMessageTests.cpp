#include <Uniformization/Communication/Messages/Tcp/RecordResponseMessage.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

TEST(RecordResponseMessageTests, constructor_shouldSetTheAttributes)
{
    constexpr uint8_t RecordId = 6;
    constexpr size_t DataSize = 3;
    constexpr uint8_t Data[DataSize] = { 0, 1, 2 };
    RecordResponseMessage message(RecordId, Data, DataSize);

    EXPECT_EQ(message.id(), 6);
    EXPECT_EQ(message.fullSize(), 12);

    EXPECT_EQ(message.recordId(), RecordId);
    EXPECT_EQ(message.data(), Data);
    EXPECT_EQ(message.dataSize(), DataSize);
}

TEST(RecordResponseMessageTests, toBuffer_shouldSerializeTheMessage)
{
    constexpr uint8_t RecordId = 6;
    constexpr size_t DataSize = 3;
    constexpr uint8_t Data[DataSize] = { 0, 1, 2 };
    RecordResponseMessage message(RecordId, Data, DataSize);
    NetworkBuffer buffer(100);

    message.toBuffer(buffer);

    EXPECT_EQ(message.id(), 6);
    EXPECT_EQ(message.fullSize(), 12);

    EXPECT_EQ(buffer.data()[8], RecordId);
    EXPECT_EQ(buffer.data()[9], Data[0]);
    EXPECT_EQ(buffer.data()[10], Data[1]);
    EXPECT_EQ(buffer.data()[11], Data[2]);
}
