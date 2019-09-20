#include <Uniformization/Communication/Messages/Tcp/FftResponseMessage.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

TEST(FftResponseMessageTests, constructor_shouldSetTheAttributes)
{
    constexpr uint8_t FftId = 6;
    constexpr size_t FftValueCount = 3;
    constexpr complex<float> FftValues[FftValueCount] = { complex<float>(1, 2), complex<float>(2, 3), complex<float>(3, 4) };
    FftResponseMessage message(FftId, FftValues, FftValueCount);

    EXPECT_EQ(message.id(), 8);
    EXPECT_EQ(message.fullSize(), 34);

    EXPECT_EQ(message.fftId(), FftId);
    EXPECT_EQ(message.fftValues(), FftValues);
    EXPECT_EQ(message.fftValueCount(), FftValueCount);
}

TEST(FftResponseMessageTests, toBuffer_shouldSerializeTheMessage)
{
    constexpr uint8_t FftId = 6;
    constexpr size_t FftValueCount = 3;
    constexpr complex<float> FftValues[FftValueCount] = { complex<float>(1, 2), complex<float>(2, 3), complex<float>(3, 4) };
    FftResponseMessage message(FftId, FftValues, FftValueCount);
    NetworkBuffer buffer(100);

    message.toBuffer(buffer);

    EXPECT_EQ(message.id(), 8);
    EXPECT_EQ(message.fullSize(), 34);

    EXPECT_EQ(buffer.data()[8], 0);
    EXPECT_EQ(buffer.data()[9], 6);
    EXPECT_EQ(*reinterpret_cast<complex<float>*>(buffer.data() + 10), FftValues[0]);
    EXPECT_EQ(*reinterpret_cast<complex<float>*>(buffer.data() + 18), FftValues[1]);
    EXPECT_EQ(*reinterpret_cast<complex<float>*>(buffer.data() + 26), FftValues[2]);
}
