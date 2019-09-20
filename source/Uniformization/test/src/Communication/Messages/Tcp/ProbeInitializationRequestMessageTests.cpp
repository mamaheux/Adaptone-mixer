#include <Uniformization/Communication/Messages/Tcp/ProbeInitializationRequestMessage.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

TEST(ProbeInitializationRequestMessageTests, constructor_shouldSetTheAttributes)
{
    constexpr uint32_t SampleFrequency = 44100;
    constexpr PcmAudioFrame::Format Format = PcmAudioFrame::Format::Float;
    ProbeInitializationRequestMessage message(44100, Format);

    EXPECT_EQ(message.id(), 2);
    EXPECT_EQ(message.fullSize(), 16);

    EXPECT_EQ(message.sampleFrequency(), SampleFrequency);
    EXPECT_EQ(message.format(), Format);
}

TEST(ProbeInitializationRequestMessageTests, toBuffer_signed8_shouldSerializeTheMessage)
{
    constexpr uint32_t SampleFrequency = 44100;
    constexpr PcmAudioFrame::Format Format = PcmAudioFrame::Format::Signed8;
    ProbeInitializationRequestMessage message(44100, Format);
    NetworkBuffer buffer(100);

    message.toBuffer(buffer);

    EXPECT_EQ(message.id(), 2);
    EXPECT_EQ(message.fullSize(), 16);

    EXPECT_EQ(buffer.data()[8], 0);
    EXPECT_EQ(buffer.data()[9], 0);
    EXPECT_EQ(buffer.data()[10], 0xAC);
    EXPECT_EQ(buffer.data()[11], 0x44);

    EXPECT_EQ(buffer.data()[12], 0);
    EXPECT_EQ(buffer.data()[13], 0);
    EXPECT_EQ(buffer.data()[14], 0);
    EXPECT_EQ(buffer.data()[15], 0);
}

TEST(ProbeInitializationRequestMessageTests, toBuffer_signed16_shouldSerializeTheMessage)
{
    constexpr uint32_t SampleFrequency = 44100;
    constexpr PcmAudioFrame::Format Format = PcmAudioFrame::Format::Signed16;
    ProbeInitializationRequestMessage message(44100, Format);
    NetworkBuffer buffer(100);

    message.toBuffer(buffer);

    EXPECT_EQ(message.id(), 2);
    EXPECT_EQ(message.fullSize(), 16);

    EXPECT_EQ(buffer.data()[8], 0);
    EXPECT_EQ(buffer.data()[9], 0);
    EXPECT_EQ(buffer.data()[10], 0xAC);
    EXPECT_EQ(buffer.data()[11], 0x44);

    EXPECT_EQ(buffer.data()[12], 0);
    EXPECT_EQ(buffer.data()[13], 0);
    EXPECT_EQ(buffer.data()[14], 0);
    EXPECT_EQ(buffer.data()[15], 1);
}

TEST(ProbeInitializationRequestMessageTests, toBuffer_signed24_shouldSerializeTheMessage)
{
    constexpr uint32_t SampleFrequency = 44100;
    constexpr PcmAudioFrame::Format Format = PcmAudioFrame::Format::Signed24;
    ProbeInitializationRequestMessage message(44100, Format);
    NetworkBuffer buffer(100);

    message.toBuffer(buffer);

    EXPECT_EQ(message.id(), 2);
    EXPECT_EQ(message.fullSize(), 16);

    EXPECT_EQ(buffer.data()[8], 0);
    EXPECT_EQ(buffer.data()[9], 0);
    EXPECT_EQ(buffer.data()[10], 0xAC);
    EXPECT_EQ(buffer.data()[11], 0x44);

    EXPECT_EQ(buffer.data()[12], 0);
    EXPECT_EQ(buffer.data()[13], 0);
    EXPECT_EQ(buffer.data()[14], 0);
    EXPECT_EQ(buffer.data()[15], 2);
}

TEST(ProbeInitializationRequestMessageTests, toBuffer_signedPadded24_shouldSerializeTheMessage)
{
    constexpr uint32_t SampleFrequency = 44100;
    constexpr PcmAudioFrame::Format Format = PcmAudioFrame::Format::SignedPadded24;
    ProbeInitializationRequestMessage message(44100, Format);
    NetworkBuffer buffer(100);

    message.toBuffer(buffer);

    EXPECT_EQ(message.id(), 2);
    EXPECT_EQ(message.fullSize(), 16);

    EXPECT_EQ(buffer.data()[8], 0);
    EXPECT_EQ(buffer.data()[9], 0);
    EXPECT_EQ(buffer.data()[10], 0xAC);
    EXPECT_EQ(buffer.data()[11], 0x44);

    EXPECT_EQ(buffer.data()[12], 0);
    EXPECT_EQ(buffer.data()[13], 0);
    EXPECT_EQ(buffer.data()[14], 0);
    EXPECT_EQ(buffer.data()[15], 3);
}

TEST(ProbeInitializationRequestMessageTests, toBuffer_signed32_shouldSerializeTheMessage)
{
    constexpr uint32_t SampleFrequency = 44100;
    constexpr PcmAudioFrame::Format Format = PcmAudioFrame::Format::Signed32;
    ProbeInitializationRequestMessage message(44100, Format);
    NetworkBuffer buffer(100);

    message.toBuffer(buffer);

    EXPECT_EQ(message.id(), 2);
    EXPECT_EQ(message.fullSize(), 16);

    EXPECT_EQ(buffer.data()[8], 0);
    EXPECT_EQ(buffer.data()[9], 0);
    EXPECT_EQ(buffer.data()[10], 0xAC);
    EXPECT_EQ(buffer.data()[11], 0x44);

    EXPECT_EQ(buffer.data()[12], 0);
    EXPECT_EQ(buffer.data()[13], 0);
    EXPECT_EQ(buffer.data()[14], 0);
    EXPECT_EQ(buffer.data()[15], 4);
}

TEST(ProbeInitializationRequestMessageTests, toBuffer_unsigned8_shouldSerializeTheMessage)
{
    constexpr uint32_t SampleFrequency = 44100;
    constexpr PcmAudioFrame::Format Format = PcmAudioFrame::Format::Unsigned8;
    ProbeInitializationRequestMessage message(44100, Format);
    NetworkBuffer buffer(100);

    message.toBuffer(buffer);

    EXPECT_EQ(message.id(), 2);
    EXPECT_EQ(message.fullSize(), 16);

    EXPECT_EQ(buffer.data()[8], 0);
    EXPECT_EQ(buffer.data()[9], 0);
    EXPECT_EQ(buffer.data()[10], 0xAC);
    EXPECT_EQ(buffer.data()[11], 0x44);

    EXPECT_EQ(buffer.data()[12], 0);
    EXPECT_EQ(buffer.data()[13], 0);
    EXPECT_EQ(buffer.data()[14], 0);
    EXPECT_EQ(buffer.data()[15], 5);
}

TEST(ProbeInitializationRequestMessageTests, toBuffer_unsigned16_shouldSerializeTheMessage)
{
    constexpr uint32_t SampleFrequency = 44100;
    constexpr PcmAudioFrame::Format Format = PcmAudioFrame::Format::Unsigned16;
    ProbeInitializationRequestMessage message(44100, Format);
    NetworkBuffer buffer(100);

    message.toBuffer(buffer);

    EXPECT_EQ(message.id(), 2);
    EXPECT_EQ(message.fullSize(), 16);

    EXPECT_EQ(buffer.data()[8], 0);
    EXPECT_EQ(buffer.data()[9], 0);
    EXPECT_EQ(buffer.data()[10], 0xAC);
    EXPECT_EQ(buffer.data()[11], 0x44);

    EXPECT_EQ(buffer.data()[12], 0);
    EXPECT_EQ(buffer.data()[13], 0);
    EXPECT_EQ(buffer.data()[14], 0);
    EXPECT_EQ(buffer.data()[15], 6);
}

TEST(ProbeInitializationRequestMessageTests, toBuffer_unsigned24_shouldSerializeTheMessage)
{
    constexpr uint32_t SampleFrequency = 44100;
    constexpr PcmAudioFrame::Format Format = PcmAudioFrame::Format::Unsigned24;
    ProbeInitializationRequestMessage message(44100, Format);
    NetworkBuffer buffer(100);

    message.toBuffer(buffer);

    EXPECT_EQ(message.id(), 2);
    EXPECT_EQ(message.fullSize(), 16);

    EXPECT_EQ(buffer.data()[8], 0);
    EXPECT_EQ(buffer.data()[9], 0);
    EXPECT_EQ(buffer.data()[10], 0xAC);
    EXPECT_EQ(buffer.data()[11], 0x44);

    EXPECT_EQ(buffer.data()[12], 0);
    EXPECT_EQ(buffer.data()[13], 0);
    EXPECT_EQ(buffer.data()[14], 0);
    EXPECT_EQ(buffer.data()[15], 7);
}

TEST(ProbeInitializationRequestMessageTests, toBuffer_unsignedPadded24_shouldSerializeTheMessage)
{
    constexpr uint32_t SampleFrequency = 44100;
    constexpr PcmAudioFrame::Format Format = PcmAudioFrame::Format::UnsignedPadded24;
    ProbeInitializationRequestMessage message(44100, Format);
    NetworkBuffer buffer(100);

    message.toBuffer(buffer);

    EXPECT_EQ(message.id(), 2);
    EXPECT_EQ(message.fullSize(), 16);

    EXPECT_EQ(buffer.data()[8], 0);
    EXPECT_EQ(buffer.data()[9], 0);
    EXPECT_EQ(buffer.data()[10], 0xAC);
    EXPECT_EQ(buffer.data()[11], 0x44);

    EXPECT_EQ(buffer.data()[12], 0);
    EXPECT_EQ(buffer.data()[13], 0);
    EXPECT_EQ(buffer.data()[14], 0);
    EXPECT_EQ(buffer.data()[15], 8);
}

TEST(ProbeInitializationRequestMessageTests, toBuffer_unsigned32_shouldSerializeTheMessage)
{
    constexpr uint32_t SampleFrequency = 44100;
    constexpr PcmAudioFrame::Format Format = PcmAudioFrame::Format::Unsigned32;
    ProbeInitializationRequestMessage message(44100, Format);
    NetworkBuffer buffer(100);

    message.toBuffer(buffer);

    EXPECT_EQ(message.id(), 2);
    EXPECT_EQ(message.fullSize(), 16);

    EXPECT_EQ(buffer.data()[8], 0);
    EXPECT_EQ(buffer.data()[9], 0);
    EXPECT_EQ(buffer.data()[10], 0xAC);
    EXPECT_EQ(buffer.data()[11], 0x44);

    EXPECT_EQ(buffer.data()[12], 0);
    EXPECT_EQ(buffer.data()[13], 0);
    EXPECT_EQ(buffer.data()[14], 0);
    EXPECT_EQ(buffer.data()[15], 9);
}

TEST(ProbeInitializationRequestMessageTests, toBuffer_float_shouldSerializeTheMessage)
{
    constexpr uint32_t SampleFrequency = 44100;
    constexpr PcmAudioFrame::Format Format = PcmAudioFrame::Format::Float;
    ProbeInitializationRequestMessage message(44100, Format);
    NetworkBuffer buffer(100);

    message.toBuffer(buffer);

    EXPECT_EQ(message.id(), 2);
    EXPECT_EQ(message.fullSize(), 16);

    EXPECT_EQ(buffer.data()[8], 0);
    EXPECT_EQ(buffer.data()[9], 0);
    EXPECT_EQ(buffer.data()[10], 0xAC);
    EXPECT_EQ(buffer.data()[11], 0x44);

    EXPECT_EQ(buffer.data()[12], 0);
    EXPECT_EQ(buffer.data()[13], 0);
    EXPECT_EQ(buffer.data()[14], 0);
    EXPECT_EQ(buffer.data()[15], 10);
}

TEST(ProbeInitializationRequestMessageTests, toBuffer_double_shouldSerializeTheMessage)
{
    constexpr uint32_t SampleFrequency = 44100;
    constexpr PcmAudioFrame::Format Format = PcmAudioFrame::Format::Double;
    ProbeInitializationRequestMessage message(44100, Format);
    NetworkBuffer buffer(100);

    message.toBuffer(buffer);

    EXPECT_EQ(message.id(), 2);
    EXPECT_EQ(message.fullSize(), 16);

    EXPECT_EQ(buffer.data()[8], 0);
    EXPECT_EQ(buffer.data()[9], 0);
    EXPECT_EQ(buffer.data()[10], 0xAC);
    EXPECT_EQ(buffer.data()[11], 0x44);

    EXPECT_EQ(buffer.data()[12], 0);
    EXPECT_EQ(buffer.data()[13], 0);
    EXPECT_EQ(buffer.data()[14], 0);
    EXPECT_EQ(buffer.data()[15], 11);
}
