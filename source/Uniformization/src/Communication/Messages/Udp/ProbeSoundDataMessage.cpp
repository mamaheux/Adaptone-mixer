#include <Uniformization/Communication/Messages/Udp/ProbeSoundDataMessage.h>

using namespace adaptone;
using namespace std;

constexpr uint32_t ProbeSoundDataMessage::Id;
constexpr size_t ProbeSoundDataMessage::MinimumMessageSize;

static constexpr size_t SoundDataIdFromBufferOffset = 8;
static constexpr size_t HoursFromBufferOffset = 10;
static constexpr size_t MinutesFromBufferOffset = 11;
static constexpr size_t SecondsFromBufferOffset = 12;
static constexpr size_t MillisecondsFromBufferOffset = 13;
static constexpr size_t MicrosecondsFromBufferOffset = 15;

static constexpr size_t HoursSerializePayloadOffset = 2;
static constexpr size_t MinutesSerializePayloadOffset = 3;
static constexpr size_t SecondsSerializePayloadOffset = 4;
static constexpr size_t MillisecondsSerializePayloadOffset = 5;
static constexpr size_t MicrosecondsSerializePayloadOffset = 7;
static constexpr size_t DataSerializePayloadOffset = 9;

ProbeSoundDataMessage::ProbeSoundDataMessage(uint16_t soundDataId, uint8_t hours, uint8_t minutes, uint8_t seconds,
    uint16_t milliseconds, uint16_t microseconds, const uint8_t* data, size_t dataSize) :
    PayloadMessage(Id, 9 + dataSize),
    m_soundDataId(soundDataId),
    m_hours(hours),
    m_minutes(minutes),
    m_seconds(seconds),
    m_milliseconds(milliseconds),
    m_microseconds(microseconds),
    m_data(data),
    m_dataSize(dataSize)
{
}

ProbeSoundDataMessage::~ProbeSoundDataMessage()
{
}

ProbeSoundDataMessage ProbeSoundDataMessage::fromBuffer(NetworkBufferView buffer, size_t messageSize)
{
    verifyId(buffer, Id);
    verifyMessageSizeAtLeast(messageSize, MinimumMessageSize);

    return ProbeSoundDataMessage(
        boost::endian::big_to_native(*reinterpret_cast<uint16_t*>(buffer.data() + SoundDataIdFromBufferOffset)),
        buffer.data()[HoursFromBufferOffset],
        buffer.data()[MinutesFromBufferOffset],
        buffer.data()[SecondsFromBufferOffset],
        boost::endian::big_to_native(*reinterpret_cast<uint16_t*>(buffer.data() + MillisecondsFromBufferOffset)),
        boost::endian::big_to_native(*reinterpret_cast<uint16_t*>(buffer.data() + MicrosecondsFromBufferOffset)),
        buffer.data() + MinimumMessageSize,
        messageSize - MinimumMessageSize);
}

void ProbeSoundDataMessage::serializePayload(NetworkBufferView buffer) const
{
    *reinterpret_cast<uint16_t*>(buffer.data()) = boost::endian::native_to_big(m_soundDataId);
    buffer.data()[HoursSerializePayloadOffset] = m_hours;
    buffer.data()[MinutesSerializePayloadOffset] = m_minutes;
    buffer.data()[SecondsSerializePayloadOffset] = m_seconds;
    *reinterpret_cast<uint16_t*>(buffer.data() + MillisecondsSerializePayloadOffset) =
        boost::endian::native_to_big(m_milliseconds);
    *reinterpret_cast<uint16_t*>(buffer.data() + MicrosecondsSerializePayloadOffset) =
        boost::endian::native_to_big(m_microseconds);

    memcpy(buffer.data() + DataSerializePayloadOffset, m_data, m_dataSize);
}
