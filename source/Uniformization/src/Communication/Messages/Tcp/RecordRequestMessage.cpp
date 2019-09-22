#include <Uniformization/Communication/Messages/Tcp/RecordRequestMessage.h>

using namespace adaptone;

constexpr uint32_t RecordRequestMessage::Id;
constexpr size_t RecordRequestMessage::MessageSize;

static constexpr size_t HoursFromBufferOffset = 8;
static constexpr size_t MinutesFromBufferOffset = 9;
static constexpr size_t SecondsFromBufferOffset = 10;
static constexpr size_t MillisecondsFromBufferOffset = 11;
static constexpr size_t DurationFromBufferOffset = 13;
static constexpr size_t RecordIdFromBufferOffset = 15;

static constexpr size_t HoursSerializePayloadOffset = 0;
static constexpr size_t MinutesSerializePayloadOffset = 1;
static constexpr size_t SecondsSerializePayloadOffset = 2;
static constexpr size_t MillisecondsSerializePayloadOffset = 3;
static constexpr size_t DurationSerializePayloadOffset = 5;
static constexpr size_t RecordIdSerializePayloadOffset = 7;

RecordRequestMessage::RecordRequestMessage(uint8_t hours, uint8_t minutes, uint8_t seconds,
    uint16_t milliseconds, uint16_t duration, uint8_t recordId) :
    PayloadMessage(Id, 8),
    m_hours(hours),
    m_minutes(minutes),
    m_seconds(seconds),
    m_milliseconds(milliseconds),
    m_duration(duration),
    m_recordId(recordId)
{
}

RecordRequestMessage::~RecordRequestMessage()
{
}

RecordRequestMessage RecordRequestMessage::fromBuffer(NetworkBufferView buffer, size_t messageSize)
{
    verifyId(buffer, Id);
    verifyMessageSize(messageSize, MessageSize);

    return RecordRequestMessage(buffer.data()[HoursFromBufferOffset],
        buffer.data()[MinutesFromBufferOffset],
        buffer.data()[SecondsFromBufferOffset],
        boost::endian::big_to_native(*reinterpret_cast<uint16_t*>(buffer.data() + MillisecondsFromBufferOffset)),
        boost::endian::big_to_native(*reinterpret_cast<uint16_t*>(buffer.data() + DurationFromBufferOffset)),
        buffer.data()[RecordIdFromBufferOffset]);
}

void RecordRequestMessage::serializePayload(NetworkBufferView buffer) const
{
    buffer.data()[HoursSerializePayloadOffset] = m_hours;
    buffer.data()[MinutesSerializePayloadOffset] = m_minutes;
    buffer.data()[SecondsSerializePayloadOffset] = m_seconds;

    *reinterpret_cast<uint16_t*>(buffer.data() + MillisecondsSerializePayloadOffset) =
        boost::endian::native_to_big(m_milliseconds);
    *reinterpret_cast<uint16_t*>(buffer.data() + DurationSerializePayloadOffset) =
        boost::endian::native_to_big(m_duration);

    buffer.data()[RecordIdSerializePayloadOffset] = m_recordId;
}
