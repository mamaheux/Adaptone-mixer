#include <Uniformization/Communication/Messages/Tcp/RecordRequestMessage.h>

using namespace adaptone;

constexpr uint32_t RecordRequestMessage::Id;

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
    verifyMessageSize(messageSize, 16);

    return RecordRequestMessage(buffer.data()[8],
        buffer.data()[9],
        buffer.data()[10],
        boost::endian::big_to_native(*reinterpret_cast<uint16_t*>(buffer.data() + 11)),
        boost::endian::big_to_native(*reinterpret_cast<uint16_t*>(buffer.data() + 13)),
        buffer.data()[15]);
}

void RecordRequestMessage::serializePayload(NetworkBufferView buffer) const
{
    buffer.data()[0] = m_hours;
    buffer.data()[1] = m_minutes;
    buffer.data()[2] = m_seconds;

    *reinterpret_cast<uint16_t*>(buffer.data() + 3) = boost::endian::native_to_big(m_milliseconds);
    *reinterpret_cast<uint16_t*>(buffer.data() + 5) = boost::endian::native_to_big(m_duration);

    buffer.data()[7] = m_recordId;
}
