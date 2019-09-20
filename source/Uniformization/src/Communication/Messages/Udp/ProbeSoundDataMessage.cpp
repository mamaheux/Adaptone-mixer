#include <Uniformization/Communication/Messages/Udp/ProbeSoundDataMessage.h>

using namespace adaptone;
using namespace std;

constexpr uint32_t ProbeSoundDataMessage::Id;

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
    constexpr size_t MinimumSize = 17;
    verifyId(buffer, Id);
    verifyMessageSizeAtLeast(messageSize, MinimumSize);

    return ProbeSoundDataMessage(boost::endian::big_to_native(*reinterpret_cast<uint16_t*>(buffer.data() + 8)),
        buffer.data()[10],
        buffer.data()[11],
        buffer.data()[12],
        boost::endian::big_to_native(*reinterpret_cast<uint16_t*>(buffer.data() + 13)),
        boost::endian::big_to_native(*reinterpret_cast<uint16_t*>(buffer.data() + 15)),
        buffer.data() + MinimumSize,
        messageSize - MinimumSize);
}

void ProbeSoundDataMessage::serializePayload(NetworkBufferView buffer)
{
    *reinterpret_cast<uint16_t*>(buffer.data()) = boost::endian::native_to_big(m_soundDataId);
    buffer.data()[2] = m_hours;
    buffer.data()[3] = m_minutes;
    buffer.data()[4] = m_seconds;
    *reinterpret_cast<uint16_t*>(buffer.data() + 5) = boost::endian::native_to_big(m_milliseconds);
    *reinterpret_cast<uint16_t*>(buffer.data() + 7) = boost::endian::native_to_big(m_microseconds);

    memcpy(buffer.data() + 9, m_data, m_dataSize);
}
