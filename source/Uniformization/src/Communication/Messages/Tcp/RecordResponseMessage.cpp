#include <Uniformization/Communication/Messages/Tcp/RecordResponseMessage.h>

using namespace adaptone;
using namespace std;

constexpr uint32_t RecordResponseMessage::Id;

RecordResponseMessage::RecordResponseMessage(uint8_t recordId, const uint8_t* data, size_t dataSize) :
    PayloadMessage(Id, sizeof(m_recordId) + dataSize),
    m_recordId(recordId),
    m_data(data),
    m_dataSize(dataSize)
{
}

RecordResponseMessage::~RecordResponseMessage()
{
}

RecordResponseMessage RecordResponseMessage::fromBuffer(NetworkBufferView buffer, size_t messageSize)
{
    constexpr size_t MinimumSize = 9;
    verifyId(buffer, Id);
    verifyMessageSizeAtLeast(messageSize, MinimumSize);

    return RecordResponseMessage(buffer.data()[8],
        buffer.data() + MinimumSize,
        messageSize - MinimumSize);
}

void RecordResponseMessage::serializePayload(NetworkBufferView buffer)
{
    buffer.data()[0] = m_recordId;
    memcpy(buffer.data() + 1, m_data, m_dataSize);
}
