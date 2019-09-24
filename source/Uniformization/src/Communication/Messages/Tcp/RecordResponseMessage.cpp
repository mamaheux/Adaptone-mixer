#include <Uniformization/Communication/Messages/Tcp/RecordResponseMessage.h>

using namespace adaptone;
using namespace std;

constexpr uint32_t RecordResponseMessage::Id;
constexpr size_t RecordResponseMessage::MinimumMessageSize;

static constexpr size_t RecordIdFromBufferOffset = 8;

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
    verifyId(buffer, Id);
    verifyMessageSizeAtLeast(messageSize, MinimumMessageSize);

    return RecordResponseMessage(buffer.data()[RecordIdFromBufferOffset],
        buffer.data() + MinimumMessageSize,
        messageSize - MinimumMessageSize);
}

void RecordResponseMessage::serializePayload(NetworkBufferView buffer) const
{
    *buffer.data() = m_recordId;
    memcpy(buffer.data() + 1, m_data, m_dataSize);
}
