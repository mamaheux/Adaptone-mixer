#include <Uniformization/Communication/Messages/Tcp/HeartbeatMessage.h>

using namespace adaptone;

constexpr uint32_t HeartbeatMessage::Id;

HeartbeatMessage::HeartbeatMessage() : ProbeMessage(Id, 0)
{
}

HeartbeatMessage::~HeartbeatMessage()
{
}

HeartbeatMessage HeartbeatMessage::fromBuffer(NetworkBufferView buffer, size_t messageSize)
{
    verifyId(buffer, Id);
    verifyMessageSize(messageSize, 4);

    return HeartbeatMessage();
}

void HeartbeatMessage::serialize(NetworkBufferView buffer) const
{
}
