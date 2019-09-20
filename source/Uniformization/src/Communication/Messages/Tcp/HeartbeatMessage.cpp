#include <Uniformization/Communication/Messages/Tcp/HeartbeatMessage.h>

using namespace adaptone;

constexpr uint32_t HeartbeatMessage::Id;

HeartbeatMessage::HeartbeatMessage() : ProbeMessage(Id, 0)
{
}

HeartbeatMessage::~HeartbeatMessage()
{
}

void HeartbeatMessage::serialize(NetworkBufferView buffer)
{
}
