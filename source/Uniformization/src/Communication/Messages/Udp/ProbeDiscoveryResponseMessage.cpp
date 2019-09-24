#include <Uniformization/Communication/Messages/Udp/ProbeDiscoveryResponseMessage.h>

using namespace adaptone;

constexpr uint32_t ProbeDiscoveryResponseMessage::Id;
constexpr size_t ProbeDiscoveryResponseMessage::MessageSize;

ProbeDiscoveryResponseMessage::ProbeDiscoveryResponseMessage() : ProbeMessage(Id, 0)
{
}

ProbeDiscoveryResponseMessage::~ProbeDiscoveryResponseMessage()
{
}

ProbeDiscoveryResponseMessage ProbeDiscoveryResponseMessage::fromBuffer(NetworkBufferView buffer, size_t messageSize)
{
    verifyId(buffer, Id);
    verifyMessageSize(messageSize, MessageSize);

    return ProbeDiscoveryResponseMessage();
}

void ProbeDiscoveryResponseMessage::serialize(NetworkBufferView buffer) const
{
}
