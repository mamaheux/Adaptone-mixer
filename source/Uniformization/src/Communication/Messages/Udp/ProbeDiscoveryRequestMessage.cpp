#include <Uniformization/Communication/Messages/Udp/ProbeDiscoveryRequestMessage.h>

using namespace adaptone;

constexpr uint32_t ProbeDiscoveryRequestMessage::Id;
constexpr size_t ProbeDiscoveryRequestMessage::MessageSize;

ProbeDiscoveryRequestMessage::ProbeDiscoveryRequestMessage() : ProbeMessage(Id, 0)
{
}

ProbeDiscoveryRequestMessage::~ProbeDiscoveryRequestMessage()
{
}

ProbeDiscoveryRequestMessage ProbeDiscoveryRequestMessage::fromBuffer(NetworkBufferView buffer, size_t messageSize)
{
    verifyId(buffer, Id);
    verifyMessageSize(messageSize, MessageSize);

    return ProbeDiscoveryRequestMessage();
}

void ProbeDiscoveryRequestMessage::serialize(NetworkBufferView buffer) const
{
}
