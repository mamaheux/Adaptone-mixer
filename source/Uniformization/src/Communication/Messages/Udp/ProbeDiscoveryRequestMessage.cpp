#include <Uniformization/Communication/Messages/Udp/ProbeDiscoveryRequestMessage.h>

using namespace adaptone;

constexpr uint32_t ProbeDiscoveryRequestMessage::Id;

ProbeDiscoveryRequestMessage::ProbeDiscoveryRequestMessage() : ProbeMessage(Id, 0)
{
}

ProbeDiscoveryRequestMessage::~ProbeDiscoveryRequestMessage()
{
}

void ProbeDiscoveryRequestMessage::serialize(NetworkBufferView& buffer)
{
}
