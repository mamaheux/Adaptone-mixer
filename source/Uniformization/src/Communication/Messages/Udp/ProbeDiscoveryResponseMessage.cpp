#include <Uniformization/Communication/Messages/Udp/ProbeDiscoveryResponseMessage.h>

using namespace adaptone;

constexpr uint32_t ProbeDiscoveryResponseMessage::Id;

ProbeDiscoveryResponseMessage::ProbeDiscoveryResponseMessage() : ProbeMessage(Id, 0)
{
}

ProbeDiscoveryResponseMessage::~ProbeDiscoveryResponseMessage()
{
}

void ProbeDiscoveryResponseMessage::serialize(NetworkBufferView buffer)
{
}
