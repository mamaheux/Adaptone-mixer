#include <Uniformization/Communication/Messages/Tcp/ProbeInitializationResponseMessage.h>

using namespace adaptone;

constexpr uint32_t ProbeInitializationResponseMessage::Id;

ProbeInitializationResponseMessage::ProbeInitializationResponseMessage(bool isCompatible, bool isMaster) :
    PayloadMessage(Id, 2),
    m_isCompatible(isCompatible),
    m_isMaster(isMaster)
{
}

ProbeInitializationResponseMessage::~ProbeInitializationResponseMessage()
{
}

void ProbeInitializationResponseMessage::serializePayload(NetworkBufferView& buffer)
{
    buffer.data()[0] = static_cast<uint8_t>(m_isCompatible);
    buffer.data()[1] = static_cast<uint8_t>(m_isMaster);
}
