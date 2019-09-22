#include <Uniformization/Communication/Messages/Tcp/ProbeInitializationResponseMessage.h>

using namespace adaptone;

constexpr uint32_t ProbeInitializationResponseMessage::Id;
constexpr size_t ProbeInitializationResponseMessage::MessageSize;

ProbeInitializationResponseMessage::ProbeInitializationResponseMessage(bool isCompatible, bool isMaster) :
    PayloadMessage(Id, 2),
    m_isCompatible(isCompatible),
    m_isMaster(isMaster)
{
}

ProbeInitializationResponseMessage::~ProbeInitializationResponseMessage()
{
}

ProbeInitializationResponseMessage ProbeInitializationResponseMessage::fromBuffer(NetworkBufferView buffer,
    size_t messageSize)
{
    verifyId(buffer, Id);
    verifyMessageSize(messageSize, MessageSize);

    return ProbeInitializationResponseMessage(static_cast<bool>(buffer.data()[8]), static_cast<bool>(buffer.data()[9]));
}

void ProbeInitializationResponseMessage::serializePayload(NetworkBufferView buffer) const
{
    buffer.data()[0] = static_cast<uint8_t>(m_isCompatible);
    buffer.data()[1] = static_cast<uint8_t>(m_isMaster);
}
