#include <Uniformization/Communication/Messages/Tcp/ProbeInitializationResponseMessage.h>

using namespace adaptone;

constexpr uint32_t ProbeInitializationResponseMessage::Id;
constexpr size_t ProbeInitializationResponseMessage::MessageSize;

static constexpr size_t IsCompatibleFromBufferOffset = 8;
static constexpr size_t IsMasterFromBufferOffset = 9;
static constexpr size_t ProbeIdFromBufferOffset = 10;

static constexpr size_t IsCompatiblePayloadOffset = 0;
static constexpr size_t IsMasterPayloadOffset = 1;
static constexpr size_t ProbeIdPayloadOffset = 2;

ProbeInitializationResponseMessage::ProbeInitializationResponseMessage(bool isCompatible,
    bool isMaster,
    uint32_t probeId) :
    PayloadMessage(Id, 6),
    m_isCompatible(isCompatible),
    m_isMaster(isMaster),
    m_probeId(probeId)
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

    return ProbeInitializationResponseMessage(static_cast<bool>(buffer.data()[IsCompatibleFromBufferOffset]),
        static_cast<bool>(buffer.data()[IsMasterFromBufferOffset]),
        boost::endian::big_to_native(*reinterpret_cast<uint32_t*>(buffer.data() + ProbeIdFromBufferOffset)));
}

void ProbeInitializationResponseMessage::serializePayload(NetworkBufferView buffer) const
{
    buffer.data()[IsCompatiblePayloadOffset] = static_cast<uint8_t>(m_isCompatible);
    buffer.data()[IsMasterPayloadOffset] = static_cast<uint8_t>(m_isMaster);
    *reinterpret_cast<uint32_t*>(buffer.data() + ProbeIdPayloadOffset) = boost::endian::native_to_big(m_probeId);
}
