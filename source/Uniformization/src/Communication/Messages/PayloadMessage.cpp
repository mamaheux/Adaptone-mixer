#include <Uniformization/Communication/Messages/PayloadMessage.h>

using namespace adaptone;
using namespace std;

PayloadMessage::PayloadMessage(uint32_t id, size_t payloadSize) :
    ProbeMessage(id, payloadSize + sizeof(m_payloadSize)), m_payloadSize(payloadSize)
{
}

PayloadMessage::~PayloadMessage()
{
}

void PayloadMessage::serialize(NetworkBufferView& buffer)
{
    *reinterpret_cast<uint32_t*>(buffer.data()) = boost::endian::native_to_big(m_payloadSize);

    NetworkBufferView view = buffer.view(sizeof(m_payloadSize));
    serializePayload(view);
}
