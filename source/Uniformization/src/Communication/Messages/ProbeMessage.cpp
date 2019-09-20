#include <Uniformization/Communication/Messages/ProbeMessage.h>

using namespace adaptone;
using namespace std;

ProbeMessage::ProbeMessage(uint32_t id, size_t payloadSize) : m_id(id), m_fullSize(payloadSize + sizeof(m_id))
{
}

ProbeMessage::~ProbeMessage()
{
}
