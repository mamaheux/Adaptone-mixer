#include <Communication/Messages/Input/ListenProbeMessage.h>

using namespace adaptone;
using namespace std;

constexpr size_t ListenProbeMessage::SeqId;

ListenProbeMessage::ListenProbeMessage() : ApplicationMessage(SeqId), m_probeId(0)
{
}

ListenProbeMessage::ListenProbeMessage(uint32_t probeId) : ApplicationMessage(SeqId), m_probeId(probeId)
{
}

ListenProbeMessage::~ListenProbeMessage()
{
}
