#include <Communication/Messages/ApplicationMessage.h>

using namespace adaptone;
using namespace std;

ApplicationMessage::ApplicationMessage(size_t seqId) : m_seqId(seqId)
{
}

ApplicationMessage::~ApplicationMessage()
{
}
