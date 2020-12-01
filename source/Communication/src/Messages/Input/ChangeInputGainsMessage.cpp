#include <Communication/Messages/Input/ChangeInputGainsMessage.h>

using namespace adaptone;
using namespace std;

constexpr size_t ChangeInputGainsMessage::SeqId;

ChangeInputGainsMessage::ChangeInputGainsMessage() : ApplicationMessage(SeqId),
    m_gains()
{
}

ChangeInputGainsMessage::ChangeInputGainsMessage(vector<ChannelGain> gains) : ApplicationMessage(SeqId),
    m_gains(move(gains))
{
}

ChangeInputGainsMessage::~ChangeInputGainsMessage()
{
}
