#include <Communication/Messages/Input/ChangeInputEqGainsMessage.h>

using namespace adaptone;
using namespace std;

constexpr size_t ChangeInputEqGainsMessage::SeqId;

ChangeInputEqGainsMessage::ChangeInputEqGainsMessage() : ApplicationMessage(SeqId),
    m_channelId(0),
    m_gains()
{
}

ChangeInputEqGainsMessage::ChangeInputEqGainsMessage(size_t channelId, const vector<double>& gains) :
    ApplicationMessage(SeqId),
    m_channelId(channelId),
    m_gains(gains)
{
}

ChangeInputEqGainsMessage::~ChangeInputEqGainsMessage()
{
}
