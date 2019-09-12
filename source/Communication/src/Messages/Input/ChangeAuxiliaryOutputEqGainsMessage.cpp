#include <Communication/Messages/Input/ChangeAuxiliaryOutputEqGainsMessage.h>

using namespace adaptone;
using namespace std;

constexpr size_t ChangeAuxiliaryOutputEqGainsMessage::SeqId;

ChangeAuxiliaryOutputEqGainsMessage::ChangeAuxiliaryOutputEqGainsMessage() : ApplicationMessage(SeqId),
    m_channelId(0),
    m_gains()
{
}

ChangeAuxiliaryOutputEqGainsMessage::ChangeAuxiliaryOutputEqGainsMessage(size_t channelId,
    const vector<double>& gains) :
    ApplicationMessage(SeqId),
    m_channelId(channelId),
    m_gains(gains)
{
}

ChangeAuxiliaryOutputEqGainsMessage::~ChangeAuxiliaryOutputEqGainsMessage()
{
}
