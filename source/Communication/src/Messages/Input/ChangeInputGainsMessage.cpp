#include <Communication/Messages/Input/ChangeInputGainsMessage.h>

using namespace adaptone;
using namespace std;

ChannelGain::ChannelGain() : m_channelId(0), m_gain(0)
{
}

ChannelGain::ChannelGain(size_t channelId, double gain) : m_channelId(channelId), m_gain(gain)
{
}

ChannelGain::~ChannelGain()
{
}

constexpr size_t ChangeInputGainsMessage::SeqId;

ChangeInputGainsMessage::ChangeInputGainsMessage() : ApplicationMessage(SeqId),
    m_gains()
{
}

ChangeInputGainsMessage::ChangeInputGainsMessage(const vector<ChannelGain>& gains) : ApplicationMessage(SeqId),
    m_gains(gains)
{
}

ChangeInputGainsMessage::~ChangeInputGainsMessage()
{
}
