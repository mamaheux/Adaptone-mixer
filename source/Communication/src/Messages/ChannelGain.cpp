#include <Communication/Messages/ChannelGain.h>

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
