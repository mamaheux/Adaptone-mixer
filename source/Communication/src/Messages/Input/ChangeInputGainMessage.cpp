#include <Communication/Messages/Input/ChangeInputGainMessage.h>

using namespace adaptone;
using namespace std;

constexpr size_t ChangeInputGainMessage::SeqId;

ChangeInputGainMessage::ChangeInputGainMessage() : ApplicationMessage(SeqId),
    m_channelId(0),
    m_gain(0)
{
}

ChangeInputGainMessage::ChangeInputGainMessage(size_t channelId, double gain) : ApplicationMessage(SeqId),
    m_channelId(channelId),
    m_gain(gain)
{
}

ChangeInputGainMessage::~ChangeInputGainMessage()
{
}
