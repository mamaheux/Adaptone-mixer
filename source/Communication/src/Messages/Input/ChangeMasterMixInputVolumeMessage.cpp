#include <Communication/Messages/Input/ChangeMasterMixInputVolumeMessage.h>

using namespace adaptone;
using namespace std;

constexpr size_t ChangeMasterMixInputVolumeMessage::SeqId;

ChangeMasterMixInputVolumeMessage::ChangeMasterMixInputVolumeMessage() : ApplicationMessage(SeqId),
    m_channelId(0),
    m_gain(0)
{
}

ChangeMasterMixInputVolumeMessage::ChangeMasterMixInputVolumeMessage(size_t channelId, double gain) :
    ApplicationMessage(SeqId),
    m_channelId(channelId),
    m_gain(gain)
{
}

ChangeMasterMixInputVolumeMessage::~ChangeMasterMixInputVolumeMessage()
{
}
