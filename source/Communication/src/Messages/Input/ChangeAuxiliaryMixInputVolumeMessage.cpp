#include <Communication/Messages/Input/ChangeAuxiliaryMixInputVolumeMessage.h>

using namespace adaptone;
using namespace std;

constexpr size_t ChangeAuxiliaryMixInputVolumeMessage::SeqId;

ChangeAuxiliaryMixInputVolumeMessage::ChangeAuxiliaryMixInputVolumeMessage() : ApplicationMessage(SeqId),
    m_channelId(0),
    m_auxiliaryId(0),
    m_gain(0)
{
}

ChangeAuxiliaryMixInputVolumeMessage::ChangeAuxiliaryMixInputVolumeMessage(size_t channelId,
    size_t auxiliaryId,
    double gain) :
    ApplicationMessage(SeqId),
    m_channelId(channelId),
    m_auxiliaryId(auxiliaryId),
    m_gain(gain)
{
}

ChangeAuxiliaryMixInputVolumeMessage::~ChangeAuxiliaryMixInputVolumeMessage()
{
}
