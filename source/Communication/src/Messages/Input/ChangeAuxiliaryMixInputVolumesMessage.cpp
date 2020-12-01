#include <Communication/Messages/Input/ChangeAuxiliaryMixInputVolumesMessage.h>

using namespace adaptone;
using namespace std;

constexpr size_t ChangeAuxiliaryMixInputVolumesMessage::SeqId;

ChangeAuxiliaryMixInputVolumesMessage::ChangeAuxiliaryMixInputVolumesMessage() : ApplicationMessage(SeqId),
    m_auxiliaryChannelId(0)
{
}

ChangeAuxiliaryMixInputVolumesMessage::ChangeAuxiliaryMixInputVolumesMessage(size_t auxiliaryChannelId,
    vector<ChannelGain> gains) :
    ApplicationMessage(SeqId),
    m_auxiliaryChannelId(auxiliaryChannelId),
    m_gains(move(gains))
{
}

ChangeAuxiliaryMixInputVolumesMessage::~ChangeAuxiliaryMixInputVolumesMessage()
{
}
