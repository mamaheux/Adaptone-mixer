#include <Communication/Messages/Input/ChangeAuxiliaryOutputVolumeMessage.h>

using namespace adaptone;
using namespace std;

constexpr size_t ChangeAuxiliaryOutputVolumeMessage::SeqId;

ChangeAuxiliaryOutputVolumeMessage::ChangeAuxiliaryOutputVolumeMessage() : ApplicationMessage(SeqId),
    m_channelId(0),
    m_gain(0)
{
}

ChangeAuxiliaryOutputVolumeMessage::ChangeAuxiliaryOutputVolumeMessage(size_t channelId, double gain) :
    ApplicationMessage(SeqId),
    m_channelId(channelId),
    m_gain(gain)
{
}

ChangeAuxiliaryOutputVolumeMessage::~ChangeAuxiliaryOutputVolumeMessage()
{
}
