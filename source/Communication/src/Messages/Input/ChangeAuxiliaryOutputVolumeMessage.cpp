#include <Communication/Messages/Input/ChangeAuxiliaryOutputVolumeMessage.h>

using namespace adaptone;
using namespace std;

constexpr size_t ChangeAuxiliaryOutputVolumeMessage::SeqId;

ChangeAuxiliaryOutputVolumeMessage::ChangeAuxiliaryOutputVolumeMessage() : ApplicationMessage(SeqId),
    m_auxiliaryId(0),
    m_gain(0)
{
}

ChangeAuxiliaryOutputVolumeMessage::ChangeAuxiliaryOutputVolumeMessage(size_t auxiliaryId, double gain) :
    ApplicationMessage(SeqId),
    m_auxiliaryId(auxiliaryId),
    m_gain(gain)
{
}

ChangeAuxiliaryOutputVolumeMessage::~ChangeAuxiliaryOutputVolumeMessage()
{
}
