#include <Communication/Messages/Input/ChangeMasterOutputVolumeMessage.h>

using namespace adaptone;
using namespace std;

constexpr size_t ChangeMasterOutputVolumeMessage::SeqId;

ChangeMasterOutputVolumeMessage::ChangeMasterOutputVolumeMessage() : ApplicationMessage(SeqId),
    m_gain(0)
{
}

ChangeMasterOutputVolumeMessage::ChangeMasterOutputVolumeMessage(double gain) : ApplicationMessage(SeqId),
    m_gain(gain)
{
}

ChangeMasterOutputVolumeMessage::~ChangeMasterOutputVolumeMessage()
{
}
