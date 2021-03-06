#include <Communication/Messages/Input/ChangeMasterMixInputVolumesMessage.h>

using namespace adaptone;
using namespace std;

constexpr size_t ChangeMasterMixInputVolumesMessage::SeqId;

ChangeMasterMixInputVolumesMessage::ChangeMasterMixInputVolumesMessage() : ApplicationMessage(SeqId)
{
}

ChangeMasterMixInputVolumesMessage::ChangeMasterMixInputVolumesMessage(vector<ChannelGain> gains) :
    ApplicationMessage(SeqId),
    m_gains(move(gains))
{
}

ChangeMasterMixInputVolumesMessage::~ChangeMasterMixInputVolumesMessage()
{
}
