#include <Communication/Messages/Input/ChangeMasterOutputEqGainsMessage.h>

using namespace adaptone;
using namespace std;

constexpr size_t ChangeMasterOutputEqGainsMessage::SeqId;

ChangeMasterOutputEqGainsMessage::ChangeMasterOutputEqGainsMessage() : ApplicationMessage(SeqId),
    m_gains()
{
}

ChangeMasterOutputEqGainsMessage::ChangeMasterOutputEqGainsMessage(vector<double> gains) :
    ApplicationMessage(SeqId),
    m_gains(move(gains))
{
}

ChangeMasterOutputEqGainsMessage::~ChangeMasterOutputEqGainsMessage()
{
}
