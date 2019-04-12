#include <Communication/Messages/Input/ChangeMasterOutputEqGainsMessage.h>

using namespace adaptone;
using namespace std;

constexpr size_t ChangeMasterOutputEqGainsMessage::SeqId;

ChangeMasterOutputEqGainsMessage::ChangeMasterOutputEqGainsMessage() : ApplicationMessage(SeqId),
    m_gains()
{
}

ChangeMasterOutputEqGainsMessage::ChangeMasterOutputEqGainsMessage(const vector<double>& gains) :
    ApplicationMessage(SeqId),
    m_gains(gains)
{
}

ChangeMasterOutputEqGainsMessage::~ChangeMasterOutputEqGainsMessage()
{
}
