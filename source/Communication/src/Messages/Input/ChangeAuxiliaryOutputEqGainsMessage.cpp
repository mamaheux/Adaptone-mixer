#include <Communication/Messages/Input/ChangeAuxiliaryOutputEqGainsMessage.h>

using namespace adaptone;
using namespace std;

constexpr size_t ChangeAuxiliaryOutputEqGainsMessage::SeqId;

ChangeAuxiliaryOutputEqGainsMessage::ChangeAuxiliaryOutputEqGainsMessage() : ApplicationMessage(SeqId),
    m_auxiliaryId(0),
    m_gains()
{
}

ChangeAuxiliaryOutputEqGainsMessage::ChangeAuxiliaryOutputEqGainsMessage(size_t auxiliaryId,
    const std::vector<double>& gains) :
    ApplicationMessage(SeqId),
    m_auxiliaryId(auxiliaryId),
    m_gains(gains)
{
}

ChangeAuxiliaryOutputEqGainsMessage::~ChangeAuxiliaryOutputEqGainsMessage()
{
}
