#include <Communication/Messages/Input/ChangeInputGainsMessage.h>

using namespace adaptone;
using namespace std;

constexpr size_t ChangeInputGainsMessage::SeqId;

ChangeInputGainsMessage::ChangeInputGainsMessage() : ApplicationMessage(SeqId),
    m_gains()
{
}

ChangeInputGainsMessage::ChangeInputGainsMessage(const std::vector<double>& gains) : ApplicationMessage(SeqId),
    m_gains(gains)
{
}

ChangeInputGainsMessage::~ChangeInputGainsMessage()
{
}
