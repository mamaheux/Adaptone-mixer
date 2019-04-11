#include <Communication/Messages/Initialization/PositionConfirmationMessage.h>

using namespace adaptone;
using namespace std;

constexpr size_t PositionConfirmationMessage::SeqId;

PositionConfirmationMessage::PositionConfirmationMessage() : ApplicationMessage(SeqId),
    m_firstSymmetryPositions(),
    m_secondSymmetryPositions()
{
}

PositionConfirmationMessage::PositionConfirmationMessage(const vector<ConfigurationPosition>& firstSymmetryPositions,
    const vector<ConfigurationPosition>& secondSymmetryPositions) : ApplicationMessage(SeqId),
    m_firstSymmetryPositions(firstSymmetryPositions),
    m_secondSymmetryPositions(secondSymmetryPositions)
{
}

PositionConfirmationMessage::~PositionConfirmationMessage()
{
}
