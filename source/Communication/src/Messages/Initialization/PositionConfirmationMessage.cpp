#include <Communication/Messages/Initialization/PositionConfirmationMessage.h>

using namespace adaptone;
using namespace std;

PositionConfirmationMessage::PositionConfirmationMessage() : ApplicationMessage(3),
    m_firstSymmetryPositions(),
    m_secondSymmetryPositions()
{
}

PositionConfirmationMessage::PositionConfirmationMessage(const vector<ConfigurationPosition>& firstSymmetryPositions,
    const vector<ConfigurationPosition>& secondSymmetryPositions) : ApplicationMessage(3),
    m_firstSymmetryPositions(firstSymmetryPositions),
    m_secondSymmetryPositions(secondSymmetryPositions)
{
}

PositionConfirmationMessage::~PositionConfirmationMessage()
{
}
