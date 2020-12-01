#include <Communication/Messages/Initialization/PositionConfirmationMessage.h>

using namespace adaptone;
using namespace std;

constexpr size_t PositionConfirmationMessage::SeqId;

PositionConfirmationMessage::PositionConfirmationMessage() : ApplicationMessage(SeqId),
    m_firstSymmetryPositions(),
    m_secondSymmetryPositions()
{
}

PositionConfirmationMessage::PositionConfirmationMessage(vector<ConfigurationPosition> firstSymmetryPositions,
    vector<ConfigurationPosition> secondSymmetryPositions) : ApplicationMessage(SeqId),
    m_firstSymmetryPositions(move(firstSymmetryPositions)),
    m_secondSymmetryPositions(move(secondSymmetryPositions))
{
}

PositionConfirmationMessage::~PositionConfirmationMessage()
{
}
